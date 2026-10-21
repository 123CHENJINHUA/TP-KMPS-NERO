"""D435i COLOR stream. Hand-eye results refer to the color optical frame."""
from bootstrap import read_json
import numpy as np
import time
import pyrealsense2 as rs
from transforms import checked


def device_info(device):
    fields = {'name': rs.camera_info.name, 'serial': rs.camera_info.serial_number,
              'firmware': rs.camera_info.firmware_version,
              'usb_type': rs.camera_info.usb_type_descriptor, 'physical_port': rs.camera_info.physical_port}
    return {name: device.get_info(key) if device.supports(key) else None for name, key in fields.items()}


def video_modes(device, stream, pixel_format):
    modes = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if (profile.stream_type() == stream and profile.format() == pixel_format and
                    profile.is_video_stream_profile()):
                video = profile.as_video_stream_profile()
                modes.add((video.width(), video.height(), profile.fps()))
    return [{'width': w, 'height': h, 'fps': fps} for w, h, fps in sorted(modes)]


def list_devices(include_profiles=False):
    result = []
    for device in rs.context().query_devices():
        info = device_info(device)
        if include_profiles:
            info['color_bgr8_modes'] = video_modes(device, rs.stream.color, rs.format.bgr8)
        result.append(info)
    return result


def require_video_mode(device, stream, pixel_format, width, height, fps, label):
    modes = video_modes(device, stream, pixel_format)
    if {'width': width, 'height': height, 'fps': fps} in modes:
        return
    info = device_info(device)
    same_size = sorted(m['fps'] for m in modes if (m['width'], m['height']) == (width, height))
    available = ', '.join(f'{m["width"]}x{m["height"]}@{m["fps"]}' for m in modes) or '无可用模式'
    hint = (f'同分辨率可用 fps={same_size}。' if same_size else '')
    if stream == rs.stream.color and same_size:
        hint += f' 可明确指定 --width {width} --height {height} --fps {max(same_size)}。'
    elif stream == rs.stream.depth:
        hint += ' 可取消 --with-depth，或选择 RGB/深度都支持的帧率。'
    raise ValueError(f'相机 {info["serial"]}（USB {info["usb_type"]}）不支持 '
                     f'{label} {width}x{height}@{fps} fps。{hint}\n当前支持: {available}\n'
                     '用 doctor.py --camera-profiles 查看；需要高速模式时检查 USB3 接口和数据线。')


def opencv_intrinsics(raw):
    result = dict(raw)
    model, coeff = raw['distortion_model'], np.array(raw['distortion_coefficients'])
    if model.endswith('.none') or not np.any(coeff):
        result.update(distortion_model='none', distortion_coefficients=[0.] * 5)
    elif model.endswith('.brown_conrady'):
        result['distortion_model'] = 'opencv_brown'
    else:
        raise ValueError(f'RealSense {model} 不能直接当作 OpenCV Brown；先采内参数据并运行 calibrate_intrinsics.py')
    return result


def extrinsic_matrix(extrinsic):
    T = np.eye(4)
    # librealsense rs2_extrinsics stores rotation COLUMN-MAJOR.
    T[:3, :3] = np.array(extrinsic.rotation).reshape(3, 3, order='F')
    T[:3, 3] = extrinsic.translation
    return checked(T)


class RealSenseCamera:
    def __init__(self, serial=None, width=1280, height=720, fps=30, intrinsics=None, depth=False):
        if min(width, height, fps) <= 0:
            raise ValueError('相机 width、height、fps 必须为正')
        connected = list(rs.context().query_devices())
        devices = [device_info(d) for d in connected]
        if not devices:
            raise RuntimeError('未发现 RealSense；检查 USB3 连接以及是否被其他程序占用')
        if serial is None and len(devices) != 1:
            raise ValueError('连接了多个相机，必须用 --serial 指定；先 doctor.py 查看序列号')
        serial = serial or devices[0]['serial']
        if serial not in [d['serial'] for d in devices]:
            raise ValueError(f'未连接序列号 {serial}')
        device = connected[[d['serial'] for d in devices].index(serial)]
        self.device_info = device_info(device)
        require_video_mode(device, rs.stream.color, rs.format.bgr8, width, height, fps, 'RGB BGR8')
        if depth:
            require_video_mode(device, rs.stream.depth, rs.format.z16, 640, 480, fps, '深度 Z16')
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.started = False
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        try:
            try:
                if not config.can_resolve(rs.pipeline_wrapper(self.pipeline)):
                    raise RuntimeError('请求的流组合无法解析')
                profile = self.pipeline.start(config)
            except RuntimeError as exc:
                raise RuntimeError(f'RealSense 相机 {serial} 启动失败：RGB {width}x{height}@{fps} fps，'
                                   f'USB {self.device_info["usb_type"]}，with_depth={depth}。\n'
                                   '检查其他采集程序是否占用相机、USB 带宽和流组合；'
                                   'doctor.py --camera-profiles 可查看支持模式。\n'
                                   f'SDK 原始错误: {exc}') from exc
            self.started = True
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            info = color.get_intrinsics()
            self.raw = {'serial': serial, 'width': info.width, 'height': info.height, 'fps': color.fps(),
                        'frame': 'color_optical', 'K': [[info.fx, 0, info.ppx], [0, info.fy, info.ppy], [0, 0, 1]],
                        'distortion_model': str(info.model), 'distortion_coefficients': list(info.coeffs)}
            if depth:
                dep = profile.get_stream(rs.stream.depth)
                self.raw['T_color_depth'] = extrinsic_matrix(dep.get_extrinsics_to(color)).tolist()
                self.raw['depth_scale_m'] = profile.get_device().first_depth_sensor().get_depth_scale()
            self.model = None
            self.model_error = None
            if intrinsics:
                custom = read_json(intrinsics)
                if (custom.get('serial'), custom.get('width'), custom.get('height'), custom.get('frame')) != (serial, width, height, 'color_optical'):
                    raise ValueError('自定义内参的 serial、width、height、frame 必须与当前 RGB 流一致')
                if custom.get('passed') is False:
                    raise ValueError('这组内参没有通过留出验证，请重新采集内参数据')
                from vision import camera_matrix
                camera_matrix(custom)
                self.model = custom
            else:
                try:
                    self.model = opencv_intrinsics(self.raw)
                except ValueError as exc:
                    self.model_error = str(exc)
            for _ in range(30):
                self.frame()
        except BaseException:
            self.close()
            raise

    def frame(self, flush=False):
        if flush:
            # Discard buffered frames; receive a new frame after a capture request.
            for _ in range(3):
                self.pipeline.wait_for_frames(5000)
        frames = self.pipeline.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError('RGB 帧缺失')
        meta = {'host_monotonic_s': time.monotonic(), 'host_unix_s': time.time(),
                'frame_number': color.get_frame_number(), 'device_timestamp_ms': color.get_timestamp(),
                'device_timestamp_domain': str(color.get_frame_timestamp_domain())}
        return np.asanyarray(color.get_data()).copy(), meta

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
