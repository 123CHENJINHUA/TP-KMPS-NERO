"""RealSense RGB-D in a rectified COLOR optical frame; passive robot snapshots."""
from .common import camera_model, checked, output_path, new_directory, write_json, utc_now, digest
from ..bootstrap import read_json
from ..data import config, physical_signature
from ..kinematics import fk_matrix
import time
import numpy as np


def project_depth_points(points_color, K, shape):
    """Project metric depth points with z-buffering. No resizing of raw depth."""
    height, width = shape
    points = np.asarray(points_color, float).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > .01)
    points = points[valid]
    uvw = points@np.asarray(K).T
    xy = np.rint(uvw[:, :2]/uvw[:, 2:]).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    result = np.full(height*width, np.inf, np.float32)
    np.minimum.at(result, xy[valid, 1]*width+xy[valid, 0], points[valid, 2])
    result = result.reshape(height, width)
    # Small splats compensate for lower native depth resolution; nearest surface wins.
    import cv2
    result = cv2.erode(result, np.ones((3, 3), np.uint8))
    result[~np.isfinite(result)] = 0
    return result


class RGBDCamera:
    def __init__(self, serial, width, height, fps, intrinsics=None):
        import cv2
        import pyrealsense2 as rs
        self.rs, self.pipeline, self.started = rs, rs.pipeline(), False
        settings = rs.config()
        settings.enable_device(serial)
        settings.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        settings.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(settings)
            self.started = True
            device = profile.get_device()
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            raw = color.get_intrinsics()
            actual_serial = device.get_info(rs.camera_info.serial_number)
            if actual_serial != serial:
                raise ValueError('相机实际序列号与请求不一致')
            raw_K = [[raw.fx, 0, raw.ppx], [0, raw.fy, raw.ppy], [0, 0, 1]]
            if intrinsics:
                self.source_model = read_json(intrinsics)
                if (self.source_model.get('serial'), self.source_model.get('width'), self.source_model.get('height'),
                    self.source_model.get('fps')) != (serial, width, height, fps):
                    raise ValueError('内参的相机序列号、分辨率或 fps 与实际流不匹配')
                self.intrinsics_sha256 = digest(intrinsics)
            else:
                model = str(raw.model)
                kind = 'none' if not np.any(raw.coeffs) else ('opencv_brown' if model.endswith('.brown_conrady') else model)
                self.source_model = dict(serial=serial, width=width, height=height, fps=fps, frame='color_optical',
                                         K=raw_K, distortion_model=kind, distortion_coefficients=list(raw.coeffs),
                                         source='factory')
                self.intrinsics_sha256 = None
            self.K, distortion = camera_model(self.source_model)
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(self.K, distortion, np.eye(3), self.K,
                                                               (width, height), cv2.CV_32FC1)
            self.model = dict(serial=serial, width=width, height=height, fps=fps, frame='color_optical',
                              K=self.K.tolist(), distortion_model='none', distortion_coefficients=[0.]*5,
                              rectified=True, registered_depth=True)
            extrinsic = profile.get_stream(rs.stream.depth).get_extrinsics_to(color)
            self.T_color_depth = np.eye(4)
            self.T_color_depth[:3, :3] = np.array(extrinsic.rotation).reshape(3, 3, order='F')
            self.T_color_depth[:3, 3] = extrinsic.translation
            checked(self.T_color_depth)
            self.points = rs.pointcloud()
            self.depth_scale_m = device.first_depth_sensor().get_depth_scale()
            self.device = dict(serial=serial, usb_type=device.get_info(rs.camera_info.usb_type_descriptor))
            for _ in range(30):
                self.pipeline.wait_for_frames(5000)
            source = ('camera_config.json 中直接填写的内参' if self.source_model.get('source') == 'inline_parameters'
                      else f'文件 {intrinsics}' if intrinsics else '相机出厂参数（当前 RGB 流）')
            print(f'RGB {serial}: {width}×{height}@{fps}；内参来源：{source}；'
                  f'fx={self.K[0, 0]:.3f}, fy={self.K[1, 1]:.3f}, '
                  f'cx={self.K[0, 2]:.3f}, cy={self.K[1, 2]:.3f}', flush=True)
        except BaseException:
            self.close()
            raise

    def frame(self):
        import cv2
        frames = self.pipeline.wait_for_frames(5000)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise ValueError('缺少配对的 RGB 或深度帧')
        if str(color.get_frame_timestamp_domain()) != str(depth.get_frame_timestamp_domain()):
            raise ValueError('RGB/深度使用不同时间戳域，无法检查配对')
        skew = abs(color.get_timestamp()-depth.get_timestamp())
        if skew > 1000/self.model['fps']:
            raise ValueError(f'RGB 与深度帧不同步：{skew:.1f}ms')
        cloud = self.points.calculate(depth)
        vertices = np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(-1, 3).copy()
        valid = vertices[:, 2] > 0
        vertices = vertices[valid]@self.T_color_depth[:3, :3].T+self.T_color_depth[:3, 3]
        registered = project_depth_points(vertices, self.K, (self.model['height'], self.model['width']))
        raw_bgr = np.asanyarray(color.get_data())
        rectified = cv2.remap(raw_bgr, self.map_x, self.map_y, cv2.INTER_LINEAR)
        return cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB), registered, dict(
            host_unix_s=time.time(), host_monotonic_s=time.monotonic(), frame_number=color.get_frame_number(),
            device_timestamp_ms=color.get_timestamp(), timestamp_domain=str(color.get_frame_timestamp_domain()),
            rgb_depth_skew_ms=skew)

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


def capture(args, reader=None):
    """Capture with an optional existing passive reader owned by the caller."""
    from ..hardware import Reader
    import cv2
    cfg = config(args.config, args.can)
    if args.frames < 3:
        raise ValueError('至少采集 3 帧用于静态稳定性检查')
    camera, owns_reader = None, False
    try:
        if not args.no_robot:
            if reader is None:
                reader = Reader(cfg)
                owns_reader = True
            reader.wait_ready(False)
        elif reader is not None:
            raise ValueError('no_robot 不能同时提供机器人 Reader')
        camera = RGBDCamera(args.serial, args.width, args.height, args.fps, args.intrinsics)
        print('保持物体、相机和从臂停稳；只采集图像和从臂反馈，不发运动指令。', flush=True)
        rgb, depth, frames, all_q = [], [], [], []
        for _ in range(args.frames):
            before = reader.sample(False) if reader else None
            color, distance, meta = camera.frame()
            after = reader.sample(False) if reader else None
            if reader:
                if np.max(abs(after['joints_rad']-before['joints_rad'])) > .005:
                    raise ValueError('拍照期间从臂正在运动，请停稳后重新采集')
                all_q.extend([before['joints_rad'], after['joints_rad']])
                meta['robot'] = dict(joints_rad=after['joints_rad'].tolist(),
                                      T_base_flange=fk_matrix(after['joints_rad']).tolist(),
                                      joint_source='measured', channel=cfg['channel'],
                                      signature=physical_signature(cfg), bracket_joint_delta_rad=float(np.max(abs(after['joints_rad']-before['joints_rad']))))
            rgb.append(color)
            depth.append(distance)
            frames.append(meta)
        if all_q and np.max(np.ptp(all_q, axis=0)) > .005:
            raise ValueError('多帧采集期间从臂发生移动；本流程要求静态物体与静态相机')
        output = new_directory(output_path(args.out))
        with (output/'rgbd.npz').open('xb') as stream:
            np.savez_compressed(stream, rgb=np.array(rgb), depth_m=np.array(depth), K=camera.K)
        write_json(output/'capture.json', dict(schema_version=1, kind='rgbd_capture', synthetic=False,
              created_utc=utc_now(), camera=camera.model, source_camera=camera.source_model,
              intrinsics_sha256=camera.intrinsics_sha256, device=camera.device, config=cfg,
              T_color_depth=camera.T_color_depth.tolist(), depth_scale_m=camera.depth_scale_m,
              frames=frames, rgbd_sha256=digest(output/'rgbd.npz'), static_capture=True))
        cv2.imwrite(str(output/'rgb.png'), cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output/'depth_mm.png'), np.clip(depth[0]*1000, 0, 65535).astype(np.uint16))
        print(f'采集完成：{output}；RGB/深度均位于去畸变 COLOR 像素坐标，深度单位 m。')
        return output
    finally:
        if camera:
            camera.close()
        if owns_reader and reader:
            reader.close()
