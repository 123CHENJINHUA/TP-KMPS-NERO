"""Meshy's documented REST API. Never retry a potentially billed POST."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, urlopen

from ..common import ROOT, new_directory, output_path, read_json, utc_now

API = 'https://api.meshy.ai/openapi/v1'
MIME = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
ENDPOINTS = ('image-to-3d', 'multi-image-to-3d')
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_MODEL_BYTES = 1024 * 1024 * 1024


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    """Atomic updates to this client's local job state, never outside vision/."""
    path = output_path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward the API bearer token to another host.


class Client:
    def __init__(self):
        self.key = os.environ.get('MESHY_API_KEY', '').strip()
        if not self.key or '\n' in self.key or '\r' in self.key:
            raise ValueError('请在终端设置 MESHY_API_KEY；不要把密钥写进代码或发到聊天中。')
        self.opener = build_opener(NoRedirect())

    def request(self, endpoint, task_id=None, payload=None):
        if endpoint not in ENDPOINTS or (task_id is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', task_id)):
            raise ValueError('无效的 Meshy 任务类型/编号')
        url = API+'/'+endpoint + ('/'+task_id if task_id else '')
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(url, data=body, method='POST' if body is not None else 'GET',
                      headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json',
                               'User-Agent': 'real-tpkmp-vision/1'})
        try:
            with self.opener.open(req, timeout=45) as response:
                return json.loads(response.read(8*1024*1024))
        except HTTPError as exc:
            explanation = {401: 'API key 无效', 402: '账户积分不足', 429: '请求频率受限'}.get(exc.code, '服务端拒绝请求')
            # Error bodies can contain a submitted data URI; never echo them.
            raise RuntimeError(f'Meshy HTTP {exc.code}：{explanation}。POST 不会自动重试；先在网页确认是否已创建任务。') from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise RuntimeError('Meshy 网络/响应错误。若发生在提交阶段，请先查看网页任务列表，避免重复扣积分。') from None


def validate_images(paths):
    import cv2
    import numpy as np
    if not 1 <= len(paths) <= 4:
        raise ValueError('Meshy 需要同一物体的 1–4 张不同视角照片')
    images, seen = [], set()
    for filename in paths:
        path = Path(filename).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in MIME:
            raise ValueError(f'需要存在的 JPG/PNG/WebP 图片：{path}')
        if not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
            raise ValueError(f'每张照片必须在 0–20 MB 之间：{path}')
        raw = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f'不能解码图片：{path}')
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum in seen:
            raise ValueError('重复输入同一张照片不会提供多视图；请使用不同视角照片')
        seen.add(checksum)
        images.append(dict(source=str(path), sha256=checksum, width=int(decoded.shape[1]),
                           height=int(decoded.shape[0]), mime=MIME[path.suffix.lower()]))
    return images


def request_body(manifest, directory):
    images = []
    for entry in manifest['images']:
        path = output_path(directory/entry['snapshot'])
        if sha256(path) != entry['sha256']:
            raise ValueError('提交前的照片副本被修改')
        images.append('data:'+entry['mime']+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii'))
    body = dict(manifest['parameters'])
    body['image_url' if len(images) == 1 else 'image_urls'] = images[0] if len(images) == 1 else images
    return body


def prepare(args):
    images = validate_images(args.images)
    directory = new_directory(output_path(args.out))
    (directory/'images').mkdir()
    for i, entry in enumerate(images):
        source = Path(entry['source'])
        snapshot = Path('images')/f'{i+1:02d}{source.suffix.lower()}'
        shutil.copyfile(source, directory/snapshot)
        entry['snapshot'] = str(snapshot)
    parameters = dict(ai_model=args.ai_model, geometry_resolution=args.geometry_resolution,
                      should_texture=args.texture, enable_pbr=False, auto_size=False,
                      image_enhancement=False, should_remesh=args.polycount is not None,
                      target_formats=['stl', 'glb'])
    if args.polycount is not None:
        parameters['target_polycount'] = args.polycount
    manifest = dict(schema_version=1, kind='meshy_generation', created_utc=utc_now(),
                    endpoint=ENDPOINTS[0 if len(images) == 1 else 1],
                    images=images, parameters=parameters, task_id=None, dry_run=bool(args.dry_run))
    save_json(directory/'request.json', manifest)  # No bearer token or base64 image dump.
    return directory, manifest


def read_job(directory):
    directory = output_path(directory)
    manifest = read_json(directory/'request.json')
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != 'meshy_generation'
            or manifest.get('endpoint') not in ENDPOINTS):
        raise ValueError('不是本工具创建的 Meshy 任务目录')
    return directory, manifest


def download(url, destination):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('只下载 Meshy 返回的 HTTPS 模型链接')
    destination = output_path(destination)
    temporary = destination.with_name(destination.name+'.part')
    try:
        # Download hosts receive no API key; signed model URLs are supplied by Meshy.
        with urlopen(Request(url, headers={'User-Agent': 'real-tpkmp-vision/1'}), timeout=45) as response:
            if urlsplit(response.url).scheme != 'https':
                raise ValueError('模型下载不允许降级到 HTTP')
            total = 0
            with temporary.open('wb') as stream:
                for block in iter(lambda: response.read(1024*1024), b''):
                    total += len(block)
                    if total > MAX_MODEL_BYTES:
                        raise ValueError('模型超过 1 GiB，停止下载；请在网页简化网格后导入')
                    stream.write(block)
        if not total:
            raise ValueError('下载得到空模型')
        temporary.replace(destination)
    except (HTTPError, URLError, TimeoutError, OSError):
        raise RuntimeError('模型下载失败；用 resume 重取下载链接，不会创建新的付费任务。') from None
    return dict(path=destination.name, sha256=sha256(destination), bytes=total)


def poll(directory, manifest, client, wait_seconds, poll_seconds):
    if not manifest.get('task_id'):
        raise ValueError('任务尚未提交；使用 submit。提交结果不明时先查 Meshy 网页，不要重复提交。')
    deadline = time.monotonic()+wait_seconds
    previous = None
    while True:
        task = client.request(manifest['endpoint'], task_id=manifest['task_id'])
        if task.get('id') != manifest['task_id']:
            raise ValueError('Meshy 返回的任务编号不匹配')
        save_json(directory/'task.json', task)
        state = task.get('status')
        progress = (state, task.get('progress'))
        if progress != previous:
            print(f'Meshy {state}：{task.get("progress", 0)}%（任务 {manifest["task_id"]}）', flush=True)
            previous = progress
        if state == 'SUCCEEDED':
            old = read_json(directory/'downloads.json') if (directory/'downloads.json').exists() else {}
            downloaded = {}
            for fmt in ('stl', 'glb'):
                url = task.get('model_urls', {}).get(fmt)
                if not url:
                    continue
                filename = directory/('model.'+fmt)
                if filename.exists():
                    if fmt not in old or sha256(filename) != old[fmt]['sha256']:
                        raise ValueError(f'{filename} 已存在且没有匹配的下载校验值；不会覆盖')
                    downloaded[fmt] = old[fmt]
                else:
                    downloaded[fmt] = download(url, filename)
                    save_json(directory/'downloads.json', {**old, **downloaded})
            if not downloaded:
                raise ValueError('任务成功但没有 STL/GLB 下载链接，请查看 task.json')
            print(f'模型已下载至 {directory}；尚未校正尺寸或绑定抓取关系。')
            return task
        if state in ('FAILED', 'CANCELED'):
            raise RuntimeError(f'Meshy 任务 {state}；详见 {directory / "task.json"}。不会自动重新创建任务。')
        if state not in ('PENDING', 'IN_PROGRESS'):
            raise RuntimeError(f'未知的 Meshy 状态：{state}；保留任务以供 resume。')
        if time.monotonic() >= deadline:
            print(f'本地等待结束，云端任务可能仍在运行。继续：bash real_tpkmp/vision/meshy.sh resume --run {directory}')
            return task
        time.sleep(min(poll_seconds, max(0., deadline-time.monotonic())))


def submit(directory, manifest, args, client=None):
    if manifest.get('task_id'):
        return poll(directory, manifest, client or Client(), args.wait_seconds, args.poll_seconds)
    if (directory/'submission.json').exists():
        raise ValueError('已有提交尝试记录；请先在 Meshy 网页核对，找到任务编号后用 resume --task-id，避免重复扣费。')
    client = client or Client()
    body = request_body(manifest, directory)
    # Write before the billed POST so a network timeout cannot silently resubmit.
    save_json(directory/'submission.json', dict(attempted_utc=utc_now(), outcome='unknown_until_task_id_is_saved'))
    reply = client.request(manifest['endpoint'], payload=body)
    task_id = reply.get('result')
    if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', task_id):
        raise RuntimeError('提交响应缺少任务编号；请在网页核对后 resume --task-id，不要重复 generate。')
    manifest.update(task_id=task_id, dry_run=False)
    save_json(directory/'request.json', manifest)
    return poll(directory, manifest, client, args.wait_seconds, args.poll_seconds)


def generate(args):
    client = None if args.dry_run else Client()
    directory, manifest = prepare(args)
    print(f'{len(manifest["images"])} 张照片 → {manifest["endpoint"]}；模型 {args.ai_model}；目录 {directory}')
    if args.dry_run:
        print('仅完成本地检查，没有上传、调用 API 或扣积分。使用 submit --run 提交此目录。')
        return
    return submit(directory, manifest, args, client)


def resume(args):
    directory, manifest = read_job(args.run)
    if args.task_id:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', args.task_id):
            raise ValueError('无效任务编号')
        if manifest.get('task_id') and manifest['task_id'] != args.task_id:
            raise ValueError('本地已经记录了另一任务编号；不会替换')
        manifest['task_id'] = args.task_id
        save_json(directory/'request.json', manifest)
    return poll(directory, manifest, Client(), args.wait_seconds, args.poll_seconds)
