"""Deterministic, bounded-memory overlay primitives; no network/model downloads."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ENGINE_VERSION = 'overlay-1.2.1'
cv2.setNumThreads(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique sibling temp prevents races between independent invocations.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as f:
        temp = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def run(args, timeout=120):
    result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit {result.returncode}: ' +
                           result.stderr.decode('utf-8', 'replace')[-6000:])
    return result.stdout


class PipeProcess:
    """Deadline kills child even when pipe read/write blocks; stderr cannot fill a pipe."""
    def __init__(self, args, timeout=600, reader=False, writer=False):
        self.err = tempfile.TemporaryFile()
        self.expired = False
        try:
            self.p = subprocess.Popen([str(x) for x in args],
                                      stdin=subprocess.PIPE if writer else subprocess.DEVNULL,
                                      stdout=subprocess.PIPE if reader else subprocess.DEVNULL,
                                      stderr=self.err, bufsize=0)
        except Exception:
            self.err.close()
            raise
        self.timer = threading.Timer(timeout, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        self.expired = True
        if self.p.poll() is None:
            self.p.kill()

    def finish(self):
        if self.p.stdin and not self.p.stdin.closed:
            self.p.stdin.close()
        self.p.wait(timeout=15)
        self.timer.cancel()
        self.err.seek(0)
        error = self.err.read().decode('utf-8', 'replace')[-6000:]
        if self.expired or self.p.returncode:
            raise RuntimeError(f'FFmpeg {"timeout" if self.expired else "exit " + str(self.p.returncode)}: {error}')

    def close(self):
        self.timer.cancel()
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait(timeout=15)
        for stream in (self.p.stdin, self.p.stdout):
            if stream and not stream.closed:
                stream.close()
        self.err.close()


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', path], timeout=60))


def rate(value, fallback=30.):
    try:
        a, *b = str(value).split('/')
        r = float(a) / (float(b[0]) if b else 1)
        return r if math.isfinite(r) and r > 0 else fallback
    except (ValueError, ZeroDivisionError):
        return fallback


def image_rgb(path):
    with Image.open(path) as im:
        # First frame for animated image formats, explicitly documented in plan.
        im = ImageOps.exif_transpose(im).convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return np.asarray(bg.convert('RGB')).copy()


def media_info(path, image_duration, quality):
    try:
        with Image.open(path) as im:
            original_format = im.format
            animated = bool(getattr(im, 'is_animated', False))
        rgb = image_rgb(path)
        h, w = rgb.shape[:2]
        info = dict(kind='image', width=w, height=h, duration=image_duration,
                    fps=30., audio=None, rotation=0, image_format=original_format,
                    animated_first_frame_only=animated, exif_transposed=True)
    except (OSError, ValueError, Image.DecompressionBombError):
        data = probe(path)
        stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'
                       and not s.get('disposition', {}).get('attached_pic')), None)
        if stream is None:
            raise ValueError('输入不是可解码的图片或视频')
        w, h = int(stream['width']), int(stream['height'])
        rotation = float(stream.get('tags', {}).get('rotate', 0) or 0)
        for side in stream.get('side_data_list', []):
            if 'rotation' in side:
                rotation = float(side['rotation'])
        if abs(rotation) % 180 > 45 and abs(rotation) % 180 < 135:
            w, h = h, w
        d = stream.get('duration') or data.get('format', {}).get('duration')
        duration = float(d) if d not in (None, 'N/A') else 0
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('无法确定有限的视频时长')
        source_fps = rate(stream.get('avg_frame_rate'), rate(stream.get('r_frame_rate')))
        fps = min(60., max(1., source_fps))
        audio = next((s for s in data['streams'] if s.get('codec_type') == 'audio'), None)
        info = dict(kind='video', width=w, height=h, duration=duration, fps=fps,
                    source_fps=source_fps, audio=audio, rotation=rotation,
                    source_start_time=data.get('format', {}).get('start_time'),
                    video_stream_index=stream['index'],
                    timing_policy='FFmpeg fps filter to CFR; VFR timestamps approximated, not preserved',
                    rotation_policy='FFmpeg default autorotate before identical analysis/render scaling')
    limit = {'preview': 1280, 'hd': 1920, 'source': max(w, h)}[quality]
    factor = min(1., limit / max(w, h))
    dw, dh = max(1, round(w * factor)), max(1, round(h * factor))
    info.update(display_width=dw, display_height=dh,
                output_width=dw + dw % 2, output_height=dh + dh % 2,
                quality=quality, even_padding=[dw % 2, dh % 2])
    return info


def video_frames(path, info, width, height, fps, timeout=600, pad=False):
    vf = f'fps={fps:.10f},scale={width}:{height}:flags=bilinear'
    ow, oh = width, height
    if pad:
        ow, oh = width + width % 2, height + height % 2
        vf += f',pad={ow}:{oh}:0:0:color=black'
    proc = PipeProcess(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                        '-filter_threads', '1', '-i', path, '-map',
                        f'0:{info["video_stream_index"]}', '-an', '-sn', '-dn',
                        '-vf', vf, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-threads', '1', 'pipe:1'],
                       timeout=timeout, reader=True)
    size = ow * oh * 3
    try:
        while True:
            chunks, remaining = [], size
            while remaining:
                chunk = proc.p.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == size:
                break
            if remaining:
                raise RuntimeError('FFmpeg解码返回不完整帧')
            yield np.frombuffer(b''.join(chunks), np.uint8).reshape(oh, ow, 3).copy()
        proc.finish()
    finally:
        proc.close()


def small_size(info):
    f = min(1., 256 / max(info['display_width'], info['display_height']))
    return max(1, round(info['display_width'] * f)), max(1, round(info['display_height'] * f))


def encode_image(image):
    ok, arr = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('缓存图像编码失败')
    return base64.b64encode(arr).decode('ascii')


def decode_image(text):
    arr = np.frombuffer(base64.b64decode(text), np.uint8)
    return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def difference(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    # Include RGB difference so equal-luminance color cuts are not invisible.
    diff = max(float(np.mean(cv2.absdiff(ga, gb))),
               float(np.mean(cv2.absdiff(a, b)))) / 255
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(ha, ha, 1, 0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1, 0, cv2.NORM_L1)
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
    return diff, hist


def normalize_cuts(cuts, duration):
    result = []
    for t in sorted(set(float(c) for c in cuts)):
        if not math.isfinite(t) or not 0 < t < duration:
            raise ValueError('cuts必须为时长范围内的有限正数')
        if t - (result[-1] if result else 0) < .25 - 1e-6 or duration - t < .25 - 1e-6:
            raise ValueError('人工cuts产生小于0.25秒的镜头，请调整cuts')
        result.append(t)
    return result


def analyze(path, info, input_hash, cache_dir, manual_cuts=None, timeout=600):
    signature = dict(version=ENGINE_VERSION, hash=input_hash, kind=info['kind'],
                     width=info['display_width'], height=info['display_height'],
                     duration=info['duration'], source_fps=info['fps'], cuts=manual_cuts)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / (key + '.json')
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            if cached['signature'] == signature and cached['shots']:
                for shot in cached['shots']:
                    decode_image(shot['representative_png'])
                return cached, True
        except (ValueError, KeyError, cv2.error, TypeError, OSError):
            pass
    sw, sh = small_size(info)
    sampling_fps = min(12., info['fps']) if info['kind'] == 'video' else 1.
    from collections import deque
    shots, pending, previous, count = [], None, None, 0
    history = deque(maxlen=24)
    cut_events = []
    cuts = normalize_cuts(manual_cuts, info['duration']) if manual_cuts is not None else None
    manual_index = 0
    frames = (video_frames(path, info, sw, sh, sampling_fps, timeout)
              if info['kind'] == 'video' else iter([cv2.resize(image_rgb(path), (sw, sh))]))
    try:
        for i, frame in enumerate(frames):
            count += 1
            t = i / sampling_fps
            if not shots:
                shots.append(dict(start=0., representative_png=encode_image(frame)))
            if cuts is not None:
                while manual_index < len(cuts) and t >= cuts[manual_index]:
                    shots.append(dict(start=cuts[manual_index], representative_png=encode_image(frame)))
                    manual_index += 1
            elif previous is not None:
                if pending is not None:
                    d, hist = difference(pending['before'], frame)
                    if d < .075 and hist < .22:
                        pending = None  # Return to pre-flash frame: reject impulse.
                    elif t - pending['time'] >= .25:
                        ct = pending['time']
                        if ct - shots[-1]['start'] >= .25 and info['duration'] - ct >= .25:
                            # Require persistence toward the new frame; gradual motion
                            # can diverge from the old frame but isn't a stable hard cut.
                            pd, ph = difference(pending['after'], frame)
                            if pd <= max(.035, d * .65) or ph <= max(.18, hist * .65):
                                shots.append(dict(start=ct, representative_png=encode_image(pending['after'])))
                                cut_events.append(pending['metrics'])
                                history.clear()
                        pending = None
                else:
                    d, hist = difference(previous, frame)
                    baseline_d = float(np.median([v[0] for v in history])) if history else 0.
                    baseline_h = float(np.median([v[1] for v in history])) if history else 0.
                    strong_change = d > .13 and hist > .38
                    # Similar-layout, differently graded/background shots have small
                    # whole-frame mean difference but strong color-distribution jumps.
                    color_jump = (hist > max(.65, baseline_h * 3 + .15)
                                  and d > max(.035, baseline_d * 4 + .015))
                    if (strong_change or color_jump) and t - shots[-1]['start'] >= .25:
                        pending = dict(time=t, before=previous, after=frame,
                                       metrics=dict(start=t, difference=d, histogram=hist,
                                                    baseline_difference=baseline_d, baseline_histogram=baseline_h,
                                                    rule='strong_change' if strong_change else 'adaptive_color_jump'))
                    else:
                        history.append((d, hist))
            previous = frame
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    if not shots:
        raise ValueError('分析未解码到任何帧')
    if cuts is not None:
        # Near EOF sampling can miss a short last interval; use last available frame.
        while manual_index < len(cuts):
            shots.append(dict(start=cuts[manual_index], representative_png=encode_image(previous)))
            manual_index += 1
    elif pending is not None and info['duration'] - pending['time'] >= .25:
        d, hist = difference(pending['before'], previous)
        pd, ph = difference(pending['after'], previous)
        if not (d < .075 and hist < .22) and (pd <= max(.035, d * .65) or ph <= max(.18, hist * .65)):
            shots.append(dict(start=pending['time'], representative_png=encode_image(pending['after'])))
            cut_events.append(pending['metrics'])
    for i, shot in enumerate(shots):
        shot['end'] = shots[i + 1]['start'] if i + 1 < len(shots) else info['duration']
    result = dict(signature=signature, shots=shots, sampling_fps=sampling_fps,
                  sampled_frames=count, cut_events=cut_events,
                  cut_method='manual override' if cuts is not None else 'frame difference + RGB histogram + adaptive color jump + persistence',
                  limitations=['低分辨率12fps采样（低fps源用全部CFR帧）；硬切时间约±1个采样间隔。',
                               '0.25s返回检测抑制短闪光；长闪光、快速运动、渐变转场可能漏检或误检。',
                               '小于0.25s的自动镜头合并；无硬切时整片一个镜头。',
                               '仅镜头代表帧评估位置；不保证之后移动主体/人脸不进入叠加区域。'])
    save_json(cache_file, result)
    return result, False


class DogLibrary:
    def __init__(self, assets):
        self.assets = Path(assets).resolve()
        manifest = json.loads((self.assets / 'dog_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('version') != 1 or not manifest.get('dogs'):
            raise ValueError('manifest须为version=1且dogs非空')
        self.entries = {}
        for dog in manifest['dogs']:
            if dog['id'] in self.entries:
                raise ValueError('manifest包含重复dog id')
            self.entries[dog['id']] = dog
        self.loaded_id, self.loaded = None, None

    def load(self, dog_id):
        if self.loaded_id == dog_id:
            return self.loaded
        e = self.entries[dog_id]
        path = (self.assets / e['file']).resolve()
        if not path.is_relative_to(self.assets) or not path.is_file():
            raise ValueError(f'{dog_id}: file必须在assets内且存在')
        if sha256(path) != e['sha256']:
            raise ValueError(f'{dog_id}: SHA256不匹配')
        with np.load(path, allow_pickle=False) as z:
            frames = z['frames']
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 4:
            raise ValueError(f'{dog_id}: frames必须uint8 N,H,W,4')
        n, h, w, _ = frames.shape
        if (n, w, h) != (int(e['frame_count']), int(e['width']), int(e['height'])) or n < 1:
            raise ValueError(f'{dog_id}: manifest尺寸/帧数不一致')
        fps = float(e['fps'])
        ax, ay = map(float, e['anchor'])
        if not math.isfinite(fps) or fps <= 0 or not all(map(math.isfinite, (ax, ay))):
            raise ValueError(f'{dog_id}: fps/anchor无效')
        if not 0 <= ax <= w or not 0 <= ay <= h:
            raise ValueError(f'{dog_id}: 脚底anchor不在裁切画布内')
        # End is an EXCLUSIVE zero-based bound, matching frames[start:end].
        # Keep original integrity checks above; do not alter the source NPZ.
        quality = e.get('quality', {})
        loop_start = int(e.get('loop_start_frame', quality.get('loop_start_frame', 0)))
        loop_end = int(e.get('loop_end_frame', quality.get('loop_end_frame', n)))
        if not 0 <= loop_start < loop_end <= n:
            raise ValueError(f'{dog_id}: loop区间须为0 <= start < end <= 原帧数（end不含）')
        jumps = [int(j) for j in quality.get('suspicious_color_jump_frames', [])]
        unsafe = [j for j in jumps if loop_start < j < loop_end]
        warnings = []
        if unsafe:
            loop_end = min(unsafe)
            warnings.append(f'{dog_id}: suspicious_color_jump_frames触发安全前缀[{loop_start},{loop_end})；源素材保留。')
        if loop_start in jumps:
            raise ValueError(f'{dog_id}: 循环起点已标记异常，不能默认安全循环')
        if loop_start != 0 or loop_end != n:
            warnings.append(f'{dog_id}: 仅循环原始帧[{loop_start},{loop_end})，原始{n}帧未删除；不宣称循环接缝已验证。')
        frames = frames[loop_start:loop_end]
        # Union over every effective phase; alpha>0, not just representative frame.
        union = np.zeros((h, w), dtype=bool)
        for frame in frames:
            alpha_mask = frame[:, :, 3] > 0
            if not np.any(alpha_mask):
                raise ValueError(f'{dog_id}: 包含全透明帧，会违反每帧恰好新增一只狗')
            union |= alpha_mask
        yy, xx = np.nonzero(union)
        if not len(xx):
            raise ValueError(f'{dog_id}: 所有帧透明')
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        data = dict(frames=frames, width=w, height=h, anchor=(ax, ay), fps=fps,
                    frame_count=len(frames), source_frame_count=n, loop_start_frame=loop_start,
                    loop_end_frame=loop_end, warnings=warnings, union_bbox=bbox, manifest=e)
        self.loaded_id, self.loaded = dog_id, data
        return data


class ShuffleBag:
    def __init__(self, ids, rng):
        self.ids, self.rng, self.bag, self.previous = sorted(ids), rng, [], None

    def next(self):
        if not self.bag:
            self.bag = list(self.ids)
            self.rng.shuffle(self.bag)
            if len(self.bag) > 1 and self.bag[-1] == self.previous:
                self.bag[-1], self.bag[0] = self.bag[0], self.bag[-1]
        self.previous = self.bag.pop()
        return self.previous


def face_boxes(rgb):
    cascade_dir = getattr(getattr(cv2, 'data', None), 'haarcascades', '')
    if not cascade_dir:
        return [], False
    cascade = Path(cascade_dir) / 'haarcascade_frontalface_default.xml'
    if not cascade.is_file():
        return [], False
    model = cv2.CascadeClassifier(str(cascade))
    if model.empty():
        return [], False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = model.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    return [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in boxes], True


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def person_boxes(rgb, faces):
    """Bundled classical HOG plus conservative face-derived body regions; no downloads."""
    h, w = rgb.shape[:2]
    boxes = []
    for x0, y0, x1, y1 in faces:
        fw, fh = x1 - x0, y1 - y0
        boxes.append([max(0, x0 - .8 * fw), max(0, y0 - .2 * fh),
                      min(w, x1 + .8 * fw), min(h, y1 + 5.5 * fh)])
    hog_available = callable(getattr(cv2, 'HOGDescriptor_getDefaultPeopleDetector', None))
    if hog_available and w >= 64 and h >= 128:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found, weights = hog.detectMultiScale(rgb, winStride=(8, 8), padding=(8, 8), scale=1.08)
        for (x, y, bw, bh), weight in zip(found, np.asarray(weights).reshape(-1)):
            if weight >= .5:
                boxes.append([max(0, x - .08 * bw), max(0, y - .04 * bh),
                              min(w, x + 1.08 * bw), min(h, y + 1.04 * bh)])
    return boxes, hog_available


def placement(rgb, info, dog, ratio, mode, rng, explicit_anchor=None):
    """Joint bounded search. Manual size bypasses preference, not safety constraints."""
    faces, haar = face_boxes(rgb)
    people, hog = person_boxes(rgb, faces)
    protection = (faces, haar, people, hog)
    sizes = [ratio] if ratio is not None else [.24, .22, .20, .18]
    choices, failures = [], []
    # Smaller sizes are a documented safety fallback, not the default preference.
    for requested in sizes + ([] if ratio is not None else [.15, .12, .09, .06]):
        if ratio is None and requested < .18 and choices:
            break
        try:
            pos = _fixed_placement(rgb, info, dog, requested, mode, rng, explicit_anchor, protection)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        # Pixel rounding + full-cycle fit determine actual size (never stale rh).
        actual = pos['height'] / info['display_height']
        pos['actual_height_ratio'] = actual
        pos['size_preference_cost'] = 0. if ratio is not None else .9 * max(0., .24 - actual)
        pos['joint_cost'] = pos['cost'] + pos['size_preference_cost']
        pos['evaluated_height_ratio'] = requested
        choices.append(pos)
    if not choices:
        raise ValueError('所有尺寸/落点均无法满足完整周期边界和检测保护: ' + '; '.join(failures[:2]))
    # Deterministic tie-break: prefer larger; random seed selects spatial near-ties.
    chosen = dict(min(choices, key=lambda p: (p['joint_cost'], -p['actual_height_ratio'])))
    chosen.update(size_mode='manual' if ratio is not None else 'adaptive', requested_height_ratio=ratio,
                  evaluated_sizes=[dict(height_ratio=p['actual_height_ratio'],
                                        requested=p['evaluated_height_ratio'], joint_cost=p['joint_cost'],
                                        anchor=p['anchor_normalized'], kind=p['kind']) for p in choices],
                  selection_reason='完整周期画布边界/检测保护硬约束；联合比较纹理显著性、下半部、支撑几何与尺寸偏好',
                  source_asset_resolution=[dog['width'], dog['height']],
                  output_asset_resolution=[chosen['width'], chosen['height']],
                  scale_xy=[chosen['width'] / dog['width'], chosen['height'] / dog['height']],
                  native_asset_provenance=dog.get('manifest', {}).get('resolution_provenance'),
                  analysis_resolution=[rgb.shape[1], rgb.shape[0]])
    if ratio is None and chosen['actual_height_ratio'] < .175:
        chosen['warnings'].append('常规0.18–0.24尺寸无安全落点；已缩小退化，不保证审美。')
    return chosen


def _fixed_placement(rgb, info, dog, ratio, mode, rng, explicit_anchor=None, protection=None):
    W, H = info['display_width'], info['display_height']
    sw, sh = rgb.shape[1], rgb.shape[0]
    scale = min(H * ratio / dog['height'], W * .8 / dog['width'], H * .8 / dog['height'])
    rw, rh = max(1, round(dog['width'] * scale)), max(1, round(dog['height'] * scale))
    ax = dog['anchor'][0] * rw / dog['width']
    ay = dog['anchor'][1] * rh / dog['height']
    faces, haar, people, hog = protection
    faces = [[x * W / sw, y * H / sh, x2 * W / sw, y2 * H / sh] for x, y, x2, y2 in faces]
    people = [[x * W / sw, y * H / sh, x2 * W / sw, y2 * H / sh] for x, y, x2, y2 in people]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)) / 1024
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    contrast = np.linalg.norm(lab - np.mean(lab, axis=(0, 1)), axis=2) / 255
    yy, xx = np.mgrid[:sh, :sw]
    center = np.exp(-(((xx / sw - .5) / .3) ** 2 + ((yy / sh - .5) / .35) ** 2))
    saliency = contrast * .6 + grad * .4 + center * .18
    warnings = ['显著性proxy不是可靠语义识别/主体避让；Hough水平线不等于理解地面。']
    if not haar:
        warnings.append('本地Haar级联不可用，未做人脸检测；不下载模型。')
    if not hog:
        warnings.append('本地HOG人物检测不可用；仅保留可用的人脸扩展近似框，不保证人物避让。')

    def evaluate(x, y, support, kind, geometry=None):
        left, top = round(x - ax), round(y - ay)
        rect = [left, top, left + rw, top + rh]
        # Full RGBA canvas bound is stronger than union-alpha bound.
        if left < 0 or top < 0 or rect[2] > W or rect[3] > H:
            return None
        if any(intersects(rect, f) for f in faces + people):
            return None
        x0, y0 = max(0, int(left * sw / W)), max(0, int(top * sh / H))
        x1, y1 = min(sw, max(x0 + 1, math.ceil(rect[2] * sw / W))), min(sh, max(y0 + 1, math.ceil(rect[3] * sh / H)))
        lower_region_cost = .55 * max(0., (.72 - y / H) / .72)
        cost = (float(np.mean(saliency[y0:y1, x0:x1]))
                + .35 * float(np.mean(grad[y0:y1, x0:x1])) - support * .15
                + lower_region_cost)
        if geometry:
            cost += geometry['cost_adjustment']
        return dict(x=left, y=top, width=rw, height=rh, anchor_pixel=[left + ax, top + ay],
                    anchor_normalized=[(left + ax) / W, (top + ay) / H], cost=cost,
                    kind=kind, support_score=support, support_geometry=geometry,
                    lower_region_cost=lower_region_cost)

    candidates = []
    if explicit_anchor is not None:
        # Shrink, do not silently move the requested normalized foot point.
        x, y = explicit_anchor[0] * W, explicit_anchor[1] * H
        for _ in range(80):
            p = evaluate(x, y, 0, 'explicit_anchor')
            if p:
                candidates = [p]
                break
            if rw <= 1 and rh <= 1:
                break
            rw, rh = max(1, int(rw * .9)), max(1, int(rh * .9))
            ax, ay = dog['anchor'][0] * rw / dog['width'], dog['anchor'][1] * rh / dog['height']
        if not candidates:
            raise ValueError('指定anchor不能在不越界/不覆盖检测人脸的条件下放置狗')
    else:
        if mode == 'auto':
            edges = cv2.Canny(gray, 60, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                                    minLineLength=max(15, int(sw * .12)), maxLineGap=12)
            # OpenCV bindings return either N,1,4 or N,4 depending on version.
            line_rows = [] if lines is None else np.asarray(lines).reshape(-1, 4)[:150].tolist()
            # Unrestricted probabilistic Hough can consume fragmented rail edges
            # into many diagonals in water/foliage. A near-horizontal accumulator
            # retains those real long supports; demand observed edge coverage.
            aligned = cv2.HoughLines(edges, 1, np.pi / 360,
                                     threshold=max(18, int(sw * .35)),
                                     min_theta=np.pi / 2 - .10, max_theta=np.pi / 2 + .10)
            if aligned is not None:
                for rho, theta in np.asarray(aligned).reshape(-1, 2)[:40]:
                    cols = np.arange(sw)
                    ys = (float(rho) - cols * math.cos(float(theta))) / math.sin(float(theta))
                    observed = np.zeros(sw, dtype=bool)
                    for offset in (-1, 0, 1):
                        rows = np.rint(ys).astype(int) + offset
                        valid = (rows >= 0) & (rows < sh)
                        observed[valid] |= edges[rows[valid], cols[valid]] > 0
                    support_x = np.flatnonzero(observed)
                    if len(support_x) < max(12, sw * .3):
                        continue
                    lx, rx = int(support_x[0]), int(support_x[-1])
                    if len(support_x) / (rx - lx + 1) < .5:
                        continue
                    line_rows.append([lx, float(ys[lx]), rx, float(ys[rx])])
            verticals = [v for v in line_rows if abs(float(v[2] - v[0])) <= abs(float(v[3] - v[1])) * .25
                         and abs(float(v[3] - v[1])) >= sh * .06]
            for line in line_rows:
                x1, y1, x2, y2 = map(float, line)
                if min(y1, y2) / sh < .52:
                    continue  # Exclude upper-frame ceiling/distant structures, not semantic detection.
                if abs(y2 - y1) > max(1, abs(x2 - x1)) * .22:
                    continue
                xa, xb = max(0, int(min(x1, x2))), min(sw, int(max(x1, x2)) + 1)
                ly = max(0, min(sh - 1, round((y1 + y2) / 2)))
                band, gap = max(2, round(sh * .035)), max(1, round(sh * .006))
                above = (max(0, ly - band - gap), max(0, ly - gap))
                below = (min(sh, ly + gap + 1), min(sh, ly + band + gap + 1))
                def strip_mean(arr, bounds):
                    region = arr[bounds[0]:bounds[1], xa:xb]
                    return float(np.mean(region)) if region.size else 0.
                texture_above = strip_mean(grad, above)
                texture_below = strip_mean(grad, below)
                smooth_both = max(0., 1 - max(texture_above, texture_below) / .10)
                length_fraction = abs(x2 - x1) / sw
                longness = min(1., max(0., (length_fraction - .5) / .35))
                normalized_y = ly / sh
                mid_upper = min(1., max(0., (.72 - normalized_y) / .12))
                horizon_penalty = .26 * longness * smooth_both * mid_upper
                lower_support_bonus = .075 * min(1., max(0., (normalized_y - .45) / .45))
                # Darker material below a bright edge is a weak top-surface cue;
                # the opposite polarity is often the underside of a beam/shelf.
                polarity = (strip_mean(gray, above) - strip_mean(gray, below)) / 255
                top_edge_bonus = .06 * max(-1., min(1., polarity))
                # A coherent, low-texture material strip immediately BELOW an
                # edge is a useful beam/shelf cue, unlike random water edge votes.
                material_bonus = .22 * max(0., 1 - texture_below / .08) * min(1., max(0., polarity / .35))
                # Thin image-bottom borders should not outrank interior supports;
                # this is a generic framing penalty, not a target anchor coordinate.
                border_penalty = .20 * min(1., max(0., (normalized_y - .90) / .10))
                for k in (.25, .5, .75):
                    px, py = x1 + k * (x2 - x1), y1 + k * (y2 - y1)
                    vertical_reward = 0.
                    for vx1, vy1, vx2, vy2 in verticals:
                        vx, vtop, vbottom = float(vx1 + vx2) / 2, float(min(vy1, vy2)), float(max(vy1, vy2))
                        # Reward only a nearby perpendicular structure that reaches
                        # the support line; not every vertical elsewhere in the shot.
                        distance_y = max(vtop - py, py - vbottom, 0.)
                        if distance_y <= sh * .08:
                            proximity = max(0., 1 - abs(vx - px) / max(sw * .3, 1.))
                            reach = max(0., 1 - distance_y / max(sh * .08, 1.))
                            vertical_reward = max(vertical_reward, proximity * reach)
                    geometry = dict(line_normalized=[x1 / sw, y1 / sh, x2 / sw, y2 / sh],
                                    texture_above=texture_above, texture_below=texture_below,
                                    horizon_penalty=horizon_penalty, lower_support_bonus=lower_support_bonus,
                                    nearby_vertical_score=vertical_reward, top_edge_bonus=top_edge_bonus,
                                    material_below_bonus=material_bonus, image_border_penalty=border_penalty,
                                    cost_adjustment=horizon_penalty + border_penalty - lower_support_bonus
                                                    - .10 * vertical_reward - top_edge_bonus - material_bonus)
                    p = evaluate(px * W / sw, py * H / sh,
                                 min(1., length_fraction), 'hough_horizontal_support', geometry)
                    if p:
                        candidates.append(p)
        if mode == 'auto':
            # Always compete with supports: a line must beat usable lower-frame
            # whitespace instead of forcing a remote ceiling/horizon placement.
            for fx in np.linspace(.08, .92, 9):
                for fy in (.68, .78, .88, .97):
                    p = evaluate(ax + fx * (W - rw), min(H - rh + ay, H * fy),
                                 0, 'lower_whitespace')
                    if p:
                        candidates.append(p)
        if not candidates:
            if mode == 'auto':
                warnings.append('无可用可靠支撑候选：退回角落/边缘留白，不声称理解地面。')
            # Corners first; if faces block every corner use conservative perimeter grid.
            corners = [(0, 1), (1, 1)] if mode == 'auto' else [(0, 1), (1, 1), (0, 0), (1, 0)]
            for fx, fy in corners:
                p = evaluate(ax + fx * (W - rw), ay + fy * (H - rh), 0, 'corner_fallback')
                if p:
                    candidates.append(p)
            if not candidates:
                for fx in np.linspace(0, 1, 9):
                    for fy in ((.65, .8, 1.) if mode == 'auto' else (0., .5, 1.)):
                        p = evaluate(ax + fx * (W - rw), ay + fy * (H - rh), 0, 'edge_fallback')
                        if p:
                            candidates.append(p)
            if not candidates:
                raise ValueError('找不到不越界且不覆盖检测人脸的放置位置')
    best = min(p['cost'] for p in candidates)
    shortlist = [p for p in candidates if p['cost'] <= best + .035]
    chosen = dict(rng.choice(shortlist))
    chosen.update(candidate_count=len(candidates), best_candidate_count=len(shortlist),
                  ranked_candidates=sorted(candidates, key=lambda p: p['cost'])[:12],
                  detected_faces=faces, detected_person_regions=people, haar_available=haar, hog_available=hog, warnings=warnings,
                  union_bbox_source=dog['union_bbox'], whole_canvas_bounds_checked=True,
                  actual_height_ratio=rh / H)
    return chosen


class PreparedDog:
    """Current shot's pre-multiplied, resized frames; bounded LRU for larger assets."""
    def __init__(self, dog, position, byte_budget=64 * 1024 * 1024):
        from collections import OrderedDict
        self.dog, self.position, self.cache = dog, position, OrderedDict()
        self.limit = max(1, min(dog['frame_count'], byte_budget // max(1, position['width'] * position['height'] * 16)))

    def frame(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        src = self.dog['frames'][index].astype(np.float32) / 255
        alpha = src[:, :, 3:4]
        premul = np.concatenate([src[:, :, :3] * alpha, alpha], axis=2)
        size = (self.position['width'], self.position['height'])
        # High-quality sampling in premultiplied space. Bound Lanczos ringing to
        # valid associated-alpha colors, so invisible RGB/negative lobes cannot fringe.
        interp = cv2.INTER_AREA if size[0] <= src.shape[1] and size[1] <= src.shape[0] else cv2.INTER_LANCZOS4
        premul = cv2.resize(premul, size, interpolation=interp)
        raw_alpha = premul[:, :, 3:4].copy()
        straight = np.clip(premul[:, :, :3] / np.maximum(raw_alpha, 1e-8), 0, 1)
        straight[raw_alpha[:, :, 0] <= 0] = 0
        premul[:, :, 3:4] = np.clip(raw_alpha, 0, 1)
        premul[:, :, :3] = straight * premul[:, :, 3:4]
        value = (premul[:, :, :3] * 255, premul[:, :, 3:4])
        self.cache[index] = value
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return value

    def overlay(self, rgb, index):
        p = self.position
        roi = rgb[p['y']:p['y'] + p['height'], p['x']:p['x'] + p['width']]
        color, alpha = self.frame(index)
        roi[:] = np.clip(np.rint(color + roi * (1 - alpha)), 0, 255).astype(np.uint8)
