#!/usr/bin/env python3
"""Generate reproducible local regression fixtures, without network or models."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def run(args):
    subprocess.run(args, check=True, capture_output=True, timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-image', help='Optional scenery reference; crop right side to exclude the existing sticker in supplied reference 01')
    a = parser.parse_args()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    palette = [((230, 167, 132), (57, 78, 91)), ((104, 145, 169), (32, 59, 65)), ((170, 172, 132), (49, 61, 42))]
    photos = []
    for i, (top, bottom) in enumerate(palette):
        t = np.linspace(0, 1, 540)[:, None, None]
        rgb = ((1-t) * np.array(top) + t * np.array(bottom)).astype(np.uint8)
        im = Image.fromarray(np.repeat(rgb, 960, axis=1))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 390, 960, 540), fill=(48+i*10, 62+i*10, 69))
        d.line((0, 385, 960, 385), fill=(210, 200, 180), width=5)
        for x in (160, 370, 680, 850):
            d.line((x, 385, x, 540), fill=(75, 78, 80), width=7)
        d.rectangle((400, 230, 620, 380), fill=(125, 110+i*20, 95))
        d.text((24, 24), f'REGRESSION SCENE {i+1}', fill=(255, 255, 255))
        p = out / f'scene_{i+1}.png'
        im.save(p)
        photos.append(p)
    if a.reference_image:
        with Image.open(a.reference_image) as im:
            im = im.convert('RGB')
            crop = im.crop((int(im.width*0.45), 0, im.width, im.height))
            crop.save(out/'scenery_crop.png')
    im = Image.open(photos[0])
    im.resize((360, 640)).save(out/'portrait.png')
    # Exact 24 fps cuts at 1 and 2 seconds, with a continuous mono AAC audio track.
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
    for p in photos:
        args += ['-loop', '1', '-framerate', '24', '-t', '1', '-i', str(p)]
    args += ['-f', 'lavfi', '-t', '3', '-i', 'sine=frequency=440:sample_rate=48000',
             '-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]',
             '-map', '[v]', '-map', '3:a', '-c:v', 'libx264', '-preset', 'ultrafast',
             '-threads', '2', '-filter_complex_threads', '1', '-c:a', 'aac', '-t', '3',
             '-movflags', '+faststart', str(out/'three_shots.mp4')]
    run(args)
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-loop', '1',
         '-framerate', '24', '-i', str(out/'portrait.png'), '-t', '1', '-an',
         '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-pix_fmt', 'yuv420p',
         str(out/'portrait_silent.mp4')])
    (out/'broken.mp4').write_bytes(b'not a video')
    provenance = {'synthetic': [p.name for p in photos]+['portrait.png', 'three_shots.mp4', 'portrait_silent.mp4'],
                  'reference_image': a.reference_image,
                  'scenery_crop': 'right 55%, cropped to exclude existing dog; use supplied reference image 01 only'}
    (out/'fixtures.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps({'ok': True, 'output_dir': str(out), 'fixtures': provenance}, ensure_ascii=False))


if __name__ == '__main__':
    main()
