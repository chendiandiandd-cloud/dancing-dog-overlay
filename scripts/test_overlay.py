#!/usr/bin/env python3
"""Black-box regression checks using bundled dogs and generated input fixtures."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def invoke(args, expect_ok=True):
    p = subprocess.run([sys.executable, str(ROOT/'scripts/render_overlay.py'), *args],
                       capture_output=True, text=True, timeout=180)
    if expect_ok and p.returncode:
        raise AssertionError(f'CLI failed ({p.returncode}): {p.stdout[-3000:]} {p.stderr[-3000:]}')
    return p


def probe(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                        '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(p.stdout)


def plans(directory):
    found = []
    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get('shots'), list) and obj['shots']:
                found.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    for p in Path(directory).rglob('*.json'):
        try:
            walk(json.loads(p.read_text()))
        except (ValueError, UnicodeError):
            continue
    if not found:
        raise AssertionError(f'No machine-readable shot plan under {directory}')
    return found


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='dog-overlay-tests-')
        cls.base = Path(cls.temp.name)
        cls.fixtures = cls.base/'fixtures'
        subprocess.run([sys.executable, str(ROOT/'scripts/make_test_inputs.py'), '--output-dir', str(cls.fixtures)],
                       capture_output=True, text=True, check=True, timeout=120)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def out(self, name):
        d = self.base/name
        d.mkdir(exist_ok=True)
        return d

    def test_01_asset_integrity(self):
        manifest = json.loads((ROOT/'assets/dog_manifest.json').read_text())
        self.assertEqual(len(manifest['dogs']), 16)
        self.assertEqual(len({d['id'] for d in manifest['dogs']}), 16)
        for d in manifest['dogs']:
            with np.load(ROOT/'assets'/d['file'], allow_pickle=False) as z:
                f = z['frames']
                self.assertEqual(f.dtype, np.uint8)
                self.assertEqual(f.shape, (d['frame_count'], d['height'], d['width'], 4))
                self.assertGreater(np.count_nonzero(f[...,3]), 100)
                self.assertGreater(np.count_nonzero(f[...,3] == 0), 100)
                self.assertTrue(0 <= d['anchor'][0] <= d['width'])
                self.assertTrue(0 <= d['anchor'][1] <= d['height'])

    def test_02_help(self):
        self.assertIn('--inputs', invoke(['--help']).stdout)

    def test_03_manual_cuts_and_nonrepeat(self):
        out = self.out('cuts')
        invoke(['--inputs', str(self.fixtures/'three_shots.mp4'), '--output-dir', str(out),
                '--cuts', '1,2', '--seed', '7', '--plan-only'])
        shot_plan = plans(out)[0]['shots']
        self.assertEqual(len(shot_plan), 3)
        ids = [s['dog_id'] for s in shot_plan]
        self.assertTrue(all(a != b for a,b in zip(ids, ids[1:])))
        self.assertAlmostEqual(shot_plan[1]['start'], 1, places=2)
        self.assertAlmostEqual(shot_plan[2]['start'], 2, places=2)

    def test_04_seed_reproducibility(self):
        results=[]
        for name in ('seed_a','seed_b'):
            out=self.out(name)
            invoke(['--inputs', str(self.fixtures/'three_shots.mp4'), '--output-dir', str(out),
                    '--cuts', '1,2', '--seed', '42', '--plan-only'])
            # Reproducibility concerns rendered dog/geometry/time, not tiny
            # parallel floating-point differences in diagnostic texture scores.
            shots=plans(out)[0]['shots']
            results.append([{
                'start':s['start'], 'end':s['end'], 'dog_id':s['dog_id'],
                'phase_frames':s['phase_frames'],
                'anchor':[round(x,6) for x in s['anchor']],
                'height_ratio':round(s['height_ratio'],6),
            } for s in shots])
        self.assertEqual(results[0], results[1])

    def test_05_photo_movie(self):
        out=self.out('photo')
        invoke(['--inputs', str(self.fixtures/'scene_1.png'), '--output-dir', str(out),
                '--image-duration', '0.5', '--seed', '1'])
        mp4=list(out.rglob('*.mp4'))
        self.assertEqual(len(mp4),1)
        d=probe(mp4[0]); v=next(s for s in d['streams'] if s['codec_type']=='video')
        self.assertEqual((v['width'],v['height']),(960,540))
        self.assertLess(abs(float(d['format']['duration'])-0.5),0.06)
        self.assertFalse(any(s['codec_type']=='audio' for s in d['streams']))

    def test_06_video_audio_duration(self):
        out=self.out('audio')
        invoke(['--inputs', str(self.fixtures/'three_shots.mp4'), '--output-dir', str(out),
                '--cuts', '1,2', '--seed','2'])
        d=probe(next(out.rglob('*.mp4')))
        self.assertTrue(any(s['codec_type']=='audio' for s in d['streams']))
        self.assertLess(abs(float(d['format']['duration'])-3),0.06)

    def test_07_portrait_silent(self):
        out=self.out('portrait')
        invoke(['--inputs',str(self.fixtures/'portrait_silent.mp4'),'--output-dir',str(out),'--seed','3'])
        d=probe(next(out.rglob('*.mp4')))
        v=next(s for s in d['streams'] if s['codec_type']=='video')
        self.assertEqual((v['width'],v['height']),(360,640))
        self.assertFalse(any(s['codec_type']=='audio' for s in d['streams']))

    def test_08_mixed_failure_isolation(self):
        out=self.out('batch')
        p=invoke(['--inputs',str(self.fixtures/'scene_2.png'),str(self.fixtures/'broken.mp4'),
                  '--output-dir',str(out),'--image-duration','0.5'],expect_ok=False)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(len(list(out.rglob('*.mp4'))),1)
        self.assertTrue(list(out.rglob('*.json')))

    def test_09_bad_input_fails(self):
        out=self.out('bad')
        p=invoke(['--inputs',str(self.fixtures/'broken.mp4'),'--output-dir',str(out)],expect_ok=False)
        self.assertNotEqual(p.returncode,0)
        self.assertFalse(list(out.rglob('*.mp4')))

    def test_10_automatic_cuts(self):
        out=self.out('automatic')
        invoke(['--inputs',str(self.fixtures/'three_shots.mp4'),'--output-dir',str(out),
                '--seed','5','--plan-only'])
        shots=plans(out)[0]['shots']
        self.assertEqual(len(shots),3)
        self.assertLess(abs(shots[1]['start']-1),0.09)
        self.assertLess(abs(shots[2]['start']-2),0.09)


    def mock_dog(self):
        frames = np.zeros((3, 100, 100, 4), np.uint8)
        frames[:, 3:96, 3:97] = [210, 80, 30, 255]
        return dict(frames=frames, frame_count=3, width=100, height=100,
                    anchor=(50, 96), union_bbox=[3, 3, 97, 96])

    def test_11_adaptive_joint_size_and_protection(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, intersects
        info = dict(display_width=400, display_height=400)
        blank = np.full((400, 400, 3), 180, np.uint8)
        with patch('overlay_core.face_boxes', return_value=([], True)), patch('overlay_core.person_boxes', return_value=([], True)):
            large = placement(blank, info, self.mock_dog(), None, 'auto', random.Random(42))
            again = placement(blank, info, self.mock_dog(), None, 'auto', random.Random(42))
        self.assertEqual(large, again)
        self.assertGreaterEqual(large['actual_height_ratio'], .22)
        self.assertGreater(large['anchor_normalized'][1], .6)
        self.assertEqual(len(large['evaluated_sizes']), 4)
        obstacle = [80, 0, 400, 400]
        with patch('overlay_core.face_boxes', return_value=([], True)), patch('overlay_core.person_boxes', return_value=([obstacle], True)):
            smaller = placement(blank, info, self.mock_dog(), None, 'auto', random.Random(42))
        self.assertLess(smaller['actual_height_ratio'], large['actual_height_ratio'])
        self.assertLessEqual(smaller['width'], 80)
        self.assertFalse(intersects([smaller['x'], smaller['y'], smaller['x']+smaller['width'], smaller['y']+smaller['height']], obstacle))
        self.assertEqual(smaller['size_mode'], 'adaptive')

    def test_12_manual_override_and_boundary_ratio(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement, PreparedDog
        blank = np.full((400, 400, 3), 180, np.uint8)
        info = dict(display_width=400, display_height=400)
        with patch('overlay_core.face_boxes', return_value=([], True)), patch('overlay_core.person_boxes', return_value=([], True)):
            exact = placement(blank, info, self.mock_dog(), .12, 'auto', random.Random(3), [.5, .8])
            fitted = placement(blank, info, self.mock_dog(), .8, 'auto', random.Random(3), [.05, .15])
        self.assertEqual(exact['size_mode'], 'manual')
        self.assertAlmostEqual(exact['actual_height_ratio'], .12)
        self.assertLess(fitted['actual_height_ratio'], .8)
        self.assertEqual(fitted['actual_height_ratio'], fitted['height']/400)
        for p, anchor in [(exact, [.5, .8]), (fitted, [.05, .15])]:
            self.assertTrue(np.allclose(p['anchor_normalized'], anchor, atol=1/400))
            self.assertGreaterEqual(p['x'], 0); self.assertGreaterEqual(p['y'], 0)
            self.assertLessEqual(p['x']+p['width'], 400); self.assertLessEqual(p['y']+p['height'], 400)
            prep = PreparedDog(self.mock_dog(), p)
            for index in range(3):
                out = blank.copy(); prep.overlay(out, index)
                self.assertTrue(np.any(out != blank))

    def test_13_premultiplied_scaling_up_and_down(self):
        from overlay_core import PreparedDog
        from prepare_assets import resize_rgba
        rgba = np.zeros((20, 20, 4), np.uint8)
        rgba[..., :3] = [0, 255, 0]  # Invisible green must never contribute.
        rgba[5:15, 5:15] = [255, 60, 20, 255]
        for size in [(8, 8), (57, 57), (8, 57)]:
            result = resize_rgba(rgba, size)
            mask = result[..., 3] > 5
            self.assertTrue(np.any(mask))
            self.assertTrue(np.all(np.abs(result[mask, :3].astype(int)-[255,60,20]) <= 1))
            prep = PreparedDog(dict(frames=rgba[None], frame_count=1), dict(width=size[0], height=size[1]))
            color, alpha = prep.frame(0)
            self.assertTrue(np.all((alpha >= 0) & (alpha <= 1)))
            self.assertTrue(np.all((color >= 0) & (color <= 255*alpha+1e-4)))
            np.testing.assert_allclose(color[..., 1], color[..., 0]*60/255, atol=1e-4)
            np.testing.assert_allclose(color[..., 2], color[..., 0]*20/255, atol=1e-4)

    def test_14_native_asset_dimensions_and_source_accounting(self):
        from overlay_core import sha256, DogLibrary
        manifest = json.loads((ROOT/'assets/dog_manifest.json').read_text())
        total = 0
        for d in manifest['dogs']:
            p = d['resolution_provenance']
            x0,y0,x1,y1 = d['crop_bbox']
            self.assertEqual([d['width'],d['height']], [x1-x0,y1-y0])
            self.assertEqual(p['cache_resolution'], p['native_crop_resolution'])
            self.assertEqual(d['scale_xy'], [1.,1.])
            self.assertTrue(p['native_pixels_preserved'])
            self.assertEqual(p['sharpen_amount'], 0)
            self.assertEqual(sha256(ROOT/'assets'/d['source']), d['source_sha256'])
            self.assertEqual(sha256(ROOT/'assets'/d['file']), d['sha256'])
            total += d['frame_count']
            self.assertTrue(DogLibrary(ROOT/'assets').load(d['id'])['frame_count'] > 0)
        self.assertEqual(total, 827)
        d15 = next(d for d in manifest['dogs'] if d['id']=='dog_15')
        self.assertEqual(d15['runtime_source_frame_range'], [0,58])
        self.assertEqual(d15['trimmed_source_frame_indices'], [58])
        d06 = next(d for d in manifest['dogs'] if d['id']=='dog_06')
        self.assertTrue(d06['enclosed_green_policy']['preserve'])
        self.assertTrue(d06['quality']['protected_green_rgb_all_unchanged'])
        self.assertTrue(d06['quality']['protected_green_all_opaque'])
        self.assertGreater(d06['height'],384)

    def test_15_native_pixels_match_original_matte(self):
        import cv2
        from prepare_assets import matte
        manifest = json.loads((ROOT/'assets/dog_manifest.json').read_text())
        for dog_id in ['dog_06','dog_15']:
            d = next(d for d in manifest['dogs'] if d['id']==dog_id)
            cap = cv2.VideoCapture(str(ROOT/'assets'/d['source']))
            try:
                ok, bgr = cap.read(); self.assertTrue(ok)
            finally:
                cap.release()
            rgba, _ = matte(bgr, d['enclosed_green_policy']['preserve'])
            x0,y0,x1,y1 = d['crop_bbox']
            expected = np.zeros((y1-y0,x1-x0,4),np.uint8)
            sx0,sy0=max(0,x0),max(0,y0); sx1,sy1=min(rgba.shape[1],x1),min(rgba.shape[0],y1)
            expected[sy0-y0:sy1-y0,sx0-x0:sx1-x0] = rgba[sy0:sy1,sx0:sx1]
            with np.load(ROOT/'assets'/d['file'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['frames'][0], expected)

    def test_16_upper_structure_not_auto_support(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        scene = np.full((400,400,3),180,np.uint8)
        scene[100:115] = 25; scene[140:150] = 20
        with patch('overlay_core.face_boxes', return_value=([],True)), patch('overlay_core.person_boxes', return_value=([],True)):
            pos = placement(scene,dict(display_width=400,display_height=400),self.mock_dog(),None,'auto',random.Random(8))
        self.assertGreater(pos['anchor_normalized'][1], .6)
        self.assertNotEqual(pos['kind'], 'hough_horizontal_support')
        self.assertIn('selection_reason',pos)

    def test_17_local_person_region_geometry(self):
        from unittest.mock import patch
        from overlay_core import person_boxes
        with patch('overlay_core.cv2.HOGDescriptor_getDefaultPeopleDetector', None, create=True):
            # Small image skips HOG but still protects face-derived body extent.
            boxes, _ = person_boxes(np.zeros((100,50,3),np.uint8), [[15,20,25,30]])
        self.assertEqual(len(boxes),1)
        self.assertLess(boxes[0][0],15); self.assertGreater(boxes[0][3],30)
        self.assertTrue(0 <= boxes[0][0] < boxes[0][2] <= 50)


    def test_18_full_cycle_real_asset_bounds(self):
        import random
        from unittest.mock import patch
        from overlay_core import DogLibrary, placement, PreparedDog
        library = DogLibrary(ROOT/'assets')
        blank = np.full((640,360,3),180,np.uint8)
        with patch('overlay_core.face_boxes',return_value=([],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            for dog_id in library.entries:
                dog = library.load(dog_id)
                p = placement(blank,dict(display_width=360,display_height=640),dog,.24,'auto',random.Random(8),[.4,.82])
                self.assertTrue(p['whole_canvas_bounds_checked'])
                prep = PreparedDog(dog,p)
                for i in range(dog['frame_count']):
                    color, alpha = prep.frame(i)
                    self.assertEqual(alpha.shape[:2],(p['height'],p['width']))
                    self.assertGreater(float(alpha.max()),0)
                    frame=blank.copy(); prep.overlay(frame,i)
                    delta=np.any(frame!=blank,axis=2)
                    outside=delta.copy(); outside[p['y']:p['y']+p['height'],p['x']:p['x']+p['width']]=False
                    self.assertFalse(np.any(outside))

    def test_19_manual_protection_conflict_fails(self):
        import random
        from unittest.mock import patch
        from overlay_core import placement
        blank=np.full((200,200,3),180,np.uint8)
        with patch('overlay_core.face_boxes',return_value=([[50,50,150,150]],True)), patch('overlay_core.person_boxes',return_value=([],True)):
            with self.assertRaises(ValueError):
                placement(blank,dict(display_width=200,display_height=200),self.mock_dog(),.24,'auto',random.Random(1),[.5,.5])

    def test_20_bundled_local_detectors_available(self):
        import cv2
        from overlay_core import face_boxes, person_boxes
        self.assertLess(int(cv2.__version__.split('.')[0]),5)
        rgb=np.zeros((256,128,3),np.uint8)
        _, haar=face_boxes(rgb)
        _, hog=person_boxes(rgb,[])
        self.assertTrue(haar); self.assertTrue(hog)

    def test_21_default_batch_photos_are_five_seconds(self):
        out = self.out('default_five_seconds')
        invoke(['--inputs', str(self.fixtures/'scene_1.png'),
                str(self.fixtures/'scene_2.png'), '--output-dir', str(out),
                '--seed', '1024', '--placement-mode', 'corner'])
        movies = list(out.rglob('*.mp4'))
        self.assertEqual(len(movies), 2)
        for movie in movies:
            data = probe(movie)
            self.assertLess(abs(float(data['format']['duration']) - 5.0), 0.06)
            self.assertFalse(any(s['codec_type'] == 'audio' for s in data['streams']))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--report',default=str(ROOT/'temp_data/test_results.json'))
    args=a.parse_args()
    start=time.perf_counter()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OverlayTests))
    report={'ok':r.wasSuccessful(),'tests_run':r.testsRun,'failures':len(r.failures),
            'errors':len(r.errors),'elapsed_seconds':round(time.perf_counter()-start,3),
            'details':[{'test':str(t),'trace':msg} for t,msg in r.failures+r.errors]}
    path=Path(args.report); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
    sys.exit(0 if r.wasSuccessful() else 1)


if __name__=='__main__':
    main()
