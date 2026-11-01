"""Build clean map plates (no pin, no giant arrow) from the extracted source frames.

Each map segment of the source is a static screenshot with an animated red pin.
A per-pixel median over frames from the segment removes the moving pin; on the
turn map (C) the baked-in black arrow is masked and inpainted.
Usage: python3 -I build_plates.py <frames_dir> <out_dir>
"""
import sys, json, os
import numpy as np, cv2

FR, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, '..', 'edit.json')))

for name, m in cfg['maps'].items():
    x0, y0, x1, y1 = m['crop']
    s, e = m['median_frames']
    idx = np.linspace(s, e, 41).astype(int)
    stack = np.stack([cv2.imread(f'{FR}/{i:05d}.jpg')[y0:y1, x0:x1] for i in idx])
    plate = np.median(stack, axis=0).astype(np.uint8)
    if 'erase_light' in m:  # baked-in white caption
        ex0, ey0, ex1, ey1 = m['erase_light']
        mask = np.zeros(plate.shape[:2], np.uint8)
        roi = plate[ey0:ey1, ex0:ex1].astype(int)
        mask[ey0:ey1, ex0:ex1] = ((roi.min(2) > 150)).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        plate = cv2.inpaint(plate, mask, 7, cv2.INPAINT_TELEA)
    if 'erase_dark' in m:
        ex0, ey0, ex1, ey1 = m['erase_dark']
        mask = np.zeros(plate.shape[:2], np.uint8)
        roi = plate[ey0:ey1, ex0:ex1].astype(int).sum(2)
        mask[ey0:ey1, ex0:ex1] = (roi < 150).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((17, 17), np.uint8))
        plate = cv2.inpaint(plate, mask, 9, cv2.INPAINT_TELEA)
        # inpaint leaves a smooth smear; add back matching grain so it reads as terrain
        # texture: band-pass noise matched to the detail level of a clean forest patch
        pf = plate.astype(np.float32)
        hp = pf - cv2.GaussianBlur(pf, (0, 0), 3)
        tx0, ty0, tx1, ty1 = m['texture_patch']
        target = hp[ty0:ty1, tx0:tx1].std()
        rng = np.random.default_rng(7)
        n = rng.normal(0, 1, plate.shape[:2]).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 1.0) * 0.7 + cv2.GaussianBlur(n, (0, 0), 2.5) * 2.0
        n = n / n.std() * target
        donor = n[..., None] * np.array([0.9, 1.0, 0.8], np.float32)  # greenish luminance grain
        mf = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 3)[..., None]
        plate = np.clip(pf + donor * mf, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{OUT}/{name}.png', plate)
    print(name, plate.shape)
