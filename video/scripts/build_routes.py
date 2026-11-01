"""Snap the route to the actual road on each clean map plate.

The source pin track is hand-keyframed and drifts off the road. For every map we:
  1. take the pin track as a rough guide (optionally extended to the plate edge,
     so the already-driven part of the road is visible),
  2. for each point search along the normal for the brightest road band
     (yellow asphalt / white dirt road are the brightest lines on the satellite),
  3. smooth the offsets and resample evenly,
  4. project the pin of every source frame onto the snapped path (arc length, monotone).
Output: assets/routes.json (+ work/routes_debug.jpg).
Usage: python3 -I build_routes.py
"""
import os, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
CFG = json.load(open(os.path.join(ROOT, 'edit.json')))
track = json.load(open(os.path.join(ROOT, 'assets', 'pin_track.json')))
RC = CFG['routes']


def resample(path, step=3.0):
    seg = np.hypot(*np.diff(path, axis=0).T)
    L = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(L[-1] / step), 2)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, path[:, 0]), np.interp(s, L, path[:, 1])], 1)


def smooth(path, sigma):
    if sigma <= 0: return path
    k = int(sigma * 3) * 2 + 1
    out = path.copy()
    for j in range(2):
        pad = np.pad(path[:, j], k // 2, mode='reflect', reflect_type='odd')
        g = cv2.getGaussianKernel(k, sigma).ravel()
        out[:, j] = np.convolve(pad, g, 'valid')
    return out


def arclen(path):
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])


def road_score(plate):
    f = cv2.GaussianBlur(plate.astype(np.float32), (0, 0), 1.5)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(2)
    yellow = np.clip(np.minimum(r, g) - b, 0, 80)
    return v + 0.8 * yellow


def snap(path, score, search, sigma, iters=3, penalty=0.6, centroid=False):
    h, w = score.shape
    for _ in range(iters):
        d = np.gradient(path, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        offs = np.arange(-search, search + 1, 0.5)
        best = np.zeros(len(path))
        for i, (p, nn) in enumerate(zip(path, n)):
            q = p[None] + offs[:, None] * nn[None]
            xs = np.clip(q[:, 0], 0, w - 1).astype(np.float32)
            ys = np.clip(q[:, 1], 0, h - 1).astype(np.float32)
            prof = cv2.remap(score, xs[None], ys[None], cv2.INTER_LINEAR)[0]
            prof = np.convolve(prof, np.ones(5) / 5, 'same')
            prof -= penalty * np.abs(offs)  # prefer staying close to the guide
            j = int(np.argmax(prof))
            if centroid:  # centre of the bright band containing the maximum, not its brightest edge
                thr = prof[j] - 0.35 * (prof[j] - np.percentile(prof, 20))
                a_, b_ = j, j
                while a_ > 0 and prof[a_ - 1] >= thr: a_ -= 1
                while b_ < len(prof) - 1 and prof[b_ + 1] >= thr: b_ += 1
                best[i] = (offs[a_] + offs[b_]) / 2
            else:
                best[i] = offs[j]
        best = np.convolve(np.pad(best, 8, mode='edge'), np.ones(17) / 17, 'valid')
        path = smooth(path + best[:, None] * n, sigma)
        search = max(search * 0.6, 4)
    return path


def white_score(plate):
    hsv = cv2.cvtColor(cv2.GaussianBlur(plate, (0, 0), 1.5), cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv[..., 2] - 1.2 * hsv[..., 1]


def trace_rows(plate, seed, y0, y1, win=3, halfw=4, guide=None, gw=0.25):
    """Follow a mostly-vertical bright road row by row from seed=(x,y) to rows y0 and y1."""
    r = cv2.GaussianBlur(plate[..., 2].astype(np.float32), (0, 0), 1.2)
    v = plate.astype(np.float32).min(2)  # penalise pure white (houses, labels) a bit less than green
    sc = r - 0.25 * np.maximum(v - 200, 0)
    ker = np.ones(2 * halfw + 1) / (2 * halfw + 1)
    rows = {}
    sx, sy = seed
    # refine seed
    yy = int(round(sy)); lo = int(max(sx - 25, 0)); hi = int(min(sx + 25, plate.shape[1] - 1))
    prof = np.convolve(sc[yy, lo:hi + 1], ker, 'same'); x = lo + int(np.argmax(prof))
    for direction, end in ((1, y1), (-1, y0)):
        cx = float(x)
        for y in range(yy, end + direction, direction):
            lo = int(max(cx - win - halfw, 0)); hi = int(min(cx + win + halfw, plate.shape[1] - 1))
            prof = np.convolve(sc[y, lo:hi + 1], ker, 'same')
            xs = np.arange(lo, hi + 1)
            okm = np.abs(xs - cx) <= win
            prof = np.where(okm, prof - 0.8 * np.abs(xs - cx), -1e9)
            if guide is not None and guide[:, 1].min() - 20 <= y <= guide[:, 1].max() + 20:
                o = np.argsort(guide[:, 1])
                gx = np.interp(y, guide[o, 1], guide[o, 0])
                prof = prof - gw * np.clip(np.abs(xs - gx) - 14, 0, None) * 4
            cx = 0.6 * xs[int(np.argmax(prof))] + 0.4 * cx
            rows[y] = cx
    ys = np.array(sorted(rows)); xs = np.array([rows[y] for y in ys])
    return np.stack([xs, ys], 1).astype(float)


def pin_frames(k):
    out = []
    for i, p in track.items():
        if p and p[0] == k:
            out.append((int(i), p[1], p[2]))
    out.sort()
    return np.array(out, float)


def extend(path, to_y=None, ext_to_y_end=None):
    """Extend the guide linearly to the top (to_y) / bottom (ext_to_y_end) plate edge."""
    if to_y is not None:
        d = path[min(15, len(path) - 1)] - path[0]
        if d[1] < 0: d = -d
        steps = (path[0][1] - to_y) / max(d[1], 1e-6)
        pre = path[0] - d * np.linspace(steps, 0, 40)[:, None]
        path = np.vstack([pre[:-1], path])
    if ext_to_y_end is not None:
        d = path[-1] - path[max(-16, -len(path))]
        steps = (ext_to_y_end - path[-1][1]) / max(d[1], 1e-6)
        post = path[-1] + d * np.linspace(0, steps, 40)[:, None]
        path = np.vstack([path, post[1:]])
    return path


routes = {}
dbg = []
for k, rc in RC.items():
    plate = cv2.imread(os.path.join(ROOT, 'assets', 'plates', f'{k}.png'))
    h, w = plate.shape[:2]
    sc = road_score(plate)
    pf = pin_frames(k)
    f0, f1 = rc.get('track_frames', [pf[0, 0], pf[-1, 0]])
    sel = pf[(pf[:, 0] >= f0) & (pf[:, 0] <= f1)]
    guide = []
    for p in sel[:, 1:]:
        if not guide or np.hypot(*(p - guide[-1])) > 4: guide.append(p)
    guide = smooth(resample(np.array(guide), 4.0), 2)
    if 'rows' in rc:  # main road: trace it on the plate
        seed = rc.get('seed', guide[len(guide) // 2].tolist())
        y0, y1 = rc['rows']
        gd = guide[guide[:, 1] <= rc.get('guide_cut_y', 1e9)]
        gd = np.vstack([gd, np.array(rc['extra_guide'], float)]) if 'extra_guide' in rc else gd
        road = smooth(resample(trace_rows(plate, seed, y0, y1, guide=gd, gw=rc.get('guide_w', 0.25)), 3.0), rc.get('sigma', 4))
        if rc.get('then_track_from'):  # C: asphalt down to the junction, then the dirt road from the track
            c = np.array(rc['corner_near'], float)
            road = road[road[:, 1] <= c[1]]
            tail = guide[np.hypot(*(guide - c).T).argmin():]
            if 'tail_prefix' in rc:  # explicit junction -> start of the dirt road
                pre = np.array(rc['tail_prefix'], float)
                tail = np.vstack([pre, tail[np.hypot(*(tail - pre[-1]).T).argmin() + 1:]])
            wsc = white_score(plate) if rc.get('tail_score') == 'white' else sc
            tail = resample(smooth(resample(tail, 3.0), 3), 3.0)
            n_fix = len(resample(np.array(rc.get('tail_prefix', [c, c + 0.1])), 3.0))
            if 'tail_suffix' in rc:  # explicit end: last road point -> into the gate
                suf = np.array(rc['tail_suffix'], float)
                tail = np.vstack([tail[:np.hypot(*(tail - suf[0]).T).argmin()], suf])
                tail = resample(tail, 3.0)
            n_end = len(resample(np.array(rc['tail_suffix'], float), 3.0)) if 'tail_suffix' in rc else 0
            snapped = snap(tail, wsc, rc.get('search', 20), rc.get('sigma', 4), penalty=rc.get('snap_penalty', 0.6), centroid=rc.get('centroid', False))
            if n_end:
                w_end = np.clip((len(tail) - n_end - np.arange(len(tail))) / 6, 0, 1)[:, None]
                snapped = snapped * w_end + tail * (1 - w_end)
            w_ = np.clip((np.arange(len(tail)) - n_fix) / 6, 0, 1)[:, None]  # keep the junction fixed
            tail = tail * (1 - w_) + snapped * w_
            path = smooth(resample(np.vstack([road, tail]), 3.0), 2)
        else:
            path = road
    else:
        guide = extend(guide, 0 if rc.get('from_top') else None, h - 1 if rc.get('to_bottom') else None)
        guide = resample(guide, 3.0)
        path = snap(guide, sc, rc.get('search', 20), rc.get('sigma', 4))
    path = path[(path[:, 1] >= -2) & (path[:, 1] <= h + 2) & (path[:, 0] >= -2) & (path[:, 0] <= w + 2)]
    path = resample(path, 3.0)
    L = arclen(path)
    # project every pin sample onto the path -> arc length, monotone
    fr, Ls = [], []
    for f, x, y in pf:
        dd = np.hypot(path[:, 0] - x, path[:, 1] - y)
        fr.append(int(f)); Ls.append(float(L[int(np.argmin(dd))]))
    Ls = np.maximum.accumulate(np.array(Ls))
    r = {'path': np.round(path, 2).tolist(), 'frames': fr, 'L': np.round(Ls, 2).tolist()}
    if 'corner_near' in rc:
        c = np.array(rc['corner_near'], float)
        r['L_corner'] = float(L[int(np.argmin(np.hypot(*(path - c).T)))])
    routes[k] = r
    vis = plate.copy()
    cv2.polylines(vis, [guide.astype(np.int32)], False, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(vis, [path.astype(np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
    dbg.append(cv2.resize(vis, (int(w * 600 / h), 600)))
    print(k, 'points', len(path), 'length', round(L[-1]), 'corner' if 'L_corner' in r else '')

json.dump(routes, open(os.path.join(ROOT, 'assets', 'routes.json'), 'w'))
os.makedirs(os.path.join(ROOT, 'work'), exist_ok=True)
cv2.imwrite(os.path.join(ROOT, 'work', 'routes_debug.jpg'), np.hstack(dbg))
