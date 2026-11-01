"""Render the 1-minute route-instruction video from extracted source frames.

Every output frame is a pure function of time: it reads source frames from the
frames dir, the clean map plates, and edit.json (scene timing, labels, cards).

Usage:
  python3 -I render.py <frames_dir> <out.mp4>            # full render
  python3 -I render.py <frames_dir> <stills_dir> --stills t1,t2,...  # PNG stills at output seconds
"""
import sys, os, json, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
CFG = json.load(open(os.path.join(ROOT, 'edit.json')))
FRAMES = sys.argv[1]
FORMAT = sys.argv[sys.argv.index('--format') + 1] if '--format' in sys.argv else 'landscape'
VERT = FORMAT == 'vertical'
W, H = (1080, 1920) if VERT else tuple(CFG['size'])
FPS = CFG['fps']
BEAT = 60.0 / CFG['bpm']
TOTAL = CFG['total_beats'] * BEAT
NFR = int(round(TOTAL * FPS))

# ---------- palette & fonts ----------
BG = (13, 16, 22)
CARD = (22, 27, 36)
LINE = (44, 52, 66)
TXT = (245, 247, 250)
TXT2 = (150, 160, 176)
ACC = (255, 184, 0)        # amber: route, turn sign, active step
ACC_DK = (40, 28, 0)
OK = (61, 220, 132)
FD = '/usr/share/fonts/opentype/inter/'
_fc = {}
def font(w, s):
    k = (w, s)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FD + {'b': 'Inter-Bold.otf', 's': 'Inter-SemiBold.otf',
                                          'm': 'Inter-Medium.otf', 'r': 'Inter-Regular.otf',
                                          'x': 'InterDisplay-Bold.otf'}[w], s)
    return _fc[k]

# ---------- layout ----------
if VERT:   # 9:16 — wide map on top, dashcam + steps/instruction below
    M = 36
    MAPR = (M, 60, W - 2 * M, 740)
    DASH = (M, 824, 576, 1036)
    INFO = (M + 576 + 20, 824, W - M - (M + 576 + 20), 1036)
    MAP_SCALE_CAP = 2.0
else:      # 16:9 — dashcam | map | steps
    M = 44
    DASH = (M, 54, 551, 972)                 # x, y, w, h
    MAPR = (M + 551 + 24, 54, 700, 972)
    INFO = (MAPR[0] + 700 + 24, 54, W - M - (MAPR[0] + 700 + 24), 972)
    MAP_SCALE_CAP = 9.0
RAD = 22

def ease(x):  # easeOutCubic, clamped
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3
def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)
def b2t(b): return b * BEAT

# ---------- time map (monotone cubic through keyframes) ----------
km = np.array(CFG['timemap'], float)
kx, ky = km[:, 0] * BEAT, km[:, 1]
d = np.diff(ky) / np.diff(kx)
m = np.zeros(len(kx))
m[0], m[-1] = d[0], d[-1]
for i in range(1, len(kx) - 1):
    m[i] = 0 if d[i - 1] * d[i] <= 0 else 2 / (1 / d[i - 1] + 1 / d[i])
def src_time(t):
    t = min(max(t, kx[0]), kx[-1])
    i = min(np.searchsorted(kx, t, 'right') - 1, len(kx) - 2)
    h = kx[i + 1] - kx[i]; s = (t - kx[i]) / h
    h00 = 2*s**3 - 3*s**2 + 1; h10 = s**3 - 2*s**2 + s; h01 = -2*s**3 + 3*s**2; h11 = s**3 - s**2
    return h00*ky[i] + h10*h*m[i] + h01*ky[i+1] + h11*h*m[i+1]
def speed(t):
    e = 0.02
    return (src_time(t + e) - src_time(t - e)) / (2 * e)

NSRC = len([f for f in os.listdir(FRAMES) if f.endswith('.jpg')])
_cache = {}
def src_frame(i):
    i = int(min(max(i, 0), NSRC - 1))
    if i not in _cache:
        if len(_cache) > 64: _cache.clear()
        _cache[i] = cv2.imread(f'{FRAMES}/{i:05d}.jpg')
    return _cache[i]

def dash_image(t):
    """Dashcam crop at output time t, with motion-blend when fast-forwarding."""
    st = src_time(t); v = speed(t)
    x0, y0, x1, y1 = CFG['dash_crop']
    f = st * FPS
    n = 1 if v < 1.6 else min(6, int(math.ceil(v / 1.2)))
    if n == 1:
        img = src_frame(round(f))[y0:y1, x0:x1].astype(np.float32)
    else:
        offs = np.linspace(-v / 2, v / 2, n)
        img = np.mean([src_frame(round(f + o))[y0:y1, x0:x1].astype(np.float32) for o in offs], axis=0)
    return img, st, v

# ---------- map segments, pin paths ----------
PL = {k: cv2.imread(os.path.join(ROOT, 'assets/plates', f'{k}.png')) for k in CFG['maps']}
order = sorted(CFG['maps'], key=lambda k: CFG['maps'][k]['from_frame'])

def seg_of(src_t):
    f = src_t * FPS; k = order[0]
    for kk in order:
        if f >= CFG['maps'][kk]['from_frame']: k = kk
    return k

ROUTES = json.load(open(os.path.join(ROOT, 'assets', 'routes.json')))
RP = {}
for k, r in ROUTES.items():
    path = np.array(r['path'], float)
    lens = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    RP[k] = (path, lens, np.array(r['frames'], float), np.array(r['L'], float))

# C: timing re-synced to the dashcam (turn + arrival) instead of the keyframed pin
cr = CFG['c_route']
_cp, _cl, _cf, _cL = RP['C']
C_L0 = float(np.interp(cr['start_src'] * FPS, _cf, _cL))
C_LC = ROUTES['C']['L_corner']
C_LE = float(_cl[-1])
CPATH = _cp
CORNER = int(np.argmin(np.abs(_cl - C_LC)))

def c_pos_len(src_t):
    a, c, z = cr['start_src'], cr['corner_src'], cr['arrive_src']
    if src_t <= c:
        return C_L0 + (C_LC - C_L0) * min(max((src_t - a) / (c - a), 0), 1)
    return C_LC + (C_LE - C_LC) * min(max((src_t - c) / (z - c), 0), 1)

def path_point(path, lens, L):
    i = min(np.searchsorted(lens, L, 'right') - 1, len(path) - 2)
    i = max(i, 0)
    s = (L - lens[i]) / max(lens[i + 1] - lens[i], 1e-6)
    return path[i] + (path[i + 1] - path[i]) * min(max(s, 0), 1)

def pin_at(k, src_t):
    """Returns (pin xy, full path, travelled length, lens) in plate coords."""
    path, lens, fr, Ls = RP[k]
    L = c_pos_len(src_t) if k == 'C' else float(np.interp(src_t * FPS, fr, Ls))
    return path_point(path, lens, L), path, L, lens

def map_view(k, t):
    """Scale + camera center (plate coords) for segment k at output t."""
    pl = PL[k]; ph, pw = pl.shape[:2]
    mw, mh = MAPR[2], MAPR[3]
    s = min(max(mw / pw, mh / ph), MAP_SCALE_CAP)
    # camera follows the pin, smoothed over +-0.6s of output time
    ts = np.linspace(t - 0.6, t + 0.6, 7)
    cs = []
    for tt in ts:
        st = src_time(tt)
        if seg_of(st) != k:
            continue
        cs.append(pin_at(k, st)[0])
    c = np.mean(cs, axis=0) if cs else pin_at(k, src_time(t))[0]
    if k == 'C':  # bias forward along the route so the turn/destination is visible ahead
        c = c + np.array([90, -20])
    # arrival: zoom out to the whole final map
    zo = ease_io((t - b2t(72)) / (b2t(74.5) - b2t(72))) if k == 'C' else 0
    s_fit = min(mw / pw, mh / ph)
    s = s + (min(s_fit, s) - s) * zo
    c = c + (np.array([pw / 2, ph / 2]) - c) * zo
    # clamp so the window stays inside the plate (when it is bigger than the window)
    hw, hh = mw / 2 / s, mh / 2 / s
    cx = min(max(c[0], hw), pw - hw) if pw > 2 * hw else pw / 2
    cy = min(max(c[1], hh), ph - hh) if ph > 2 * hh else ph / 2
    return s, np.array([cx, cy])

def draw_map(t):
    st = src_time(t); k = seg_of(st)
    pl = PL[k]; mw, mh = MAPR[2], MAPR[3]
    s, c = map_view(k, t)
    # affine: plate -> card
    A = np.array([[s, 0, mw / 2 - c[0] * s], [0, s, mh / 2 - c[1] * s]], np.float32)
    img = cv2.warpAffine(pl, A, (mw, mh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=CARD[::-1])
    ph_, pw_ = pl.shape[:2]
    if s * pw_ < mw - 1 or s * ph_ < mh - 1:
        sc = max(mw / pw_, mh / ph_) * 1.1
        Ab = np.array([[sc, 0, mw / 2 - pw_ / 2 * sc], [0, sc, mh / 2 - ph_ / 2 * sc]], np.float32)
        bgm = cv2.warpAffine(pl, Ab, (mw, mh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bgm = (cv2.GaussianBlur(bgm, (0, 0), 22).astype(np.float32) * 0.45).astype(np.uint8)
        inb = cv2.warpAffine(np.full((ph_, pw_), 255, np.uint8), A, (mw, mh), flags=cv2.INTER_LINEAR)
        inb = (inb.astype(np.float32) / 255)[..., None]
        img = (img * inb + bgm * (1 - inb)).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (0, 0), 1.4)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)  # light unsharp after upscale
    img = (img.astype(np.float32) * 0.86).astype(np.uint8)  # dim so overlays pop
    pin, path, L, lens = pin_at(k, st)
    P = lambda p: (p * s + np.array([mw / 2 - c[0] * s, mh / 2 - c[1] * s]))
    SS = 4  # subpixel shift for anti-aliased cv2 drawing
    def poly(pts, col, wdt):
        q = (P(np.asarray(pts)) * (1 << 2)).astype(np.int32)
        cv2.polylines(img, [q], False, col[::-1], wdt, cv2.LINE_AA, shift=2)
    # route: future (white, thin) then travelled (amber, thick with dark casing)
    poly(path, (255, 255, 255), 5)
    done = list(path[lens < L]) + [pin]
    if len(done) > 1:
        poly(done, (20, 14, 0), 15)
        poly(done, ACC, 9)
    # turn marker on C (pulse ring at the corner, until we are well past it)
    if k == 'C':
        cpt = P(CPATH[CORNER])
        ph_ = (t / BEAT) % 1
        if st < cr['corner_src'] + 3:
            r = 22 + 26 * ph_
            ov = img.copy()
            cv2.circle(ov, tuple((cpt * 4).astype(int)), int(r * 4), ACC[::-1], 4, cv2.LINE_AA, shift=2)
            img = cv2.addWeighted(ov, 1 - ph_, img, ph_, 0)
            cv2.circle(img, tuple((cpt * 4).astype(int)), 14 * 4, ACC[::-1], -1, cv2.LINE_AA, shift=2)
            draw_turn_glyph_cv(img, cpt, 11)
        # destination ring
        dpt = P(CPATH[-1])
        if st > 120:
            r = 26 + 30 * ph_
            ov = img.copy()
            cv2.circle(ov, tuple((dpt * 4).astype(int)), int(r * 4), (255, 255, 255), 4, cv2.LINE_AA, shift=2)
            img = cv2.addWeighted(ov, 1 - ph_, img, ph_, 0)
    # pin: glow + white ring + amber core
    pp = P(pin)
    q = tuple((pp * 4).astype(int))
    ov = img.copy()
    cv2.circle(ov, q, 30 * 4, ACC[::-1], -1, cv2.LINE_AA, shift=2)
    img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)
    cv2.circle(img, q, 17 * 4, (255, 255, 255), -1, cv2.LINE_AA, shift=2)
    cv2.circle(img, q, 11 * 4, ACC[::-1], -1, cv2.LINE_AA, shift=2)
    return img, k

def draw_turn_glyph_cv(img, center, sz):
    """Tiny 'turn' dot glyph for the map corner (dark arrow tip)."""
    x, y = center
    cv2.circle(img, (int(x * 4), int(y * 4)), int(sz * 0.45 * 4), (20, 14, 0)[::-1], -1, cv2.LINE_AA, shift=2)

# ---------- PIL helpers ----------
def rounded_mask(w, h, r):
    mk = Image.new('L', (w * 2, h * 2), 0)
    ImageDraw.Draw(mk).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], r * 2, fill=255)
    return mk.resize((w, h), Image.LANCZOS)

MASK_DASH = rounded_mask(DASH[2], DASH[3], RAD)
MASK_MAP = rounded_mask(MAPR[2], MAPR[3], RAD)

def arrow_icon(kind, size, col, bg=None):
    """Vector arrows drawn at 4x then downsampled. kind: straight|left|right_look|pin|check."""
    S = size * 4
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    if bg:
        dr.rounded_rectangle([0, 0, S - 1, S - 1], int(S * 0.2), fill=bg)
    u = S / 100
    w = 13 * u
    if kind == 'straight':
        dr.rectangle([50 * u - w / 2, 40 * u, 50 * u + w / 2, 84 * u], fill=col)
        dr.polygon([(50 * u, 14 * u), (76 * u, 44 * u), (24 * u, 44 * u)], fill=col)
    elif kind == 'left':  # driver's view: go up, then turn left
        dr.rectangle([62 * u - w / 2, 46 * u, 62 * u + w / 2, 86 * u], fill=col)
        dr.rectangle([36 * u, 40 * u - w / 2, 62 * u + w / 2, 40 * u + w / 2], fill=col)
        dr.rectangle([62 * u - w / 2, 40 * u - w / 2, 62 * u + w / 2, 60 * u], fill=col)
        dr.polygon([(12 * u, 40 * u), (40 * u, 16 * u), (40 * u, 64 * u)], fill=col)
    elif kind == 'right_look':  # eye-ish: arrow pointing right
        dr.rectangle([16 * u, 50 * u - w / 2, 62 * u, 50 * u + w / 2], fill=col)
        dr.polygon([(88 * u, 50 * u), (58 * u, 24 * u), (58 * u, 76 * u)], fill=col)
    elif kind == 'left_look':
        dr.rectangle([38 * u, 50 * u - w / 2, 84 * u, 50 * u + w / 2], fill=col)
        dr.polygon([(12 * u, 50 * u), (42 * u, 24 * u), (42 * u, 76 * u)], fill=col)
    elif kind == 'pin':
        dr.ellipse([28 * u, 12 * u, 72 * u, 56 * u], fill=col)
        dr.polygon([(31 * u, 44 * u), (69 * u, 44 * u), (50 * u, 88 * u)], fill=col)
        dr.ellipse([41 * u, 25 * u, 59 * u, 43 * u], fill=bg or (0, 0, 0, 0))
    elif kind == 'check':
        dr.line([(20 * u, 52 * u), (42 * u, 74 * u), (82 * u, 28 * u)], fill=col, width=int(14 * u), joint='curve')
    return im.resize((size, size), Image.LANCZOS)

def text_w(txt, f):
    return f.getbbox(txt)[2]

def fit_font(txt, w, maxw, start):
    s = start
    while s > 20 and text_w(txt, font(w, s)) > maxw:
        s -= 2
    return font(w, s)

def alpha_paste(base, im, xy, a):
    if a <= 0: return
    if a < 1:
        im = im.copy()
        al = im.getchannel('A').point(lambda v: int(v * a))
        im.putalpha(al)
    base.alpha_composite(im, (int(xy[0]), int(xy[1])))

def text_layer(size, items):
    """items: list of (xy, text, font, color(rgba))"""
    im = Image.new('RGBA', size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    for xy, txt, f, col in items:
        dr.text(xy, txt, font=f, fill=col)
    return im

# ---------- static background ----------
def make_bg():
    g = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W * 0.7) / W, (yy - H * 0.2) / H)
    base = np.array(BG, np.float32)
    g[:] = base
    g += (np.clip(0.5 - r, 0, 1)[..., None] * np.array([10, 14, 22], np.float32))
    return Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).convert('RGBA')
BGIMG = make_bg()

# ---------- info column ----------
STEPS = CFG['steps']
CARDS = CFG['cards']

def draw_info(base, t, appear):
    x, y, w, h = INFO
    b = t / BEAT
    lay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    # header
    dr.text((4, 0), 'МАРШРУТ', font=font('s', 22), fill=TXT2 + (255,))
    dr.text((4, 30), 'до RE:FRAME', font=font('x', 46), fill=TXT + (255,))
    # steps
    sy = 130; rowh = 92; cx = 30
    # connector line
    dr.line([(cx, sy + 30), (cx, sy + rowh * (len(STEPS) - 1) + 30)], fill=LINE + (255,), width=3)
    for i, st in enumerate(STEPS):
        b0, b1 = st['beats']
        ry = sy + i * rowh
        active = b0 <= b < b1
        done = b >= b1
        if active:
            k = ease((b - b0) / 0.6)
            dr.rounded_rectangle([0, ry - 8, w, ry + rowh - 16], 16,
                                 fill=(ACC[0], ACC[1], ACC[2], int(30 * k)),
                                 outline=(ACC[0], ACC[1], ACC[2], int(110 * k)), width=2)
        r = 22
        if done:
            dr.ellipse([cx - r, ry + 30 - r, cx + r, ry + 30 + r], fill=(58, 66, 82, 255))
            dr.line([(cx - 9, ry + 30), (cx - 2, ry + 37), (cx + 10, ry + 22)], fill=TXT + (255,), width=4, joint='curve')
            tc, sc = TXT + (150,), TXT2 + (140,)
        elif active:
            dr.ellipse([cx - r, ry + 30 - r, cx + r, ry + 30 + r], fill=ACC + (255,))
            n = str(i + 1); f = font('b', 24)
            dr.text((cx - text_w(n, f) / 2, ry + 15), n, font=f, fill=ACC_DK + (255,))
            tc, sc = TXT + (255,), (230, 236, 244, 255)
        else:
            dr.ellipse([cx - r, ry + 30 - r, cx + r, ry + 30 + r], fill=CARD + (255,), outline=LINE + (255,), width=3)
            n = str(i + 1); f = font('s', 22)
            dr.text((cx - text_w(n, f) / 2, ry + 16), n, font=f, fill=TXT2 + (255,))
            tc, sc = TXT + (190,), TXT2 + (200,)
        dr.text((66, ry + 2), st['title'], font=font('b' if active else 's', 30), fill=tc)
        dr.text((66, ry + 40), st['sub'], font=font('r', 22), fill=sc)
    # instruction card
    cy0 = 610; ch = h - cy0
    cur = None; prev = None
    for cdef in CARDS:
        if cdef['beats'][0] <= b < cdef['beats'][1]: cur = cdef
    if cur is None: cur = CARDS[-1] if b >= CARDS[-1]['beats'][0] else CARDS[0]
    accent = cur.get('accent')
    k_in = ease((b - cur['beats'][0]) / 0.55)
    pulse = 0
    if cur.get('pulse'):
        pulse = 0.5 + 0.5 * math.cos(2 * math.pi * ((b - cur['beats'][0]) % 1))
    fill = ACC if accent else CARD
    dr.rounded_rectangle([0, cy0, w, h], 22, fill=fill + (255,),
                         outline=(255, 255, 255, 40 if not accent else 0), width=2)
    if accent and pulse:
        dr.rounded_rectangle([-4 * pulse, cy0 - 4 * pulse, w + 4 * pulse, h + 4 * pulse], 26,
                             outline=ACC + (int(160 * pulse),), width=4)
    base.alpha_composite(lay, (x, y)) if appear >= 1 else alpha_paste(base, lay, (x, y + 30 * (1 - appear)), appear)
    # card content (animated in)
    tcol = ACC_DK if accent else TXT
    scol = (70, 50, 0) if accent else TXT2
    isz = 150 if accent else 120
    ibg = (25, 18, 0, 255) if accent else (40, 47, 60, 255)
    icol = ACC + (255,) if accent else (TXT + (255,) if cur['icon'] != 'check' else OK + (255,))
    icon = arrow_icon(cur['icon'], isz, icol, ibg)
    if cur.get('pulse'):
        sc_ = 1 + 0.06 * pulse
        icon = icon.resize((int(isz * sc_), int(isz * sc_)), Image.LANCZOS)
    content = Image.new('RGBA', (w, ch), (0, 0, 0, 0))
    cd = ImageDraw.Draw(content)
    cd.text((32, 26), 'СЕЙЧАС', font=font('s', 20), fill=(scol if accent else TXT2) + (255,))
    stacked = w < 480
    if stacked:   # narrow column (vertical format): icon on top, text below
        iy = 70
        content.alpha_composite(icon, (int(32 + (isz - icon.width) / 2), int(iy + (isz - icon.height) / 2)))
        tx, ty = 32, iy + isz + 22
        tf = fit_font(cur['title'], 'x', w - 64, 58 if accent else 50)
        cd.text((tx, ty), cur['title'], font=tf, fill=tcol + (255,))
        sy_ = ty + tf.size + 14
        maxw = w - 64
    else:
        content.alpha_composite(icon, (int(32 + (isz - icon.width) / 2), int((ch - icon.height) / 2 - 34)))
        tx = 32 + isz + 26
        tf = fit_font(cur['title'], 'x', w - tx - 20, 56 if accent else 50)
        cd.text((tx, ch / 2 - 34 - 46), cur['title'], font=tf, fill=tcol + (255,))
        sy_ = ch / 2 - 34 + 16
        maxw = w - tx - 20
    # sub may wrap
    sf = font('m', 26)
    words = cur['sub'].split(); lines = ['']
    for wd in words:
        trial = (lines[-1] + ' ' + wd).strip()
        if text_w(trial, sf) > maxw and lines[-1]: lines.append(wd)
        else: lines[-1] = trial
    for li, ln in enumerate(lines):
        cd.text((tx, sy_ + li * 34), ln, font=sf, fill=scol + (255,))
    # beat-progress bar at card bottom
    pf = min(max((b - cur['beats'][0]) / (cur['beats'][1] - cur['beats'][0]), 0), 1)
    cd.rounded_rectangle([32, ch - 40, w - 32, ch - 34], 3, fill=(0, 0, 0, 50) if accent else LINE + (255,))
    cd.rounded_rectangle([32, ch - 40, 32 + (w - 64) * pf, ch - 34], 3, fill=(ACC_DK + (200,)) if accent else ACC + (255,))
    a = k_in * appear
    alpha_paste(base, content, (x, y + cy0 + 24 * (1 - k_in) + 30 * (1 - appear)), a)

# ---------- dashcam overlays ----------
def ff_chip(v):
    txt = f'×{v:.0f}' if v >= 2.5 else f'×{v:.1f}'
    f = font('b', 26)
    tw = text_w(txt, f)
    w_, h_ = 56 + tw + 24, 48
    im = Image.new('RGBA', (w_, h_), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, w_ - 1, h_ - 1], 24, fill=(10, 12, 16, 170))
    for ox in (18, 32):
        dr.polygon([(ox, 15), (ox + 13, 24), (ox, 33)], fill=TXT + (255,))
    dr.text((56, 9), txt, font=f, fill=TXT + (255,))
    return im

def turn_hud(t, st):
    """Big amber 'НАЛЕВО' HUD with sweeping chevrons, driver's perspective."""
    b = t / BEAT
    hb0, hb1 = CFG['turn_hud']['beats']
    if not (hb0 <= b < hb1 + 0.5): return None, 0
    a = ease((b - hb0) / 0.6) * (1 - ease((b - hb1) / 0.5))
    now = st >= CFG['turn_hud']['src_now']
    w_, h_ = 470, 150
    im = Image.new('RGBA', (w_ * 2, h_ * 2), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, w_ * 2 - 1, h_ * 2 - 1], 40, fill=ACC + (245,))
    # chevrons sweeping left
    ph = (b * 2) % 1
    for i in range(3):
        cx = 150 - i * 44 - ph * 44 + 88
        al = int(255 * (0.35 + 0.65 * (1 - abs((i + ph) / 3 - 0.5) * 2) if True else 255))
        al = max(60, min(255, al))
        pts = [(cx + 34, 70), (cx - 6, 150), (cx + 34, 230), (cx + 62, 230), (cx + 22, 150), (cx + 62, 70)]
        dr.polygon(pts, fill=ACC_DK + (al,))
    txt = 'НАЛЕВО' if now else 'СКОРО НАЛЕВО'
    f = fit_font(txt, 'x', w_ * 2 - 340, 104)
    dr.text((300, 150 - f.getbbox(txt)[3] / 2 - 4), txt, font=f, fill=ACC_DK + (255,))
    im = im.resize((w_, h_), Image.LANCZOS)
    if now:
        pulse = 0.5 + 0.5 * math.cos(2 * math.pi * (b % 1))
        s = 1 + 0.05 * pulse
        im = im.resize((int(w_ * s), int(h_ * s)), Image.LANCZOS)
    return im, a

def caption_chip(txt, sub=None):
    f = font('b', 30); f2 = font('m', 22)
    tw = max(text_w(txt, f), text_w(sub, f2) if sub else 0)
    w_, h_ = tw + 48, 64 if not sub else 96
    im = Image.new('RGBA', (w_, h_), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, w_ - 1, h_ - 1], 18, fill=(10, 12, 16, 190))
    dr.text((24, 12), txt, font=f, fill=TXT + (255,))
    if sub: dr.text((24, 54), sub, font=f2, fill=TXT2 + (255,))
    return im

# ---------- frame ----------
def frame(i):
    t = i / FPS
    b = t / BEAT
    base = BGIMG.copy()
    dimg, st, v = dash_image(t)
    dimg = np.clip(dimg, 0, 255).astype(np.uint8)
    ib0, ib1 = CFG['intro']['beats']
    appear = ease((b - (ib1 - 0.6)) / 1.0)   # panels assemble at the end of the intro

    # --- intro: full-bleed blurred drive + title
    if b < ib1 + 0.6:
        ia = 1 - ease((b - (ib1 - 0.6)) / 0.9)
        sc_ = max(W / dimg.shape[1], (H + 200) / dimg.shape[0])
        full = cv2.resize(dimg, (int(dimg.shape[1] * sc_), int(dimg.shape[0] * sc_)), interpolation=cv2.INTER_LINEAR)
        ox = (full.shape[1] - W) // 2
        oy = min((full.shape[0] - H) // 2 + int(160 * (b / ib1)), full.shape[0] - H)
        full = full[oy:oy + H, ox:ox + W]
        full = cv2.GaussianBlur(full, (0, 0), 18)
        full = (full.astype(np.float32) * 0.45).astype(np.uint8)
        bgf = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)).convert('RGBA')
        base = Image.blend(base, bgf, ia)
        # title + route chips
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(lay)
        k1 = ease((b - 0.3) / 1.2); k2 = ease((b - 1.2) / 1.2)
        names = ['А-180', 'Кемполово', 'налево', 'вдоль поля', 'RE:FRAME']
        cf = font('s', 32)
        if VERT:
            tlines = ['Как доехать', 'до RE:FRAME']; tf = font('x', 116)
            for li, tl in enumerate(tlines):
                dr.text(((W - text_w(tl, tf)) / 2, 520 + li * 136 + 30 * (1 - k1)), tl, font=tf, fill=TXT + (int(255 * k1),))
            sub = CFG['intro']['sub']; sf = font('m', 44)
            dr.text(((W - text_w(sub, sf)) / 2, 820 + 20 * (1 - k1)), sub, font=sf, fill=ACC + (int(255 * k1),))
            cf = font('s', 38)
            yy = 960
            for j, n in enumerate(names):
                kj = ease((b - 1.6 - j * 0.5) / 0.8)
                cw = text_w(n, cf) + 64
                x0 = (W - cw) / 2
                if kj > 0:
                    isturn = n == 'налево'
                    fillc = ACC + (int(255 * kj),) if isturn else (255, 255, 255, int(28 * kj))
                    oy = 16 * (1 - kj)
                    dr.rounded_rectangle([x0, yy + oy, x0 + cw, yy + 76 + oy], 38, fill=fillc)
                    dr.text((x0 + 32, yy + 14 + oy), n, font=cf, fill=(ACC_DK if isturn else TXT) + (int(255 * kj),))
                    if j < len(names) - 1:
                        ax = W / 2
                        dr.polygon([(ax - 10, yy + 92), (ax + 10, yy + 92), (ax, yy + 106)], fill=TXT2 + (int(255 * kj),))
                yy += 76 + 38
        else:
            ttl = CFG['intro']['title']; tf = font('x', 104)
            dr.text(((W - text_w(ttl, tf)) / 2, 330 + 30 * (1 - k1)), ttl, font=tf, fill=TXT + (int(255 * k1),))
            sub = CFG['intro']['sub']; sf = font('m', 40)
            dr.text(((W - text_w(sub, sf)) / 2, 470 + 20 * (1 - k1)), sub, font=sf, fill=ACC + (int(255 * k1),))
            gaps = 64
            widths = [text_w(n, cf) + 48 for n in names]
            tot = sum(widths) + gaps * (len(names) - 1)
            x0 = (W - tot) / 2; yy = 600
            for j, (n, cw) in enumerate(zip(names, widths)):
                kj = ease((b - 1.6 - j * 0.5) / 0.8)
                if kj <= 0: x0 += cw + gaps; continue
                isturn = n == 'налево'
                fillc = ACC + (int(255 * kj),) if isturn else (255, 255, 255, int(28 * kj))
                dr.rounded_rectangle([x0, yy + 16 * (1 - kj), x0 + cw, yy + 64 + 16 * (1 - kj)], 32, fill=fillc)
                dr.text((x0 + 24, yy + 12 + 16 * (1 - kj)), n, font=cf, fill=(ACC_DK if isturn else TXT) + (int(255 * kj),))
                if j < len(names) - 1:
                    ax = x0 + cw + gaps / 2
                    dr.polygon([(ax - 8, yy + 22), (ax + 8, yy + 32), (ax - 8, yy + 42)], fill=TXT2 + (int(255 * kj),))
                x0 += cw + gaps
        alpha_paste(base, lay, (0, -40 * (1 - ia)), ia)

    if appear > 0:
        # --- dashcam card
        dx, dy, dw, dh = DASH
        dimg2 = cv2.resize(dimg, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
        dimg2 = cv2.addWeighted(dimg2, 1.25, cv2.GaussianBlur(dimg2, (0, 0), 1.6), -0.25, 0)
        dpil = Image.fromarray(cv2.cvtColor(dimg2, cv2.COLOR_BGR2RGB)).convert('RGBA')
        dpil.putalpha(MASK_DASH)
        # chips
        if v > 1.3:
            chip = ff_chip(v)
            dpil.alpha_composite(chip, (20, dh - 68))
        hud, ha = turn_hud(t, st)
        if hud is not None:
            alpha_paste(dpil, hud, ((dw - hud.width) / 2, 60 + (150 - hud.height) / 2), ha)
        if 66 <= b < 72.5:
            ka = ease((b - 66) / 0.6) * (1 - ease((b - 72) / 0.5))
            ch_ = caption_chip('Слева — ворота RE:FRAME', 'с рисунком')
            alpha_paste(dpil, ch_, ((dw - ch_.width) / 2, 60), ka)
        alpha_paste(base, dpil, (dx, dy + 40 * (1 - appear)), appear)

        # --- map card (dip at map switches)
        mimg, k = draw_map(t)
        cuts = [CFG['maps'][kk]['from_frame'] / FPS for kk in order[1:]]
        dip = 1.0
        for cs in cuts:
            # find output time of this cut
            lo, hi = 0, TOTAL
            for _ in range(40):
                mid = (lo + hi) / 2
                if src_time(mid) < cs: lo = mid
                else: hi = mid
            dt = abs(t - lo)
            dip = min(dip, min(1, dt / 0.2))
        mp = Image.fromarray(cv2.cvtColor(mimg, cv2.COLOR_BGR2RGB)).convert('RGBA')
        if dip < 1:
            mp = Image.blend(Image.new('RGBA', mp.size, CARD + (255,)), mp, ease_io(dip))
        mp.putalpha(MASK_MAP)
        mx, my, mw, mh = MAPR
        alpha_paste(base, mp, (mx, my + 40 * (1 - ease((b - (ib1 - 0.4)) / 1.0))), ease((b - (ib1 - 0.4)) / 1.0))

        # --- info column
        app3 = ease((b - (ib1 - 0.2)) / 1.0)
        if app3 > 0:
            draw_info(base, t, app3)

    # outro fade
    fo = ease((t - (TOTAL - 1.2)) / 1.2)
    out = np.array(base.convert('RGB'))
    if fo > 0:
        out = (out.astype(np.float32) * (1 - fo)).astype(np.uint8)
    return out

def render_still(args):
    t, path = args
    Image.fromarray(frame(int(round(t * FPS)))).save(path)

if __name__ == '__main__':
    if '--stills' in sys.argv:
        out = sys.argv[2]; os.makedirs(out, exist_ok=True)
        ts = [float(x) for x in sys.argv[sys.argv.index('--stills') + 1].split(',')]
        with Pool(4) as p:
            p.map(render_still, [(t, f'{out}/s_{t:06.2f}.png') for t in ts])
        sys.exit()
    import subprocess
    outp = sys.argv[2]
    src = os.path.join(ROOT, 'source', os.path.basename(CFG['source']))
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                           '-r', str(FPS), '-i', '-',
                           '-ss', str(CFG['audio_offset']), '-t', f'{TOTAL:.3f}', '-i', src,
                           '-map', '0:v', '-map', '1:a',
                           '-af', f'afade=t=in:d=0.3,afade=t=out:st={TOTAL - 2.5:.3f}:d=2.5,loudnorm=I=-16:TP=-1.5:LRA=11',
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', '-shortest', outp],
                          stdin=subprocess.PIPE)
    with Pool(4) as p:
        for n, fr_ in enumerate(p.imap(frame, range(NFR), chunksize=8)):
            ff.stdin.write(fr_.tobytes())
            if n % 150 == 0: print(f'{n}/{NFR}', flush=True)
    ff.stdin.close(); ff.wait()
    print('done', outp)
