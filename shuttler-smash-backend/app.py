# import io, math
# from typing import Optional
# from fastapi import FastAPI, UploadFile, File, Form
# from pydantic import BaseModel
# import numpy as np
# import av
# import cv2
# import mediapipe as mp

# app = FastAPI(title="ShuttleX Smash Speed API")

# # Tunables
# DEFAULT_ALPHA = 1.25   # maps wrist/racket peak speed -> estimated shuttle speed
# SMOOTH_WIN = 5         # moving average window for velocity smoothing
# IMPACT_NEIGHBOR = 6    # frames around peak when averaging

# class SmashResult(BaseModel):
#     mode_used: str                 # "no_shuttle" for now
#     max_kmh: float
#     avg_kmh: float
#     confidence: float
#     frames_used: int
#     notes: Optional[str] = None

# def moving_average(x, w):
#     if len(x) < w:
#         return np.array(x, dtype=np.float32)
#     csum = np.cumsum(np.insert(x, 0, 0))
#     out = (csum[w:] - csum[:-w]) / float(w)
#     pad = [out[0]] * (w-1)
#     return np.array(pad + out.tolist(), dtype=np.float32)

# def velocity_series(points_px, px_per_meter, dt):
#     v = []
#     for i in range(1, len(points_px)):
#         x1, y1 = points_px[i-1]
#         x2, y2 = points_px[i]
#         dpx = math.hypot(x2 - x1, y2 - y1)
#         dm  = dpx / max(px_per_meter, 1e-6)
#         v.append(dm / dt)  # m/s
#     return np.array(v, dtype=np.float32)

# def confidence_from_signal(v):
#     if len(v) < 10: return 0.3
#     p95 = float(np.percentile(v, 95))
#     p50 = float(np.percentile(v, 50))
#     if p50 <= 1e-6: return 0.3
#     ratio = min(max(p95 / p50, 1.0), 4.0)  # 1..4 -> 0.25..1.0 scaled
#     return 0.25 + 0.75 * ((ratio - 1.0) / 3.0)

# @app.post("/smash/analyze", response_model=SmashResult)
# async def analyze(
#     video: UploadFile = File(...),
#     px_per_meter: float = Form(...),
#     fps: Optional[float] = Form(None),
#     alpha: Optional[float] = Form(None)
# ):
#     """
#     Estimate smash shuttle speed from a 30s video.
#     If shuttle not visible, uses wrist motion as proxy and maps with alpha.
#     """
#     raw = await video.read()
#     container = av.open(io.BytesIO(raw))
#     vstream = container.streams.video[0]

#     vid_fps = float(fps) if fps else (float(vstream.average_rate) if vstream.average_rate else 30.0)
#     dt = 1.0 / vid_fps

#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

#     wrist_pts = []
#     used_frames = 0

#     for frame in container.decode(vstream):
#         img = frame.to_ndarray(format="bgr24")
#         h, w = img.shape[:2]
#         # downscale for speed
#         scale = 720 / max(h, w) if max(h, w) > 720 else 1.0
#         if scale < 1.0:
#             img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
#             h, w = img.shape[:2]

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         res = pose.process(img_rgb)

#         if res.pose_landmarks:
#             lm = res.pose_landmarks.landmark
#             rw = lm[16]  # right wrist
#             if 0 <= rw.x <= 1 and 0 <= rw.y <= 1:
#                 wrist_pts.append((rw.x * w, rw.y * h))
#                 used_frames += 1
#             else:
#                 wrist_pts.append(None)
#         else:
#             wrist_pts.append(None)

#     pose.close()

#     valid = [(i,p) for i,p in enumerate(wrist_pts) if p is not None]
#     if len(valid) < 6:
#         return SmashResult(
#             mode_used="no_shuttle",
#             max_kmh=0.0, avg_kmh=0.0, confidence=0.2,
#             frames_used=used_frames,
#             notes="Not enough wrist detections. Improve lighting and keep striking arm visible."
#         )

#     # interpolate single-frame gaps
#     pts = wrist_pts[:]
#     for i in range(1, len(pts)-1):
#         if pts[i] is None and pts[i-1] is not None and pts[i+1] is not None:
#             x = (pts[i-1][0] + pts[i+1][0]) / 2.0
#             y = (pts[i-1][1] + pts[i+1][1]) / 2.0
#             pts[i] = (x,y)
#     pts = [p for p in pts if p is not None]
#     if len(pts) < 6:
#         return SmashResult(
#             mode_used="no_shuttle",
#             max_kmh=0.0, avg_kmh=0.0, confidence=0.25,
#             frames_used=used_frames,
#             notes="Insufficient continuous motion. Keep racket in view."
#         )

#     v = velocity_series(pts, px_per_meter, dt)
#     if len(v) == 0:
#         return SmashResult(
#             mode_used="no_shuttle",
#             max_kmh=0.0, avg_kmh=0.0, confidence=0.25,
#             frames_used=used_frames,
#             notes="Could not compute velocity."
#         )

#     vs = moving_average(v, SMOOTH_WIN)
#     peak_idx = int(np.argmax(vs))
#     lo = max(0, peak_idx - IMPACT_NEIGHBOR)
#     hi = min(len(vs), peak_idx + IMPACT_NEIGHBOR + 1)
#     window = vs[lo:hi] if hi > lo else vs

#     peak_ms = float(np.max(window))
#     avg_ms  = float(np.mean(window))

#     ALPHA = float(alpha) if alpha else DEFAULT_ALPHA
#     est_ms_peak = ALPHA * peak_ms
#     est_ms_avg  = 0.65 * est_ms_peak

#     conf = float(confidence_from_signal(vs))

#     return SmashResult(
#         mode_used="no_shuttle",
#         max_kmh=round(est_ms_peak * 3.6, 1),
#         avg_kmh=round(est_ms_avg * 3.6, 1),
#         confidence=round(conf, 2),
#         frames_used=used_frames,
#         notes=f"Estimated from wrist motion. α={ALPHA}. Use good calibration."
#     )
import io, math
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import av
import cv2
import mediapipe as mp

app = FastAPI(title="ShuttleX Smash Speed API")

# ------------------- Tunables (fast + robust) -------------------
# Processing speed
FRAME_STRIDE = 3           # process every Nth frame
MAX_FRAMES = 2400          # cap CPU work
DOWNSCALE_LONG_SIDE = 640  # max dimension after downscale
MEDIAPIPE_COMPLEXITY = 0   # 0 fastest (enough for wrist)

# Speed estimation
SMOOTH_WIN = 5             # moving average over velocity
IMPACT_NEIGHBOR = 6        # +/- frames around peak window
DEFAULT_ALPHA = 1.6        # wrist->shuttle multiplier (tune per product)

# Auto-scale (pixels-per-meter)
DEFAULT_PX_PER_METER = 280.0      # typical for side view @ ~5–8m distance
PX_MIN, PX_MAX = 80.0, 1200.0     # sanity clamp

# ------------------- Models -------------------
class SmashResult(BaseModel):
    mode_used: str                 # "no_shuttle" for now
    max_kmh: float
    avg_kmh: float
    confidence: float
    frames_used: int
    notes: Optional[str] = None

# ------------------- Helpers -------------------
def moving_average(x, w):
    if len(x) < w:
        return np.array(x, dtype=np.float32)
    csum = np.cumsum(np.insert(x, 0, 0))
    out = (csum[w:] - csum[:-w]) / float(w)
    pad = [out[0]] * (w-1)
    return np.array(pad + out.tolist(), dtype=np.float32)

def velocity_series(points_px, px_per_meter, dt):
    v = []
    for i in range(1, len(points_px)):
        x1, y1 = points_px[i-1]
        x2, y2 = points_px[i]
        dpx = math.hypot(x2 - x1, y2 - y1)
        dm  = dpx / max(px_per_meter, 1e-6)
        v.append(dm / dt)  # m/s
    return np.array(v, dtype=np.float32)

def confidence_from_signal(v):
    if len(v) < 10: return 0.3
    p95 = float(np.percentile(v, 95))
    p50 = float(np.percentile(v, 50))
    if p50 <= 1e-6: return 0.3
    ratio = min(max(p95 / p50, 1.0), 4.0)  # 1..4 -> 0.25..1.0 scaled
    return 0.25 + 0.75 * ((ratio - 1.0) / 3.0)

def clamp_px_per_meter(px: float) -> float:
    try:
        if px != px:  # NaN
            return DEFAULT_PX_PER_METER
    except Exception:
        return DEFAULT_PX_PER_METER
    return max(PX_MIN, min(PX_MAX, float(px)))

def estimate_px_per_meter_from_first_frame(bgr) -> float:
    """
    Heuristic for side view: use vertical gap floor↔net-tape ≈ 1.524 m.
    Works if net & floor visible and exposure is reasonable.
    Falls back to DEFAULT_PX_PER_METER if unreliable/too dark.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Dark frame => bail out
    if float(gray.mean()) < 8.0:
        return DEFAULT_PX_PER_METER

    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=int(w*0.4), maxLineGap=10)
    ys = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if abs(y2-y1) <= 3:  # near-horizontal
                ys.append(int((y1+y2)//2))
    if len(ys) < 2:
        return DEFAULT_PX_PER_METER

    ys.sort()
    y_floor = max(ys)
    mid = [y for y in ys if h*0.25 < y < h*0.75]
    y_net  = min(mid) if mid else min(ys)

    gap = float(abs(y_floor - y_net))
    if gap < 20:
        return DEFAULT_PX_PER_METER
    return clamp_px_per_meter(gap / 1.524)

# ------------------- API -------------------
@app.get("/")
def health():
    return {"ok": True}

@app.post("/smash/analyze", response_model=SmashResult)
async def analyze(
    video: UploadFile = File(...),
    px_per_meter: Optional[float] = Form(None),   # optional now
    fps: Optional[float] = Form(None),
    alpha: Optional[float] = Form(None)
):
    """
    Estimate smash shuttle speed from a 30s video.
    If no px_per_meter provided (or it's bad), auto-estimate from first frame.
    Uses wrist motion as proxy; maps via alpha to shuttle exit speed.
    """
    # ---- Read & video info
    raw = await video.read()
    container = av.open(io.BytesIO(raw))
    vstream = container.streams.video[0]
    vid_fps = float(fps) if fps else (float(vstream.average_rate) if vstream.average_rate else 30.0)
    dt = 1.0 / vid_fps

    # ---- Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=MEDIAPIPE_COMPLEXITY, enable_segmentation=False)

    wrist_pts = []
    used_frames = 0

    # ---- px/m: user-provided or auto from first frame
    PX = clamp_px_per_meter(px_per_meter) if px_per_meter is not None else None

    processed = 0
    for idx, frame in enumerate(container.decode(vstream)):
        # frame stride
        if (idx % FRAME_STRIDE) != 0:
            continue
        if processed >= MAX_FRAMES:
            break
        processed += 1

        bgr = frame.to_ndarray(format="bgr24")
        h, w = bgr.shape[:2]

        # Downscale for speed
        scale = DOWNSCALE_LONG_SIDE / max(h, w) if max(h, w) > DOWNSCALE_LONG_SIDE else 1.0
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]

        # auto px/m on first processed frame
        if PX is None:
            PX = estimate_px_per_meter_from_first_frame(bgr)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            rw = lm[16]  # right wrist
            if 0 <= rw.x <= 1 and 0 <= rw.y <= 1:
                wrist_pts.append((rw.x * w, rw.y * h))
                used_frames += 1
            else:
                wrist_pts.append(None)
        else:
            wrist_pts.append(None)

    pose.close()

    if PX is None:
        PX = DEFAULT_PX_PER_METER

    valid = [(i,p) for i,p in enumerate(wrist_pts) if p is not None]
    if len(valid) < 6:
        return SmashResult(
            mode_used="no_shuttle",
            max_kmh=0.0, avg_kmh=0.0, confidence=0.2,
            frames_used=used_frames,
            notes="Not enough wrist detections. Improve lighting and keep striking arm visible."
        )

    # interpolate single-frame gaps
    pts = wrist_pts[:]
    for i in range(1, len(pts)-1):
        if pts[i] is None and pts[i-1] is not None and pts[i+1] is not None:
            x = (pts[i-1][0] + pts[i+1][0]) / 2.0
            y = (pts[i-1][1] + pts[i+1][1]) / 2.0
            pts[i] = (x,y)
    pts = [p for p in pts if p is not None]
    if len(pts) < 6:
        return SmashResult(
            mode_used="no_shuttle",
            max_kmh=0.0, avg_kmh=0.0, confidence=0.25,
            frames_used=used_frames,
            notes="Insufficient continuous motion. Keep racket in view."
        )

    # ---- Speeds
    v = velocity_series(pts, PX, dt)
    if len(v) == 0:
        return SmashResult(
            mode_used="no_shuttle",
            max_kmh=0.0, avg_kmh=0.0, confidence=0.25,
            frames_used=used_frames,
            notes="Could not compute velocity."
        )

    vs = moving_average(v, SMOOTH_WIN)
    peak_idx = int(np.argmax(vs))
    lo = max(0, peak_idx - IMPACT_NEIGHBOR)
    hi = min(len(vs), peak_idx + IMPACT_NEIGHBOR + 1)
    window = vs[lo:hi] if hi > lo else vs

    peak_ms = float(np.max(window))
    avg_ms  = float(np.mean(window))

    ALPHA = float(alpha) if alpha else DEFAULT_ALPHA
    est_ms_peak = ALPHA * peak_ms
    est_ms_avg  = 0.65 * est_ms_peak

    conf = float(confidence_from_signal(vs))

    return SmashResult(
        mode_used="no_shuttle",
        max_kmh=round(est_ms_peak * 3.6, 1),
        avg_kmh=round(est_ms_avg * 3.6, 1),
        confidence=round(conf, 2),
        frames_used=used_frames,
        notes=f"Auto px/m={round(PX,1)}, α={ALPHA}, fps={round(1.0/dt,1)}, stride={FRAME_STRIDE}."
    )
