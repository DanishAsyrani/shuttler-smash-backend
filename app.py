import io, math
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import av
import cv2
import mediapipe as mp

app = FastAPI(title="ShuttleX Smash Speed API")

# Tunables
DEFAULT_ALPHA = 1.25   # maps wrist/racket peak speed -> estimated shuttle speed
SMOOTH_WIN = 5         # moving average window for velocity smoothing
IMPACT_NEIGHBOR = 6    # frames around peak when averaging

class SmashResult(BaseModel):
    mode_used: str                 # "no_shuttle" for now
    max_kmh: float
    avg_kmh: float
    confidence: float
    frames_used: int
    notes: Optional[str] = None

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

@app.post("/smash/analyze", response_model=SmashResult)
async def analyze(
    video: UploadFile = File(...),
    px_per_meter: float = Form(...),
    fps: Optional[float] = Form(None),
    alpha: Optional[float] = Form(None)
):
    """
    Estimate smash shuttle speed from a 30s video.
    If shuttle not visible, uses wrist motion as proxy and maps with alpha.
    """
    raw = await video.read()
    container = av.open(io.BytesIO(raw))
    vstream = container.streams.video[0]

    vid_fps = float(fps) if fps else (float(vstream.average_rate) if vstream.average_rate else 30.0)
    dt = 1.0 / vid_fps

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    wrist_pts = []
    used_frames = 0

    for frame in container.decode(vstream):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        # downscale for speed
        scale = 720 / max(h, w) if max(h, w) > 720 else 1.0
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

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

    v = velocity_series(pts, px_per_meter, dt)
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
        notes=f"Estimated from wrist motion. α={ALPHA}. Use good calibration."
    )
