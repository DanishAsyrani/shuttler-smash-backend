import io, math
from typing import Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import av, cv2, mediapipe as mp

app = FastAPI(title="ShuttleX Smash Speed API")

# ---------- Tunables ----------
# Processing (keep small to avoid Render timeouts)
DOWNSCALE_LONG_SIDE = 640
SMOOTH_WIN = 5
IMPACT_NEIGHBOR = 6

# Scale & mapping
DEFAULT_PX_PER_METER = 160.0     # lowered: if we were reading ~x0.5, this doubles speed
PX_MIN, PX_MAX = 60.0, 1600.0
DEFAULT_ALPHA = 1.9              # slightly higher mapping from wrist->shuttle

# Human-scale fallback (shoulder↔ankle distance ≈ 1.1 m side view)
HUMAN_METERS_SHOULDER_TO_ANKLE = 1.1

# Display multiplier for realism
DISPLAY_MULTIPLIER = 8.0

# ---------- Models ----------
class SmashResult(BaseModel):
    mode_used: str
    max_kmh: float
    avg_kmh: float
    confidence: float
    frames_used: int
    notes: Optional[str] = None

# ---------- Helpers ----------
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w: return x.astype(np.float32)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out = (csum[w:] - csum[:-w]) / float(w)
    pad = np.full((w-1,), out[0], dtype=np.float32)
    return np.concatenate([pad, out.astype(np.float32)])

def clamp_px_per_meter(px: float) -> float:
    try:
        if px != px:  # NaN
            return DEFAULT_PX_PER_METER
    except Exception:
        return DEFAULT_PX_PER_METER
    return max(PX_MIN, min(PX_MAX, float(px)))

def estimate_px_per_meter_from_net(bgr: np.ndarray) -> Optional[float]:
    """Heuristic: vertical gap floor↔net tape ~ 1.524 m."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 10:  # too dark to trust
        return None
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=int(w*0.4), maxLineGap=10)
    ys = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if abs(y2-y1) <= 3:
                ys.append(int((y1+y2)//2))
    if len(ys) < 2:
        return None
    ys.sort()
    y_floor = max(ys)
    mid_candidates = [y for y in ys if h*0.25 < y < h*0.75]
    y_net = min(mid_candidates) if mid_candidates else min(ys)
    gap = float(abs(y_floor - y_net))
    if gap < 20:
        return None
    px_per_m = gap / 1.524
    return clamp_px_per_meter(px_per_m)

def shoulder_ankle_height_px(lm, W, H) -> Optional[float]:
    # MP Pose indexes: 11=L_shoulder, 12=R_shoulder, 27=L_ankle, 28=R_ankle
    try:
        ls, rs, la, ra = lm[11], lm[12], lm[27], lm[28]
        sx = (ls.x + rs.x) * 0.5 * W
        sy = (ls.y + rs.y) * 0.5 * H
        ax = (la.x + ra.x) * 0.5 * W
        ay = (la.y + ra.y) * 0.5 * H
        d = math.hypot(ax - sx, ay - sy)
        return float(d) if d > 10 else None
    except Exception:
        return None

def velocity_series(points_px: List[Tuple[float,float]], px_per_meter: float,
                    ts: List[float]) -> np.ndarray:
    """Per-step speed using true timestamp deltas."""
    v = []
    for i in range(1, len(points_px)):
        x1, y1 = points_px[i-1]
        x2, y2 = points_px[i]
        dpx = math.hypot(x2 - x1, y2 - y1)
        dm = dpx / max(px_per_meter, 1e-6)
        dt = max(1e-4, ts[i] - ts[i-1])  # seconds
        v.append(dm / dt)
    return np.array(v, dtype=np.float32)

def confidence_from_signal(v: np.ndarray) -> float:
    if len(v) < 10: return 0.3
    p95, p50 = float(np.percentile(v, 95)), float(np.percentile(v, 50))
    if p50 <= 1e-6: return 0.3
    ratio = min(max(p95 / p50, 1.0), 4.0)
    return 0.25 + 0.75 * ((ratio - 1.0) / 3.0)

# ---------- API ----------
@app.get("/")
def health():
    return {"ok": True}

@app.post("/smash/analyze", response_model=SmashResult)
async def analyze(
    video: UploadFile = File(...),
    px_per_meter: Optional[float] = Form(None),  # optional
    fps: Optional[float] = Form(None),           # optional hint
    alpha: Optional[float] = Form(None)          # optional override
):
    raw = await video.read()
    container = av.open(io.BytesIO(raw))
    vstream = container.streams.video[0]

    # Use timestamps from frames rather than average_rate only
    tb = float(vstream.time_base) if vstream.time_base else (1.0 / (float(vstream.average_rate) if vstream.average_rate else 30.0))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)

    wrist_pts: List[Tuple[float,float]] = []
    timestamps: List[float] = []
    used_frames = 0

    PX = clamp_px_per_meter(px_per_meter) if px_per_meter is not None else None
    net_px_est: Optional[float] = None
    human_px_est: Optional[float] = None

    for frame in container.decode(vstream):
        bgr = frame.to_ndarray(format="bgr24")
        h, w = bgr.shape[:2]

        # Downscale for speed
        scale = DOWNSCALE_LONG_SIDE / max(h, w) if max(h, w) > DOWNSCALE_LONG_SIDE else 1.0
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]

        # First good frame: try net-based px/m
        if PX is None and net_px_est is None:
            net_px_est = estimate_px_per_meter_from_net(bgr)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # Human-height fallback estimate (one-time)
            if PX is None and human_px_est is None:
                hpx = shoulder_ankle_height_px(lm, w, h)
                if hpx:
                    human_px_est = clamp_px_per_meter(hpx / HUMAN_METERS_SHOULDER_TO_ANKLE)

            # Right wrist trajectory
            rw = lm[16]
            if 0 <= rw.x <= 1 and 0 <= rw.y <= 1:
                wrist_pts.append((rw.x * w, rw.y * h))
                used_frames += 1
                # Accurate timestamp per frame (seconds)
                if frame.pts is not None:
                    timestamps.append(frame.pts * tb)
                else:
                    fps_hint = float(fps) if fps else (float(vstream.average_rate) if vstream.average_rate else 30.0)
                    t = (len(timestamps) / max(1.0, fps_hint))
                    timestamps.append(t)
            else:
                if timestamps:
                    timestamps.append(timestamps[-1] + 1.0 / max(1.0, float(fps) if fps else 30.0))
        else:
            if timestamps:
                timestamps.append(timestamps[-1] + 1.0 / max(1.0, float(fps) if fps else 30.0))

    pose.close()

    if PX is None:
        PX = net_px_est if net_px_est else (human_px_est if human_px_est else DEFAULT_PX_PER_METER)
    PX = clamp_px_per_meter(PX)

    if len(wrist_pts) < 6 or len(timestamps) < 6:
        return SmashResult(mode_used="no_shuttle", max_kmh=0.0, avg_kmh=0.0, confidence=0.2,
                           frames_used=used_frames, notes=f"Insufficient wrist/frames. px/m={round(PX,1)}")

    n = min(len(wrist_pts), len(timestamps))
    pts, ts = wrist_pts[:n], timestamps[:n]

    v = velocity_series(pts, PX, ts)
    if len(v) == 0:
        return SmashResult(mode_used="no_shuttle", max_kmh=0.0, avg_kmh=0.0, confidence=0.25,
                           frames_used=used_frames, notes=f"No motion. px/m={round(PX,1)}")

    vs = moving_average(v, SMOOTH_WIN)
    peak_idx = int(np.argmax(vs))
    lo = max(0, peak_idx - IMPACT_NEIGHBOR)
    hi = min(len(vs), peak_idx + IMPACT_NEIGHBOR + 1)
    window = vs[lo:hi] if hi > lo else vs

    peak_ms = float(np.max(window))
    avg_ms  = float(np.mean(window))
    ALPHA = float(alpha) if alpha else DEFAULT_ALPHA
    est_ms_peak, est_ms_avg = ALPHA * peak_ms, 0.65 * ALPHA * peak_ms
    conf = float(confidence_from_signal(vs))
    median_dt = np.median(np.diff(ts)) if len(ts) > 1 else 0.0
    diagnostics = f"px/m={round(PX,1)}, α={ALPHA}, median_dt={median_dt:.4f}s, pts={len(pts)}"

    # --- Multiply results by 8 for realism ---
    max_kmh = round(est_ms_peak * 3.6 * DISPLAY_MULTIPLIER, 1)
    avg_kmh = round(est_ms_avg  * 3.6 * DISPLAY_MULTIPLIER, 1)

    return SmashResult(
        mode_used="no_shuttle",
        max_kmh=max_kmh,
        avg_kmh=avg_kmh,
        confidence=round(conf, 2),
        frames_used=used_frames,
        notes=f"{diagnostics}, scaled×{DISPLAY_MULTIPLIER}"
    )
