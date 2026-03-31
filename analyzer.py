"""
analyzer.py  —  SurgiScore v3.1 (YOLOv8 + camera jump filter)
==============================================================
Fixed: camera angle change jumps are now filtered out from
probe path and jitter calculations so precision
scores are accurate.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import time
import textwrap
import requests
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

YOLO_WEIGHTS   = "weights/best.pt"
CONF_THRESHOLD = 0.35

# ── KEY FILTER: max pixels an instrument can realistically
#    move between two consecutive frames. Any jump larger
#    than this is treated as a camera cut and discarded. ──
MAX_JUMP_PX    = 120

CLASS_ARTHROSCOPE = 0
CLASS_PROBE       = 1

EXPERT_BASELINES = {
    "beginner":     {"stability": 60, "efficiency": 55, "precision": 58, "overall": 58},
    "intermediate": {"stability": 75, "efficiency": 70, "precision": 72, "overall": 72},
    "expert":       {"stability": 88, "efficiency": 82, "precision": 85, "overall": 85},
}


# ── YOLO model loader ─────────────────────────────────────────────────────────

_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        if os.path.exists(YOLO_WEIGHTS):
            try:
                from ultralytics import YOLO
                _yolo_model = YOLO(YOLO_WEIGHTS)
                print(f"[YOLOv8] Loaded weights: {YOLO_WEIGHTS}")
            except Exception as e:
                print(f"[YOLOv8] Load failed: {e}. Falling back to background subtraction.")
                _yolo_model = "fallback"
        else:
            print(f"[YOLOv8] No weights at {YOLO_WEIGHTS}. Using background subtraction.")
            _yolo_model = "fallback"
    return _yolo_model


# ── per-frame detection ───────────────────────────────────────────────────────

def detect_instruments(frame, back_sub=None):
    """Returns (arthroscope_center, probe_center) — each is (cx,cy) or None."""
    model = get_model()

    if model != "fallback":
        results        = model(frame, conf=CONF_THRESHOLD, verbose=False)
        arthroscope_pt = None
        probe_pt       = None
        for r in results:
            for box in r.boxes:
                cls_id       = int(box.cls[0])
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx, cy       = (x1+x2)//2, (y1+y2)//2
                if cls_id == CLASS_ARTHROSCOPE and arthroscope_pt is None:
                    arthroscope_pt = (cx, cy)
                elif cls_id == CLASS_PROBE and probe_pt is None:
                    probe_pt = (cx, cy)
        return arthroscope_pt, probe_pt

    else:
        if back_sub is None:
            return None, None
        mask   = back_sub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt, best_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200 and area > best_area:
                best_area, best_cnt = area, cnt
        if best_cnt is not None:
            x,y,w,h = cv2.boundingRect(best_cnt)
            return None, (x+w//2, y+h//2)
        return None, None


# ── Claude API ────────────────────────────────────────────────────────────────

def get_ai_feedback(stability, efficiency, precision, overall, stats, level):
    level_context = {
        "beginner":     "This is a beginner trainee — be encouraging and focus on foundational habits.",
        "intermediate": "This is an intermediate trainee — be direct and push toward expert standards.",
        "expert":       "This is an advanced trainee — be clinically rigorous and highly specific.",
    }.get(level, "")

    prompt = f"""You are a world-class arthroscopic surgery trainer.

{level_context}
Tracking: YOLOv8 object detection — arthroscope and probe tracked separately.
Camera jump filtering applied (jumps >{MAX_JUMP_PX}px discarded).

Performance scores:
- Stability:  {stability:.1f}/100  (arthroscope steadiness — jitter between frames)
- Efficiency: {efficiency:.1f}/100  (probe movement — controlled vs erratic)
- Precision:  {precision:.1f}/100   (probe path — smooth vs tremorous)
- Overall:    {overall:.1f}/100

Raw data:
- Arthroscope jitter : {stats['arthroscope_jitter']:.1f} px/frame
- Probe path ratio   : {stats['path_ratio']:.2f}x  (1.0 = perfectly smooth/intentional)
- Avg probe speed    : {stats['speed_mean']:.1f} px/s
- Valid probe frames : {stats['valid_probe_frames']}
- Filtered jumps     : {stats['filtered_jumps']}
- Total frames       : {stats['frames']}

Write exactly 3 sentences of professional, clinical feedback:
1. Acknowledge the strongest metric with a clinical reason why it matters.
2. Identify the weakest metric and the surgical risk if not improved.
3. Give ONE specific, immediately actionable drill or technique to improve it.

Tone: Respected senior surgeon mentor — direct, warm, evidence-based. No fluff."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 280,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=25
        )
        data = response.json()
        return data["content"][0]["text"]
    except Exception as e:
        print(f"AI feedback error: {e}")
        scores = [("stability", stability), ("efficiency", efficiency),
                  ("precision", precision)]
        best  = max(scores, key=lambda x: x[1])
        worst = min(scores, key=lambda x: x[1])
        tips  = {
            "stability":  "Anchor your elbow on a fixed surface to reduce hand tremor.",
            "efficiency": "Visualize the target before inserting — plan the path, then move decisively.",
            "precision":  "Focus on fluid, continuous movements rather than stuttering, hesitant corrections."
        }
        return (f"Your {best[0]} of {best[1]:.0f}/100 is your standout strength. "
                f"Your {worst[0]} of {worst[1]:.0f}/100 needs priority attention — "
                f"inconsistency here increases procedure time and tissue trauma risk. "
                f"{tips[worst[0]]}")


def score_color(s):
    if s >= 80: return "#00ffb3"
    if s >= 65: return "#00d4ff"
    if s >= 50: return "#ffc820"
    return "#ff3d5a"


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_video(video_path, results_dir="results", level="intermediate"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    # separate tracking lists
    arthroscope_positions = []
    probe_positions       = []
    probe_speeds          = []
    arthroscope_jitters   = []

    frame_idx      = 0
    filtered_jumps = 0   # count how many camera-cut jumps we discarded
    prev_scope     = None
    prev_probe     = None

    print(f"[Analyzer] Model: {'YOLOv8' if get_model() != 'fallback' else 'background subtraction'}")
    print(f"[Analyzer] Jump filter: >{MAX_JUMP_PX}px discarded")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # crop to left half for dual-view surgical video
        USE_CROP = True
        roi = frame[:, :frame_w//2].copy() if USE_CROP else frame.copy()

        scope_pt, probe_pt = detect_instruments(roi, back_sub)

        # ── arthroscope tracking → stability ──────────────────────────────────
        if scope_pt is not None:
            if prev_scope is not None:
                jitter = np.hypot(scope_pt[0] - prev_scope[0],
                                  scope_pt[1] - prev_scope[1])
                if jitter < MAX_JUMP_PX:
                    # real movement — record it
                    arthroscope_jitters.append(float(jitter))
                else:
                    # camera cut — reset scope tracking
                    filtered_jumps += 1
                    prev_scope = None
                    continue
            arthroscope_positions.append(scope_pt)
            prev_scope = scope_pt

        # ── probe tracking → precision, efficiency ────────────────────────────
        if probe_pt is not None:
            if prev_probe is not None:
                dist = np.hypot(probe_pt[0] - prev_probe[0],
                                probe_pt[1] - prev_probe[1])
                spd  = dist / dt

                if dist < MAX_JUMP_PX and spd < 3000:
                    # real movement — record everything
                    probe_speeds.append(float(spd))
                    probe_positions.append(probe_pt)
                else:
                    # camera cut or teleport — reset probe tracking
                    filtered_jumps += 1
                    prev_probe = None
                    continue
            else:
                # first detection or after reset — just record position
                probe_positions.append(probe_pt)

            prev_probe = probe_pt

    cap.release()

    print(f"[Analyzer] Frames: {frame_idx} | "
          f"Scope detections: {len(arthroscope_positions)} | "
          f"Probe detections: {len(probe_positions)} | "
          f"Filtered jumps: {filtered_jumps}")

    # ── Invalid video detection ─────────────────────────────────────────────
    total_detections = len(arthroscope_positions) + len(probe_positions)
    detection_rate = total_detections / max(frame_idx, 1)

    if frame_idx < 10:
        raise ValueError(
            "Invalid video: Too few frames to analyze. "
            "Please upload a valid surgical video."
        )

    if detection_rate < 0.05:
        raise ValueError(
            "Invalid video: No surgical instruments detected in the footage. "
            f"Only {total_detections} detections in {frame_idx} frames "
            f"({detection_rate*100:.1f}% detection rate). "
            "Please upload a valid arthroscopic surgery video."
        )

    # Check movement coherence — if all positions cluster in a tiny area, not real surgery
    all_positions = probe_positions + arthroscope_positions
    if len(all_positions) >= 10:
        all_xs = [p[0] for p in all_positions]
        all_ys = [p[1] for p in all_positions]
        if np.std(all_xs) < 5 and np.std(all_ys) < 5:
            raise ValueError(
                "Invalid video: Detected objects show no meaningful movement. "
                "This does not appear to be a surgical procedure video. "
                "Please upload a valid arthroscopic surgery video."
            )

    # fallback if probe not detected — use arthroscope positions
    if len(probe_positions) < 10:
        if len(arthroscope_positions) >= 10:
            probe_positions = arthroscope_positions
            print("[Analyzer] Using arthroscope positions for path metrics (no probe detected)")
        else:
            raise ValueError(
                "Invalid video: Not enough instrument detections. "
                "Please upload a valid arthroscopic surgery video."
            )

    # fallback jitter — compute from probe if no scope detections
    if len(arthroscope_jitters) < 10:
        arthroscope_jitters = []
        for i in range(1, len(probe_positions)):
            d = np.hypot(probe_positions[i][0] - probe_positions[i-1][0],
                         probe_positions[i][1] - probe_positions[i-1][1])
            if d < MAX_JUMP_PX:
                arthroscope_jitters.append(float(d))

    # fallback path & speed — compute from scope if probe detection is too sparse (< 5% of frames)
    if len(probe_positions) < max(10, frame_idx * 0.05) and len(arthroscope_positions) >= 10:
        print("[Analyzer] Using arthroscope positions for path metrics (sparse probe)")
        probe_positions = list(arthroscope_positions)
        probe_speeds = [j * 30 for j in arthroscope_jitters] # approx speed assuming 30fps

    # ── compute all metrics ───────────────────────────────────────────────────
    jitter_mean = float(np.mean(arthroscope_jitters)) if arthroscope_jitters else 30.0
    speed_mean  = float(np.mean(probe_speeds))        if probe_speeds        else 0.0
    speed_std   = float(np.std(probe_speeds))         if probe_speeds        else 0.0
    # Use max(..., 200) so that intermittent movements/pauses do not massively inflate the CV
    speed_cv    = speed_std / max(speed_mean, 200.0)

    xs = [p[0] for p in probe_positions] if probe_positions else [0]
    ys = [p[1] for p in probe_positions] if probe_positions else [0]

    total_path = 0.0
    for i in range(1, len(probe_positions)):
        d = np.hypot(probe_positions[i][0] - probe_positions[i-1][0],
                     probe_positions[i][1] - probe_positions[i-1][1])
        if d < MAX_JUMP_PX:  # Only sum valid continuous segments to avoid teleport jumps
            total_path += float(d)

    # Use moving average to calculate an idealized "smooth" intended path (15-frame window)
    if len(xs) > 15:
        w = 15
        smooth_xs = np.convolve(xs, np.ones(w)/w, mode='valid')
        smooth_ys = np.convolve(ys, np.ones(w)/w, mode='valid')
        smooth_path = 0.0
        for i in range(1, len(smooth_xs)):
            smooth_path += float(np.hypot(smooth_xs[i] - smooth_xs[i-1], smooth_ys[i] - smooth_ys[i-1]))
        
        # Guard against near-stationary trembling where smooth_path approaches 0 
        bbox_diag_current = float(np.hypot(max(xs)-min(xs), max(ys)-min(ys)) + 1e-6)
        # Compare actual raw path to this idealized smoothed path
        path_ratio = float(min(total_path / max(smooth_path, bbox_diag_current * 0.5, 10.0), 10.0))
    else:
        # Fallback for very short segments
        bbox_diag  = float(np.hypot(max(xs)-min(xs), max(ys)-min(ys)) + 1e-6) if xs else 100.0
        path_ratio = float(min(total_path / bbox_diag, 5.0))

    # ── three scores ──────────────────────────────────────────────────────────
    stability_score  = float(max(0, min(100, 100 - ((jitter_mean - 10) / 70) * 100)))

    small_moves      = sum(1 for d in arthroscope_jitters if d < jitter_mean)
    efficiency_score = float(max(0, min(100,
        30 + (small_moves / max(len(arthroscope_jitters), 1)) * 70
    )))

    # precision: relaxed penalty curve so microscopic flutter doesn't plummet score to 0.
    precision_score  = float(max(0, min(100, 100 - ((path_ratio - 1.0) / 5.0) * 100)))

    overall_score = (stability_score  * 0.40 +
                     efficiency_score * 0.30 +
                     precision_score  * 0.30)

    grade = ("A+" if overall_score >= 90 else "A"  if overall_score >= 80 else
             "B+" if overall_score >= 70 else "B"  if overall_score >= 60 else
             "C"  if overall_score >= 50 else "D"  if overall_score >= 40 else "F")

    stats = {
        "arthroscope_jitter":  jitter_mean,
        "path_ratio":          path_ratio,
        "speed_mean":          speed_mean,
        "frames":              frame_idx,
        "valid_probe_frames":  len(probe_positions),
        "filtered_jumps":      filtered_jumps,
        "scope_detections":    len(arthroscope_positions),
        "probe_detections":    len(probe_positions),
    }

    ai_comment = get_ai_feedback(
        stability_score, efficiency_score, precision_score,
        overall_score, stats, level
    )

    job_id      = os.path.basename(video_path).split("_")[0]
    report_path = os.path.join(results_dir, f"{job_id}_report.png")

    _generate_report(
        xs, ys, probe_speeds, arthroscope_jitters,
        stability_score, efficiency_score, precision_score,
        overall_score, jitter_mean, speed_mean, path_ratio,
        total_path, frame_idx, grade, ai_comment, level,
        len(arthroscope_positions), len(probe_positions),
        filtered_jumps, report_path
    )

    return {
        "stability":          round(stability_score,  1),
        "efficiency":         round(efficiency_score, 1),
        "precision":          round(precision_score,  1),
        "overall":            round(overall_score,    1),
        "grade":              grade,
        "frames":             frame_idx,
        "jitter_mean":        round(jitter_mean,  1),
        "speed_mean":         round(speed_mean,   1),
        "path_ratio":         round(path_ratio,   2),
        "total_path":         round(total_path,   0),
        "filtered_jumps":     filtered_jumps,
        "scope_detections":   len(arthroscope_positions),
        "probe_detections":   len(probe_positions),
        "tracking_quality":   0,
        "tracking_method":    "YOLOv8" if get_model() != "fallback" else "Motion detection",
        "ai_comment":         ai_comment,
        "level":              level,
        "report_image":       f"{job_id}_report.png",
        "expert":             EXPERT_BASELINES[level],
    }


# ── report ────────────────────────────────────────────────────────────────────

def _generate_report(xs, ys, speeds, jitters,
                     stability, efficiency, precision,
                     overall, jitter_mean, speed_mean, path_ratio,
                     total_path, frame_idx, grade, ai_comment, level,
                     scope_count, probe_count, filtered_jumps, save_path):

    BG     = "#060b12"; PANEL  = "#0b1520"; PANEL2 = "#0f1d30"
    BORDER = "#193352"; GREEN  = "#00ffb3"; CYAN   = "#00d4ff"
    AMBER  = "#ffc820"; RED    = "#ff3d5a"; PURPLE = "#b388ff"
    WHITE  = "#ddeeff"; GREY   = "#3d5975"

    plt.rcParams.update({
        "font.family": "monospace", "text.color": WHITE,
        "axes.labelcolor": WHITE, "xtick.color": GREY,
        "ytick.color": GREY, "axes.edgecolor": BORDER,
        "figure.facecolor": BG, "axes.facecolor": PANEL2,
    })

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.text(0.5, 0.972,
             "SURGISCORE  ·  AI SURGICAL PERFORMANCE REPORT  ·  YOLOv8 EDITION",
             ha="center", fontsize=15, fontweight="bold", color=CYAN)
    fig.text(0.5, 0.952,
             f"Level: {level.title()}   ·   Frames: {frame_idx}   ·   "
             f"Scope: {scope_count}   ·   Probe: {probe_count}   ·   "
             f"Filtered jumps: {filtered_jumps}   ·   "
             f"Jitter: {jitter_mean:.1f}px   ·   Path ratio: {path_ratio:.2f}x",
             ha="center", fontsize=8.5, color=GREY)
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.942, 0.942],
                   transform=fig.transFigure, color=BORDER, lw=1.2))

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           top=0.93, bottom=0.23, hspace=0.50, wspace=0.38)

    # 1. Score bars
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL2)
    labels  = ["Stability", "Efficiency", "Precision", "Overall"]
    scores  = [stability, efficiency, precision, overall]
    weights = ["40%", "30%", "30%", ""]
    for i, (lbl, sc, wt) in enumerate(zip(labels, scores, weights)):
        col = score_color(sc)
        ax1.barh(i, 100, color="#0d2035", height=0.6, zorder=1)
        ax1.barh(i, sc,  color=col, height=0.6, alpha=0.85, zorder=2)
        ax1.text(sc + 1.5, i, f"{sc:.1f}", va="center", color=col,
                 fontsize=9, fontweight="bold")
        if wt:
            ax1.text(102, i - 0.35, wt, va="bottom", color=GREY, fontsize=6)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlim(0, 118)
    ax1.set_xlabel("Score / 100", fontsize=8)
    ax1.set_title("SCORE BREAKDOWN", fontsize=9, color=CYAN, pad=8, fontweight="bold")
    ax1.spines[:].set_color(BORDER)
    ax1.tick_params(colors=GREY, labelsize=7)
    ax1.axvline(x=50, color=BORDER, lw=0.8, linestyle=":")
    ax1.axvline(x=80, color=BORDER, lw=0.8, linestyle=":")

    # 2. Probe path
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL2)
    if len(xs) > 1:
        for i in range(1, len(xs)):
            t = i / len(xs)
            ax2.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                     color=(int(255*t)/255, int(100+100*(1-t))/255, int(255*(1-t))/255),
                     lw=0.9, alpha=0.6)
        step = max(1, len(xs)//300)
        ax2.scatter(xs[::step], ys[::step],
                    c=range(0, len(xs), step)[:len(xs[::step])],
                    cmap="plasma", s=4, alpha=0.7, zorder=3)
        ax2.plot(xs[0],  ys[0],  "o", color=GREEN, ms=9,  zorder=6, label="Start")
        ax2.plot(xs[-1], ys[-1], "*", color=RED,   ms=13, zorder=6, label="End")
    ax2.set_title("PROBE PATH  (jumps filtered)", fontsize=9,
                  color=CYAN, pad=8, fontweight="bold")
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=7)
    ax2.spines[:].set_color(BORDER)
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7,
               loc="upper right", framealpha=0.7)

    # 3. Speed profile
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL2)
    if speeds:
        ax3.fill_between(range(len(speeds)), speeds, alpha=0.12, color=AMBER)
        ax3.plot(speeds, color=AMBER, lw=0.9, alpha=0.85)
        ax3.axhline(speed_mean, color=GREEN, lw=1.8, linestyle="--",
                    label=f"Mean {speed_mean:.0f}")
        w = 30
        if len(speeds) > w:
            roll = np.convolve(speeds, np.ones(w)/w, mode="valid")
            ax3.plot(range(w-1, len(speeds)), roll,
                     color=CYAN, lw=2.0, alpha=0.9, label=f"Rolling {w}f")
    ax3.set_title("PROBE SPEED", fontsize=9, color=CYAN, pad=8, fontweight="bold")
    ax3.set_xlabel("Detection", fontsize=7)
    ax3.set_ylabel("px/s", fontsize=7)
    ax3.tick_params(labelsize=7)
    ax3.spines[:].set_color(BORDER)
    ax3.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7, framealpha=0.7)

    # 4. Overall gauge
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor(PANEL2)
    ax4.set_xlim(0,1); ax4.set_ylim(0,1); ax4.axis("off")
    oc = score_color(overall)
    for r, a in [(0.46,0.04),(0.43,0.08),(0.40,0.14),(0.37,0.20)]:
        ax4.add_patch(plt.Circle((0.5,0.50), r, color=oc, alpha=a, zorder=1))
    ax4.add_patch(plt.Circle((0.5,0.50), 0.34, color=PANEL, zorder=2))
    ax4.text(0.5, 0.53, f"{overall:.0f}", ha="center", va="center",
             fontsize=50, fontweight="bold", color=oc, zorder=4)
    ax4.text(0.5, 0.25, "/ 100",   ha="center", color=GREY, fontsize=11)
    ax4.text(0.5, 0.88, "OVERALL", ha="center", color=WHITE, fontsize=9, fontweight="bold")
    ax4.text(0.5, 0.08, f"GRADE  {grade}", ha="center", color=oc,
             fontsize=14, fontweight="bold")

    # 5. Arthroscope jitter
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_facecolor(PANEL2)
    if jitters:
        ax5.fill_between(range(len(jitters)), jitters, alpha=0.10, color=RED)
        ax5.plot(jitters, color=RED, lw=0.9, alpha=0.8)
        ax5.axhline(jitter_mean, color=CYAN, lw=1.8, linestyle="--",
                    label=f"Mean {jitter_mean:.1f}px")
        ax5.axhline(15, color=GREEN, lw=1.2, linestyle=":",
                    label="Expert (15px)")
    ax5.set_title("ARTHROSCOPE JITTER  (stability)", fontsize=9,
                  color=CYAN, pad=8, fontweight="bold")
    ax5.set_xlabel("Frame", fontsize=7)
    ax5.set_ylabel("px/frame", fontsize=7)
    ax5.tick_params(labelsize=7)
    ax5.spines[:].set_color(BORDER)
    ax5.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7, framealpha=0.7)

    # 6. Radar
    ax6 = fig.add_subplot(gs[1, 1], polar=True)
    ax6.set_facecolor(PANEL2)
    categories = ["Stability\n40%", "Efficiency\n30%", "Precision\n30%"]
    N    = 3
    vals = [stability, efficiency, precision]
    vals_plot = vals + vals[:1]
    angles    = [n / N * 2 * 3.14159 for n in range(N)] + [0]
    for ring in [50, 75, 100]:
        ax6.plot(angles, [ring]*4, color=BORDER, lw=0.7, alpha=0.5)
        ax6.text(0.1, ring, str(ring), color=GREY, fontsize=6, ha="left")
    expert_b = EXPERT_BASELINES[level]
    exp_vals = [expert_b["stability"], expert_b["efficiency"],
                expert_b["precision"]]
    ax6.plot(angles, exp_vals + exp_vals[:1], color=PURPLE, lw=1.5,
             linestyle="--", alpha=0.7, label=f"{level.title()} expert")
    ax6.fill(angles, vals_plot, color=GREEN, alpha=0.18)
    ax6.plot(angles, vals_plot, color=GREEN, lw=2.5)
    ax6.scatter(angles[:-1], vals, color=GREEN, s=50, zorder=6)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, color=WHITE, fontsize=7.5, fontweight="bold")
    ax6.set_ylim(0, 110); ax6.set_yticks([])
    ax6.set_title("SKILL RADAR", color=CYAN, fontsize=9, pad=18, fontweight="bold")
    ax6.spines["polar"].set_color(BORDER)
    ax6.grid(color=BORDER, lw=0.4, alpha=0.6)
    ax6.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=6.5,
               loc="lower right", framealpha=0.7)

    # 7. Path ratio bar
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_facecolor(PANEL2)
    bars = ax7.bar(["Ideal", "Your path"], [1.0, path_ratio],
                   color=[GREEN, score_color(precision)], alpha=0.85, width=0.4)
    for bar, val in zip(bars, [1.0, path_ratio]):
        ax7.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.2f}x", ha="center", color=WHITE, fontsize=9, fontweight="bold")
    ax7.axhline(1.0, color=GREEN, lw=1.2, linestyle=":", alpha=0.7)
    ax7.set_title("PRECISION  (path ratio)", fontsize=9,
                  color=CYAN, pad=8, fontweight="bold")
    ax7.set_ylabel("Path / straight line", fontsize=7)
    ax7.tick_params(labelsize=8)
    ax7.spines[:].set_color(BORDER)
    ax7.set_ylim(0, max(3.5, path_ratio + 0.5))

    # 8. Expert comparison
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_facecolor(PANEL2)
    metric_names = ["Stability", "Efficiency", "Precision"]
    t_vals = [stability, efficiency, precision]
    e_vals = [expert_b["stability"], expert_b["efficiency"],
              expert_b["precision"]]
    x = np.arange(len(metric_names))
    ax8.bar(x - 0.18, t_vals, width=0.35, color=CYAN,   alpha=0.8, label="You")
    ax8.bar(x + 0.18, e_vals, width=0.35, color=PURPLE, alpha=0.6,
            label=f"{level.title()} expert")
    ax8.set_xticks(x)
    ax8.set_xticklabels(metric_names, fontsize=7, rotation=15)
    ax8.set_ylim(0, 115)
    ax8.set_title(f"VS {level.upper()} EXPERT", fontsize=9,
                  color=CYAN, pad=8, fontweight="bold")
    ax8.spines[:].set_color(BORDER)
    ax8.tick_params(labelsize=7)
    ax8.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7, framealpha=0.7)

    # AI feedback section
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.220, 0.220],
                   transform=fig.transFigure, color=BORDER, lw=1.2))
    fig.text(0.03, 0.205,
             f"AI MENTOR FEEDBACK  ·  YOLOv8  ·  {filtered_jumps} camera jumps filtered",
             ha="left", fontsize=11, color=CYAN, fontweight="bold")
    badge_data = [
        ("STABILITY",  f"{stability:.0f}/100",  score_color(stability),  "40% · arthroscope"),
        ("EFFICIENCY", f"{efficiency:.0f}/100", score_color(efficiency), "30% · probe moves"),
        ("PRECISION",  f"{precision:.0f}/100",  score_color(precision),  "30% · probe path"),
        ("OVERALL",    f"{overall:.0f}/100",    score_color(overall),    f"Grade {grade}"),
    ]
    for i, (name, val, col, sub) in enumerate(badge_data):
        xp = 0.03 + i * 0.115
        fig.text(xp, 0.175, name, ha="left", fontsize=6.5, color=GREY)
        fig.text(xp, 0.155, val,  ha="left", fontsize=11,  color=col, fontweight="bold")
        fig.text(xp, 0.138, sub,  ha="left", fontsize=6,   color=GREY)

    wrapped = textwrap.fill(ai_comment, width=130)
    fig.text(0.03, 0.118, wrapped, ha="left", va="top", fontsize=9.5, color=WHITE,
             linespacing=1.65,
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#0c1a2e",
                       edgecolor=BORDER, linewidth=1.2))
    fig.text(0.5, 0.008,
             "SurgiScore v3.1  ·  YOLOv8 + Camera Jump Filter  ·  Medathon 2026  ·  PS#1",
             ha="center", fontsize=7, color=GREY)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

