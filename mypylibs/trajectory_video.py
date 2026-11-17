import cv2 as cv
import numpy as np


# BGR colours
MEASURED = (255, 255, 255)   # optical flow measurement
CORRECTED = (0, 200, 0)      # filter, corrected by a measurement
PREDICTED = (0, 0, 255)      # filter, prediction only (no measurement)
GHOST = (255, 0, 255)        # prediction if the object had never come back
VELOCITY = (255, 200, 0)
ERROR = (0, 255, 255)

DEFAULTS = dict(
    velocity_every=5,   # draw a velocity vector every n frames
    velocity_scale=3,   # velocity vectors are drawn this many frames long
    sigma=2,            # uncertainty ellipse at this many standard deviations
)


def _pt(state):
    return int(round(state[0])), int(round(state[1]))


def _position(t):
    return t["corrected"] if t["corrected"] is not None else t["predicted"]


def _draw_ellipse(frame, state, P, colour, n_sigma):
    axes = (int(n_sigma * np.sqrt(P[0, 0])), int(n_sigma * np.sqrt(P[1, 1])))
    cv.ellipse(frame, _pt(state), axes, 0, 0, 360, colour, 1, cv.LINE_AA)


def _draw_dashed(frame, points, colour, thickness=2, on=2):
    """Polyline with every other run of `on` segments left out."""
    for i in range(len(points) - 1):
        if (i // on) % 2 == 0:
            cv.line(frame, points[i], points[i + 1], colour, thickness, cv.LINE_AA)


def _legend(frame, items):
    x, y = 15, 15
    h = 28 * len(items) + 12
    overlay = frame.copy()
    cv.rectangle(overlay, (x, y), (x + 430, y + h), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (colour, text) in enumerate(items):
        cy = y + 22 + 28 * i
        cv.line(frame, (x + 12, cy), (x + 48, cy), colour, 3, cv.LINE_AA)
        cv.putText(frame, text, (x + 60, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


def _text(frame, text, org, scale=0.7):
    cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv.LINE_AA)


def render_trajectories(video_path, output_path, track, measurements, split, config=None):
    """Draw the measured, predicted, corrected and ghost trajectories on the video."""
    cfg = {**DEFAULTS, **(config or {})}

    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    by_frame = {t["frame"]: t for t in track}
    measured = {m["frame"]: m for m in measurements}
    ghost = {g["frame"]: g for g in split["ghost"]} if split else {}
    errors = split["errors"] if split else None
    margin = max(width, height)

    frame_number = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # Measurements so far (raw optical flow)
        for f, m in measured.items():
            if f <= frame_number:
                cv.circle(frame, (int(m["x"]), int(m["y"])), 3, MEASURED, -1, cv.LINE_AA)

        # Velocity of the filter state every few frames, under the trajectories
        history = [t for t in track if t["frame"] <= frame_number]
        for t in history:
            if t["frame"] % cfg["velocity_every"] == 0:
                s = _position(t)
                tip = (int(s[0] + cfg["velocity_scale"] * s[2]), int(s[1] + cfg["velocity_scale"] * s[3]))
                cv.arrowedLine(frame, _pt(s), tip, VELOCITY, 1, cv.LINE_AA, tipLength=0.3)

        # Filter track so far: green where corrected, red where only predicted
        for a, b in zip(history, history[1:]):
            colour = CORRECTED if b["measured"] else PREDICTED
            cv.line(frame, _pt(_position(a)), _pt(_position(b)), colour, 3, cv.LINE_AA)

        # Ghost: what the filter would keep predicting if the object had not come back
        ghost_pts = [
            _pt(g["state"]) for f, g in sorted(ghost.items())
            if f <= frame_number
            and -margin < g["state"][0] < width + margin and -margin < g["state"][1] < height + margin
        ]
        if len(ghost_pts) > 1:
            _draw_dashed(frame, ghost_pts, GHOST)
        if frame_number in ghost:
            g = ghost[frame_number]
            cv.drawMarker(frame, _pt(g["state"]), GHOST, cv.MARKER_TILTED_CROSS, 18, 2, cv.LINE_AA)
            _draw_ellipse(frame, g["state"], g["P"], GHOST, cfg["sigma"])

        # Error at reappearance: predicted vs measured position
        if errors and frame_number >= errors["frame"] and errors["frame"] in measured:
            m = measured[errors["frame"]]
            g = ghost[errors["frame"]]
            cv.line(frame, _pt(g["state"]), (int(m["x"]), int(m["y"])), ERROR, 2, cv.LINE_AA)

        # Current filter state and its uncertainty
        current = by_frame.get(frame_number)
        if current is not None:
            colour = CORRECTED if current["measured"] else PREDICTED
            cv.circle(frame, _pt(_position(current)), 9, colour, -1, cv.LINE_AA)
            _draw_ellipse(frame, _position(current), current["P"], colour, cfg["sigma"])
            status = "measured + corrected" if current["measured"] else "predicting (no measurement)"
            if current["rejected"]:
                status = "predicting (measurement rejected)"
        else:
            status = "no track yet"

        _legend(frame, [
            (MEASURED, "measured (optical flow)"),
            (CORRECTED, "Kalman: corrected"),
            (PREDICTED, "Kalman: predicted only"),
            (GHOST, "if it had not come back (no corrections)"),
            (ERROR, "prediction error at reappearance"),
            (VELOCITY, f"velocity x{cfg['velocity_scale']}, every {cfg['velocity_every']} frames"),
        ])
        _text(frame, f"frame {frame_number}: {status}", (15, height - 20))

        if errors and frame_number >= errors["frame"]:
            lines = [
                f"reappeared at frame {errors['frame']} after {errors['frame'] - split['hidden']} hidden frames",
                f"prediction error {errors['pos_error']:.0f} px ({errors['sigma']:.1f} sigma), "
                f"velocity error {errors['vel_error']:.1f} px/frame",
            ]
            for i, line in enumerate(lines):
                (w, _), _ = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, 0.6, 4)
                _text(frame, line, (width - 15 - w, height - 50 + 30 * i), 0.6)

        out.write(frame)

    cap.release()
    out.release()
