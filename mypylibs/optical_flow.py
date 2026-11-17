import cv2 as cv
import numpy as np

from collections import deque


# Shi-Tomasi corner detection (cv.goodFeaturesToTrack)
FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

# Pyramidal Lucas-Kanade (cv.calcOpticalFlowPyrLK)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)

# Defaults, can be overridden per video through in/XX.yaml (key "of")
DEFAULTS = dict(
    redetect_every=2,   # look for new corners every n frames
    diff_thresh=12,     # grey-level change between frames that counts as motion
    fb_max_err=1.0,     # px, max forward-backward LK error for a track to be trusted
    min_speed=1.0,      # px/frame, slower keypoints are static background
    min_points=10,      # fewer object keypoints than this -> no measurement (occluded or noise)
    cluster_radius=40,  # px, moving keypoints closer than ~2x this belong to the same object
    roi=None,           # [x0, y0, x1, y1] px, only keypoints inside can be the object
    centre="pooled",    # "pooled": rolling weighted average over recent frames, "median": this frame only
    centre_window=10,   # frames of object keypoints pooled for the centre
    centre_decay=0.8,   # weight of a frame's keypoints is decay^age (age in frames)
)


class CentreEstimator:
    """Object centre from a weighted rolling average of its keypoints over recent frames.

    In any one frame the corners sit on whichever part of the object has texture or
    contrast (often one edge), and they come and go, so their median jumps around and
    is pulled towards that part. Keypoints from earlier frames are shifted forward by
    the object's displacement since then, so together they cover more of the object;
    on a rolling object a tracked point also circles the centre, so its shifted
    history averages out towards it. Newer frames get more weight because the shift
    accumulates error.

    Its noise: each pooled frame is treated as a roughly independent view of the
    object, so the keypoint spread shrinks with the square root of the effective
    number of frames (the sum of their weights). That ignores bias common to all
    frames, so once there is enough history it is never allowed below the measured
    residual noise: from one frame to the next the centre should move by the
    measured velocity, and what is left over is noise (the difference of two noisy
    positions has twice their variance, hence the /2).
    """

    def __init__(self, window, decay):
        self.decay = decay
        self.frames = deque(maxlen=window)
        self.residuals = deque(maxlen=window)
        self.last = None

    def reset(self):
        self.frames.clear()
        self.residuals.clear()
        self.last = None

    def update(self, points, velocity):
        """Returns the centre and its per-axis noise std."""
        # Bring earlier keypoints forward by this frame's object displacement
        for f in self.frames:
            f += velocity
        self.frames.appendleft(points.astype(np.float64).copy())

        pts = np.vstack(self.frames)
        w = np.concatenate([
            np.full(len(f), self.decay ** age) for age, f in enumerate(self.frames)
        ])
        c = (w[:, None] * pts).sum(axis=0) / w.sum()

        if self.last is not None:
            self.residuals.append(c - (self.last + velocity))
        self.last = c

        # A variance from a handful of residuals can easily be several times too
        # small, so only trust it once the window is full
        # Spread of this frame's keypoints, shrunk by the effective number of frames
        n_eff = sum(self.decay ** age for age in range(len(self.frames)))
        noise = points.std(axis=0) / np.sqrt(n_eff)
        if len(self.residuals) == self.residuals.maxlen:
            noise = np.maximum(noise, np.sqrt((np.array(self.residuals) ** 2).mean(axis=0) / 2))
        return c, noise


def motion_mask(old_gray, gray, cfg):
    """Pixels that changed between frames, dilated so the object's edges are included."""
    diff = cv.absdiff(
        cv.GaussianBlur(old_gray, (5, 5), 0),
        cv.GaussianBlur(gray, (5, 5), 0)
    )
    _, mask = cv.threshold(diff, cfg["diff_thresh"], 255, cv.THRESH_BINARY)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv.dilate(mask, np.ones((15, 15), np.uint8))


def detect_keypoints(gray, existing, mask):
    """Add corners inside mask that are not already being tracked."""
    mask = mask.copy()
    for x, y in existing.reshape(-1, 2):
        cv.circle(mask, (int(x), int(y)), FEATURE_PARAMS["minDistance"], 0, -1)

    new = cv.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
    if new is None:
        return existing
    return np.vstack([existing, new.astype(np.float32)])


def cluster_labels(points, radius, shape):
    """Label points lying within ~radius of each other as one connected cluster."""
    canvas = np.zeros(shape, np.uint8)
    for x, y in np.int32(points):
        cv.circle(canvas, (int(x), int(y)), radius, 255, -1)
    _, labels = cv.connectedComponents(canvas)
    xs = np.clip(np.int32(points[:, 0]), 0, shape[1] - 1)
    ys = np.clip(np.int32(points[:, 1]), 0, shape[0] - 1)
    return labels[ys, xs]


def select_object(points, velocities, shape, cfg):
    """Pick the keypoints belonging to the moving object.

    Returns a per-keypoint label: 0 static, 1 moving but not the object, 2 object.
    """
    label = np.zeros(len(points), np.uint8)
    moving = np.linalg.norm(velocities, axis=1) >= cfg["min_speed"]

    if cfg["roi"] is not None:
        x0, y0, x1, y1 = cfg["roi"]
        moving &= (
            (points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 1] >= y0) & (points[:, 1] <= y1)
        )

    label[moving] = 1
    if moving.sum() < cfg["min_points"]:
        return label

    # Other things move too (hands, flicker, occluder edges): keep the biggest cluster
    idx = np.flatnonzero(moving)
    clusters = cluster_labels(points[idx], cfg["cluster_radius"], shape)
    ids, counts = np.unique(clusters, return_counts=True)
    idx = idx[clusters == ids[np.argmax(counts)]]

    if len(idx) >= cfg["min_points"]:
        label[idx] = 2
    return label


def draw(frame, old_points, new_points, label, measurement, frame_number, pooled=()):

    # Earlier keypoints shifted to this frame, fading with age (see CentreEstimator)
    for age, points in enumerate(pooled):
        if age == 0:
            continue
        shade = int(255 * (1 - age / len(pooled)))
        for x, y in points:
            cv.circle(frame, (int(x), int(y)), 2, (shade, 0, shade), -1)

    for old, new, lb in zip(old_points, new_points, label):
        a, b = tuple(np.int32(old)), tuple(np.int32(new))
        if lb == 2:
            cv.arrowedLine(frame, a, b, (0, 255, 0), 2, tipLength=0.4)
        elif lb == 1:
            cv.arrowedLine(frame, a, b, (0, 200, 255), 1, tipLength=0.4)
        else:
            cv.circle(frame, b, 2, (160, 160, 160), -1)

    if measurement is not None:
        x, y = measurement["x"], measurement["y"]
        vx, vy = measurement["vx"], measurement["vy"]
        # This frame's keypoint median (hollow) vs the pooled centre (filled)
        cv.circle(frame, (int(measurement["x_raw"]), int(measurement["y_raw"])), 8, (0, 0, 255), 2)
        cv.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
        # Velocity scaled x5 so it is visible
        cv.arrowedLine(
            frame, (int(x), int(y)), (int(x + 5 * vx), int(y + 5 * vy)),
            (0, 0, 255), 3, tipLength=0.3
        )
        text = (
            f"#{frame_number}  pos=({x:.0f},{y:.0f})  "
            f"vel=({vx:+.1f},{vy:+.1f}) px/f  n={measurement['n']}"
        )
    else:
        text = f"#{frame_number}  no measurement"

    lines = [
        text,
        "green: object keypoints  yellow: other motion  grey: static",
        "purple: earlier keypoints shifted to now  red dot: centre (hollow: this frame's median)",
    ]
    cv.putText(frame, lines[0], (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(frame, lines[0], (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Legend on a dark box: outlined small text is unreadable
    width = max(cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines[1:])
    overlay = frame.copy()
    cv.rectangle(overlay, (12, 55), (28 + width, 55 + 26 * (len(lines) - 1) + 8), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines[1:]):
        cv.putText(frame, line, (20, 76 + 26 * i), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)


def calculate_optical_flow(video_path, output_path, config=None):
    """Track the moving object with Shi-Tomasi corners and pyramidal LK.

    Returns one measurement per frame where the object is visible: its centre
    (see CentreEstimator; x_raw/y_raw is this frame's keypoint median), velocity
    (median keypoint flow, px/frame), their noise std (sx, sy, svx, svy) and the
    number of keypoints used. Frames without a measurement
    (e.g. object occluded) are left out. Writes the visualisation to output_path.
    """
    cfg = {**DEFAULTS, **(config or {})}

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, old_frame = cap.read()

    if not ret:
        cap.release()
        out.release()
        raise ValueError("Cannot read first frame")

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    draw(old_frame, [], [], [], None, 0)
    out.write(old_frame)

    p0 = np.empty((0, 1, 2), np.float32)
    centre = CentreEstimator(cfg["centre_window"], cfg["centre_decay"])
    measurements = []
    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        old_points = new_points = np.empty((0, 2), np.float32)
        label = np.empty(0, np.uint8)
        measurement = None

        if len(p0):
            # Track forwards, then back again: a reliable track returns to where it started
            p1, status, _ = cv.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
            back, status_back, _ = cv.calcOpticalFlowPyrLK(gray, old_gray, p1, None, **LK_PARAMS)
            fb_err = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
            good = (
                (status.ravel() == 1) &
                (status_back.ravel() == 1) &
                (fb_err < cfg["fb_max_err"])
            )

            old_points = p0.reshape(-1, 2)[good]
            new_points = p1.reshape(-1, 2)[good]
            velocities = new_points - old_points

            label = select_object(new_points, velocities, (height, width), cfg)
            obj = label == 2

            if obj.any():
                pos, vel = new_points[obj], velocities[obj]
                vx, vy = np.median(vel, axis=0)
                x_raw, y_raw = np.median(pos, axis=0)

                # Position noise: the spread of this frame's keypoints, or the pooled
                # centre's own estimate (see CentreEstimator)
                sx, sy = pos.std(axis=0)
                if cfg["centre"] == "pooled":
                    (x, y), (sx, sy) = centre.update(pos, np.array([vx, vy]))
                else:
                    x, y = x_raw, y_raw
                svx, svy = vel.std(axis=0)
                measurement = {
                    "frame": frame_number,
                    "x": float(x), "y": float(y),
                    "x_raw": float(x_raw), "y_raw": float(y_raw),
                    "vx": float(vx), "vy": float(vy),
                    "sx": float(sx), "sy": float(sy),
                    "svx": float(svx), "svy": float(svy),
                    "n": int(obj.sum())
                }
                measurements.append(measurement)

            p0 = new_points.reshape(-1, 1, 2)

        if measurement is None:
            # The object's displacement is unknown while it is not seen, so start afresh
            centre.reset()

        # Only look for new corners where something moved, the static background is useless
        if frame_number % cfg["redetect_every"] == 0:
            p0 = detect_keypoints(gray, p0, motion_mask(old_gray, gray, cfg))

        draw(frame, old_points, new_points, label, measurement, frame_number, centre.frames)
        out.write(frame)
        old_gray = gray

    cap.release()
    out.release()

    return measurements
