import cv2 as cv
import numpy as np


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
)


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


def draw(frame, old_points, new_points, label, measurement, frame_number):

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

    cv.putText(frame, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(frame, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)


def calculate_optical_flow(video_path, output_path, config=None):
    """Track the moving object with Shi-Tomasi corners and pyramidal LK.

    Returns one measurement per frame where the object is visible: its position
    (median of its keypoints, px), velocity (median keypoint flow, px/frame), the
    spread of both and the number of keypoints used. Frames without a measurement
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
                x, y = np.median(pos, axis=0)
                vx, vy = np.median(vel, axis=0)
                sx, sy = pos.std(axis=0)
                svx, svy = vel.std(axis=0)
                measurement = {
                    "frame": frame_number,
                    "x": float(x), "y": float(y),
                    "vx": float(vx), "vy": float(vy),
                    "sx": float(sx), "sy": float(sy),
                    "svx": float(svx), "svy": float(svy),
                    "n": int(obj.sum())
                }
                measurements.append(measurement)

            p0 = new_points.reshape(-1, 1, 2)

        # Only look for new corners where something moved, the static background is useless
        if frame_number % cfg["redetect_every"] == 0:
            p0 = detect_keypoints(gray, p0, motion_mask(old_gray, gray, cfg))

        draw(frame, old_points, new_points, label, measurement, frame_number)
        out.write(frame)
        old_gray = gray

    cap.release()
    out.release()

    return measurements
