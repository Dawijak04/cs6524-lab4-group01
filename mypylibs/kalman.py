import cv2 as cv
import numpy as np


# State x = [x, y, vx, vy] in px and px/frame (constant-velocity model, dt = 1 frame).
# Measurement z = [x, y, vx, vy]: optical flow gives both position and velocity.
DEFAULTS = dict(
    accel_std=0.5,       # px/frame^2, unmodelled acceleration (friction, bumps, pushes) -> Q
    min_pos_std=2.0,     # px, floor on measurement position noise -> R
    min_vel_std=0.5,     # px/frame, floor on measurement velocity noise -> R
    init_pos_std=10.0,   # px, extra initial position uncertainty -> P0
    init_vel_std=5.0,    # px/frame, extra initial velocity uncertainty -> P0
    gate=18.47,          # chi-square (4 dof, 99.9%) gate on the innovation; None disables
    init_frames=3,       # consecutive consistent measurements needed to start the track
    init_min_speed=5.0,  # px/frame, slower motion (jitter, hands settling) cannot start a track
    init_vel_tol=3.0,    # px/frame, max velocity change between frames while starting
    reset_after=3,       # consecutive gated-out but mutually consistent measurements that re-start the track
    reset_min_speed=2.0, # px/frame, slower rejected measurements never re-start the track
)


def transition_matrix():
    """F: position advances by one frame of velocity, velocity stays constant."""
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], np.float32)


def process_noise(accel_std):
    """Q for a random (white-noise) acceleration of std accel_std acting over one frame.

    An acceleration a changes position by a/2 and velocity by a, so per axis
    Q = accel_std^2 * [[1/4, 1/2], [1/2, 1]].
    """
    g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], np.float32)
    return (accel_std ** 2 * g @ g.T).astype(np.float32)


def measurement_noise(m, cfg):
    """R from the spread of the object's keypoints in this frame.

    The error is not independent per keypoint (it depends on which part of the
    object happens to have corners), so the spread is used as-is, not divided by
    sqrt(n). The floors stop R collapsing when only a few points agree by chance.
    """
    std = np.array([
        max(m["sx"], cfg["min_pos_std"]),
        max(m["sy"], cfg["min_pos_std"]),
        max(m["svx"], cfg["min_vel_std"]),
        max(m["svy"], cfg["min_vel_std"]),
    ], np.float32)
    return np.diag(std ** 2).astype(np.float32)


def as_vector(m):
    return np.array([[m["x"]], [m["y"]], [m["vx"]], [m["vy"]]], np.float32)


def is_steady(run, min_speed, cfg):
    """True if run is consecutive frames of motion at >= min_speed with a stable velocity."""
    if run[-1]["frame"] - run[0]["frame"] != len(run) - 1:
        return False
    v = np.array([[m["vx"], m["vy"]] for m in run])
    speed = np.linalg.norm(v, axis=1)
    change = np.linalg.norm(np.diff(v, axis=0), axis=1)
    return speed.min() >= min_speed and np.all(change <= max(cfg["init_vel_tol"], 0.3 * speed.min()))


def find_start(measurements, cfg):
    """Index of the first measurement that starts a confirmed track, or None.

    Single measurements can come from jitter or a hand, so the track only starts
    after init_frames consecutive frames of fast, consistent motion.
    """
    k = cfg["init_frames"]
    for i in range(len(measurements) - k + 1):
        if is_steady(measurements[i:i + k], cfg["init_min_speed"], cfg):
            return i
    return None


def create_kalman(first, cfg):
    """Build a cv.KalmanFilter initialised on the first optical flow measurement."""
    kf = cv.KalmanFilter(4, 4)
    kf.transitionMatrix = transition_matrix()
    kf.measurementMatrix = np.eye(4, dtype=np.float32)
    kf.processNoiseCov = process_noise(cfg["accel_std"])
    kf.measurementNoiseCov = measurement_noise(first, cfg)

    # P0: the spread of the keypoints (how far the tracked point may be from the
    # object's true reference point) plus a fixed margin for the first estimate
    init = np.array([
        first["sx"] ** 2 + cfg["init_pos_std"] ** 2,
        first["sy"] ** 2 + cfg["init_pos_std"] ** 2,
        first["svx"] ** 2 + cfg["init_vel_std"] ** 2,
        first["svy"] ** 2 + cfg["init_vel_std"] ** 2,
    ], np.float32)
    kf.errorCovPost = np.diag(init).astype(np.float32)
    kf.statePost = as_vector(first)
    return kf


def run_kalman(measurements, n_frames, config=None):
    """Predict every frame and correct whenever optical flow gives a measurement.

    The track starts at the first confirmed measurement (see find_start).

    Returns one dict per frame from the first measurement on:
      frame, predicted (4,), corrected (4,) or None, P (4x4), K (4x4) or None,
      measured (bool), rejected (bool, measurement failed the gate),
      reset (bool, filter re-initialised on this measurement).
    """
    cfg = {**DEFAULTS, **(config or {})}
    start = find_start(measurements, cfg)
    if start is None:
        return []

    by_frame = {m["frame"]: m for m in measurements[start:]}
    first = measurements[start]
    kf = create_kalman(first, cfg)
    rejected_run = []

    track = [{
        "frame": first["frame"],
        "predicted": kf.statePost.ravel().copy(),
        "corrected": kf.statePost.ravel().copy(),
        "P": kf.errorCovPost.copy(),
        "K": None,
        "measured": True,
        "rejected": False,
        "reset": False,
    }]

    for frame in range(first["frame"] + 1, n_frames):
        predicted = kf.predict().ravel().copy()
        step = {
            "frame": frame,
            "predicted": predicted,
            "corrected": None,
            "P": kf.errorCovPre.copy(),
            "K": None,
            "measured": False,
            "rejected": False,
            "reset": False,
        }

        m = by_frame.get(frame)
        if m is not None:
            z = as_vector(m)
            R = measurement_noise(m, cfg)

            # Reject measurements that are implausible given the prediction
            # (e.g. the hand instead of the object). S grows while occluded,
            # so the gate widens and the object can be reacquired.
            y = z - kf.measurementMatrix @ kf.statePre
            S = kf.measurementMatrix @ kf.errorCovPre @ kf.measurementMatrix.T + R
            d2 = (y.T @ np.linalg.inv(S) @ y).item()

            if cfg["gate"] is None or d2 <= cfg["gate"]:
                kf.measurementNoiseCov = R
                corrected = kf.correct(z).ravel().copy()
                step.update(
                    corrected=corrected,
                    P=kf.errorCovPost.copy(),
                    K=kf.gain.copy(),
                    measured=True,
                )
                rejected_run = []
            else:
                step["rejected"] = True
                rejected_run.append(m)

                # Several rejected measurements in a row that agree with each other mean
                # the object did something the constant-velocity model cannot follow
                # (sudden stop, wrong initial velocity): start again from the measurements
                k = cfg["reset_after"]
                if len(rejected_run) >= k and is_steady(rejected_run[-k:], cfg["reset_min_speed"], cfg):
                    kf = create_kalman(m, cfg)
                    step.update(
                        corrected=kf.statePost.ravel().copy(),
                        P=kf.errorCovPost.copy(),
                        measured=True,
                        rejected=False,
                        reset=True,
                    )
                    rejected_run = []
        else:
            rejected_run = []

        # Without a correction, predict() has already copied statePre into statePost,
        # so the next prediction carries on from this one
        track.append(step)

    return track
