import numpy as np

from mypylibs.kalman import transition_matrix, process_noise, DEFAULTS as KF_DEFAULTS


DEFAULTS = dict(
    min_gap=5,        # frames; shorter gaps in the measurements are dropouts, not an occlusion
    eval_frames=10,   # frames after reappearance over which the ghost is compared to the measurements
)


def find_occlusion(track, cfg):
    """The occlusion is the longest run of frames without an accepted measurement
    that ends with the object being measured again.

    Returns (first hidden frame, reappearance frame) or None.
    """
    best, start = None, None
    for t in track:
        if not t["measured"]:
            if start is None:
                start = t["frame"]
        elif start is not None:
            gap = t["frame"] - start
            if gap >= cfg["min_gap"] and (best is None or gap > best[1] - best[0]):
                best = (start, t["frame"])
            start = None
    return best


def ghost_trajectory(state, P, first_frame, last_frame, kf_cfg):
    """Keep predicting with no corrections at all: where the filter thinks the
    object would be if it had never come back."""
    F = transition_matrix()
    Q = process_noise(kf_cfg["accel_std"])
    x, P = state.astype(np.float64), P.astype(np.float64)

    ghost = []
    for frame in range(first_frame, last_frame + 1):
        ghost.append({"frame": frame, "state": x.copy(), "P": P.copy()})
        x = F @ x
        P = F @ P @ F.T + Q
    return ghost


def split_trajectories(track, measurements, n_frames, config=None, kf_config=None):
    """Split the track at the moment the object reappears.

    Returns None if the object is never occluded and seen again, otherwise a dict:
      hidden, reappear      frames where the occlusion starts and ends
      back                  the filter's track from reappearance on (corrected when measured)
      ghost                 pure prediction from the reappearance frame on, as if the
                            object had never come back
      errors                comparison of the ghost against the measurements
    """
    cfg = {**DEFAULTS, **(config or {})}
    kf_cfg = {**KF_DEFAULTS, **(kf_config or {})}

    occlusion = find_occlusion(track, cfg)
    if occlusion is None:
        return None
    hidden, reappear = occlusion

    by_frame = {t["frame"]: t for t in track}
    at_reappear = by_frame[reappear]

    # The prediction made for the reappearance frame uses nothing seen after the
    # object disappeared, so continuing it gives the "never came back" trajectory
    ghost = ghost_trajectory(at_reappear["predicted"], at_reappear["P_pred"], reappear, n_frames - 1, kf_cfg)
    back = [t for t in track if t["frame"] >= reappear]

    measured = {m["frame"]: m for m in measurements}
    errors = reappearance_errors(ghost, back, measured, cfg)

    return {
        "hidden": hidden,
        "reappear": reappear,
        "back": back,
        "ghost": ghost,
        "errors": errors,
    }


def reappearance_errors(ghost, back, measured, cfg):
    """Error metrics for the prediction across the occlusion.

    pos_error        px, predicted vs measured position at reappearance (innovation)
    vel_error        px/frame, predicted vs measured velocity at reappearance
    sigma            position error in standard deviations of the predicted position,
                     i.e. was the error as large as the filter itself expected
    mean_ghost_error px, mean distance between the ghost and the measurements over
                     the following eval_frames (how fast the two trajectories split)
    """
    g0 = ghost[0]
    m0 = measured.get(g0["frame"])
    if m0 is None:
        # Reappearance through a filter reset carries no measurement on the frame itself
        m0 = next((measured[t["frame"]] for t in back if t["frame"] in measured), None)
    if m0 is None:
        return None

    z = np.array([m0["x"], m0["y"], m0["vx"], m0["vy"]])
    x = g0["state"]
    diff = z[:2] - x[:2]
    pos_error = float(np.linalg.norm(diff))
    vel_error = float(np.linalg.norm(z[2:] - x[2:]))
    sigma = float(np.sqrt(diff @ np.linalg.inv(g0["P"][:2, :2]) @ diff))

    ghost_by_frame = {g["frame"]: g for g in ghost}
    later = [
        np.linalg.norm(ghost_by_frame[f]["state"][:2] - [measured[f]["x"], measured[f]["y"]])
        for f in range(g0["frame"], g0["frame"] + cfg["eval_frames"])
        if f in measured and f in ghost_by_frame
    ]

    return {
        "frame": g0["frame"],
        "pos_error": pos_error,
        "vel_error": vel_error,
        "sigma": sigma,
        "mean_ghost_error": float(np.mean(later)) if later else None,
    }
