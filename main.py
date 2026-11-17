
import csv
import cv2 as cv
import yaml

from pathlib import Path
from mypylibs.optical_flow import calculate_optical_flow
from mypylibs.kalman import run_kalman
from mypylibs.occlusion import split_trajectories
from mypylibs.trajectory_video import render_trajectories


def save_track(track, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "measured", "rejected", "reset",
            "pred_x", "pred_y", "pred_vx", "pred_vy",
            "corr_x", "corr_y", "corr_vx", "corr_vy",
            "P_x", "P_y", "P_vx", "P_vy", "K_x", "K_vx"
        ])
        for t in track:
            corrected = t["corrected"] if t["corrected"] is not None else [""] * 4
            K = t["K"]
            writer.writerow([
                t["frame"], int(t["measured"]), int(t["rejected"]), int(t["reset"]),
                *[f"{v:.2f}" for v in t["predicted"]],
                *[v if v == "" else f"{v:.2f}" for v in corrected],
                *[f"{v:.2f}" for v in t["P"].diagonal()],
                "" if K is None else f"{K[0, 0]:.3f}",
                "" if K is None else f"{K[2, 2]:.3f}",
            ])


def main():

    input_dir = Path("in")
    output_dir = Path("out")

    output_dir.mkdir(exist_ok=True)

    summary = []

    for video_number in range(1, 7):

        name = f"{video_number:02d}"

        input_path = input_dir / f"{name}.mp4"
        output_path = output_dir / f"{name}_of.mp4"

        # Optional per-video settings, e.g. a region of interest
        config_path = input_dir / f"{name}.yaml"
        config = {}
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}

        measurements = calculate_optical_flow(
            str(input_path),
            str(output_path),
            config.get("of")
        )

        # Kalman filter: predict every frame, correct when OF has a measurement
        cap = cv.VideoCapture(str(input_path))
        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()

        track = run_kalman(measurements, n_frames, config.get("kf"))
        save_track(track, output_dir / f"{name}_kf.csv")

        corrected = sum(t["measured"] for t in track)
        rejected = sum(t["rejected"] for t in track)
        resets = sum(t["reset"] for t in track)

        print(
            f"Video {name}: "
            f"{len(measurements)} measurements, "
            f"track from frame {track[0]['frame'] if track else '-'}, "
            f"{corrected} corrections, "
            f"{rejected} rejected by gate, "
            f"{resets} resets, "
            f"{len(track) - corrected} prediction-only frames"
        )
        print(f"Saved: {output_path}")

        # Occlusion: compare the prediction with the object once it is back, and
        # split into the real trajectory and the one the filter would have kept
        split = split_trajectories(
            track, measurements, n_frames,
            config.get("occlusion"), config.get("kf")
        )

        tr_path = output_dir / f"{name}_tr.mp4"
        render_trajectories(
            str(input_path), str(tr_path),
            track, measurements, split, config.get("tr")
        )

        if split and split["errors"]:
            e = split["errors"]
            summary.append({
                "video": name,
                "hidden": split["hidden"],
                "reappear": e["frame"],
                "hidden_frames": e["frame"] - split["hidden"],
                "pos_error_px": round(e["pos_error"], 1),
                "sigma": round(e["sigma"], 2),
                "vel_error_px_per_frame": round(e["vel_error"], 2),
                "mean_ghost_error_px": None if e["mean_ghost_error"] is None else round(e["mean_ghost_error"], 1),
            })
            print(
                f"  occluded {split['hidden']}-{e['frame'] - 1}, "
                f"prediction error at reappearance {e['pos_error']:.0f} px "
                f"({e['sigma']:.1f} sigma), velocity error {e['vel_error']:.1f} px/frame"
            )
        else:
            print("  no occlusion with reappearance found")
        print(f"Saved: {tr_path}")

    if summary:
        with open(output_dir / "reappearance.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


if __name__ == "__main__":
    main()
