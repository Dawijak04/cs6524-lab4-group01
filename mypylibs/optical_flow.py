
import cv2 as cv
import numpy as np

from mypylibs.ball_detection import detect_ball


def calculate_optical_flow(video_path, output_path):

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

    feature_params = dict(
        maxCorners=30,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=3
    )

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    ret, old_frame = cap.read()

    if not ret:
        cap.release()
        out.release()
        raise ValueError("Cannot read first frame")

    old_gray = cv.cvtColor(
        old_frame, cv.COLOR_BGR2GRAY
    )

    p0 = None
    previous_center = None

    measurements = []
    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        gray = cv.cvtColor(
            frame, cv.COLOR_BGR2GRAY
        )

        # Detect the ball in the current frame
        ball, mask = detect_ball(frame)

        if ball is not None:

            x, y, radius = ball
            center = np.array(
                [x, y], dtype=np.float32
            )

            # Display detected ball
            cv.circle(
                frame, (x, y), radius,
                (255, 0, 0), 2
            )

            # Calculate ball centre velocity
            # Only when visible in consecutive frames
            if previous_center is not None:

                vx, vy = center - previous_center

                measurements.append({
                    "frame": frame_number,
                    "x": float(x),
                    "y": float(y),
                    "vx": float(vx),
                    "vy": float(vy)
                })

                cv.arrowedLine(
                    frame,
                    (int(previous_center[0]),
                     int(previous_center[1])),
                    (x, y),
                    (0, 0, 255),
                    2
                )

            previous_center = center

            # Detect keypoints inside ball region
            if p0 is None or len(p0) < 5:

                p0 = cv.goodFeaturesToTrack(
                    old_gray,
                    mask=mask,
                    **feature_params
                )

            # Lucas-Kanade optical flow
            if p0 is not None:

                p1, status, error = (
                    cv.calcOpticalFlowPyrLK(
                        old_gray,
                        gray,
                        p0,
                        None,
                        **lk_params
                    )
                )

                if p1 is not None and status is not None:

                    valid = status.flatten() == 1

                    new_points = p1[valid].reshape(-1, 2)
                    old_points = p0[valid].reshape(-1, 2)

                    # Keep points near detected ball
                    distances = np.linalg.norm(
                        new_points - center,
                        axis=1
                    )

                    inside = distances <= radius

                    new_points = new_points[inside]
                    old_points = old_points[inside]

                    for new, old in zip(
                        new_points, old_points
                    ):

                        nx, ny = new
                        ox, oy = old

                        cv.circle(
                            frame,
                            (int(nx), int(ny)),
                            3,
                            (0, 255, 0),
                            -1
                        )

                        cv.arrowedLine(
                            frame,
                            (int(ox), int(oy)),
                            (int(nx), int(ny)),
                            (0, 0, 255),
                            2
                        )

                    p0 = new_points.reshape(
                        -1, 1, 2
                    ).astype(np.float32)

                else:
                    p0 = None

        else:
            # Ball is occluded
            previous_center = None
            p0 = None

        out.write(frame)
        old_gray = gray.copy()

    cap.release()
    out.release()

    return measurements
