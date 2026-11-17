
from pathlib import Path
from mypylibs.optical_flow import calculate_optical_flow


def main():

    input_dir = Path("in")
    output_dir = Path("out")

    output_dir.mkdir(exist_ok=True)

    for video_number in range(11, 14):

        name = f"{video_number:02d}"

        input_path = input_dir / f"{name}.mp4"
        output_path = output_dir / f"{name}_of.mp4"

        measurements = calculate_optical_flow(
            str(input_path),
            str(output_path)
        )

        print(
            f"Video {name}: "
            f"{len(measurements)} measurements"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
