
from pathlib import Path
import csv

import cv2
import mediapipe as mp

# 1. Duong dan
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "2_student.mp4"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker.task"
OUTPUT_PATH = BASE_DIR / "landmarks_student.csv"

# 2. Cau hinh MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 3. Tao tieu de CSV
header = ["frame", "timestamp_ms", "detected"]

for i in range(33):
    header.extend([
        f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"
    ])


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay video: {VIDEO_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Khong the mo video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_index = 0
    detected_count = 0

    try:
        with open(
            OUTPUT_PATH,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(header)

            with PoseLandmarker.create_from_options(
                options
            ) as landmarker:

                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    timestamp_ms = int(
                        frame_index * 1000 / fps
                    )

                    rgb_frame = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    result = landmarker.detect_for_video(
                        mp_image, timestamp_ms
                    )

                    row = [
                        frame_index,
                        timestamp_ms,
                        0,
                    ]

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        world_landmarks = (
                            result.pose_world_landmarks[0]
                        )

                        row[2] = 1
                        detected_count += 1

                        for i, point in enumerate(landmarks):
                            world_point = world_landmarks[i]

                            row.extend([
                                point.x,
                                point.y,
                                point.z,
                                getattr(
                                    point,
                                    "visibility",
                                    1.0,
                                ),
                            ])

                    else:
                        # Khong phat hien nguoi:
                        # dien NaN cho 33 diem
                        for _ in range(33):
                            row.extend([
                                float("nan"),
                                float("nan"),
                                float("nan"),
                                float("nan"),
                            ])

                    writer.writerow(row)
                    frame_index += 1

                    if frame_index % 100 == 0:
                        print(
                            f"Da xu ly {frame_index} frames..."
                        )

    finally:
        cap.release()

    print("\nHoan tat trich xuat!")
    print(f"Tong so frames: {frame_index}")
    print(f"Frames co nhan dien: {detected_count}")
    print(f"Ty le nhan dien: "
          f"{detected_count / max(frame_index, 1) * 100:.2f}%")
    print(f"File CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
