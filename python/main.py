
from pathlib import Path

import cv2
import mediapipe as mp

# 1. Duong dan cua du an
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "2_student.mp4"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker.task"

# 2. Kiem tra cac file dau vao
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay video: {VIDEO_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

# 3. Cau hinh Pose Landmarker cho video
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

# 4. Cac cap diem can noi thanh bo xuong
CONNECTIONS = [
    (11, 12),  # Hai vai
    (11, 13), (13, 15),  # Tay trai
    (12, 14), (14, 16),  # Tay phai
    (11, 23), (12, 24), (23, 24),  # Than nguoi
    (23, 25), (25, 27),  # Chan trai
    (24, 26), (26, 28),  # Chan phai
    (27, 29), (29, 31),  # Ban chan trai
    (28, 30), (30, 32),  # Ban chan phai
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (27, 31), (28, 32),
]


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    # Chuyen toa do chuan hoa thanh toa do pixel
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Ve cac duong noi khop
    for start, end in CONNECTIONS:
        p1 = points[start]
        p2 = points[end]

        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    # Ve cac diem moc
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_index = 0
    detected_frames = 0

    print("Dang nhan dien tu the...")
    print("Nhan Q de thoat.")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                # Moi khung hinh co timestamp tang dan
                timestamp_ms = int(frame_index * 1000 / fps)
                frame_index += 1

                # OpenCV BGR -> MediaPipe RGB
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

                if result.pose_landmarks:
                    detected_frames += 1
                    draw_pose(frame, result.pose_landmarks[0])

                    cv2.putText(
                        frame,
                        "Pose detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "No pose detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                window_name = "Danceup - Pose Detection"

                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 700, 1050)

                cv2.imshow(window_name, frame)  

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Tong so khung hinh da xu ly: {frame_index}")
    print(f"So khung hinh nhan dien duoc nguoi: {detected_frames}")
    print("Da ket thuc.")


if __name__ == "__main__":
    main()
