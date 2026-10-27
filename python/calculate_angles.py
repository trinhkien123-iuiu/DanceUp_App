
from pathlib import Path
import csv
import math

import numpy as np

# 1. Duong dan
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "landmarks_student.csv"
OUTPUT_PATH = BASE_DIR / "joint_angles_student.csv"


def get_point(row, landmark_id):
    """Lay toa do x, y, z cua mot diem tu CSV."""
    return np.array([
        float(row[f"x_{landmark_id}"]),
        float(row[f"y_{landmark_id}"]),
        float(row[f"z_{landmark_id}"]),
    ])


def calculate_angle(a, b, c):
    """
    Tinh goc ABC, voi B la khop can do.
    Tra ve goc theo don vi do.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Tranh chia cho 0
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

    # Tranh loi sai so so hoc
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def get_angle(row, a_id, b_id, c_id):
    """Tinh goc; bo qua frame khong nhan dien duoc."""
    if row["detected"] != "1":
        return float("nan")

    try:
        a = get_point(row, a_id)
        b = get_point(row, b_id)
        c = get_point(row, c_id)

        if not (
            np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
            and np.all(np.isfinite(c))
        ):
            return float("nan")

        return calculate_angle(a, b, c)

    except (ValueError, KeyError):
        return float("nan")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {INPUT_PATH}"
        )

    # Moi dong dinh nghia mot goc va ba landmark tuong ung.
    joints = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
    }

    with open(
        INPUT_PATH,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as input_file:

        reader = csv.DictReader(input_file)
        output_rows = []

        for row in reader:
            output_row = {
                "frame": row["frame"],
                "timestamp_ms": row["timestamp_ms"],
                "detected": row["detected"],
            }

            for name, (a_id, b_id, c_id) in joints.items():
                output_row[name] = get_angle(
                    row, a_id, b_id, c_id
                )

            output_rows.append(output_row)

    fieldnames = [
        "frame",
        "timestamp_ms",
        "detected",
        *joints.keys(),
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Hoan tat tinh goc khop!")
    print(f"So frames: {len(output_rows)}")
    print(f"File ket qua: {OUTPUT_PATH}")

    # In thu 5 dong dau de kiem tra
    print("\n5 dong dau tien:")

    for row in output_rows[:5]:
        print(
            f"Frame {row['frame']}: "
            f"left_elbow={float(row['left_elbow']):.2f}, "
            f"right_elbow={float(row['right_elbow']):.2f}, "
            f"left_knee={float(row['left_knee']):.2f}, "
            f"right_knee={float(row['right_knee']):.2f}"
        )


if __name__ == "__main__":
    main()
