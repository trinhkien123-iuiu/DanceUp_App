
import pandas as pd
import matplotlib

# Dùng backend không cần Tkinter
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# Xác định đường dẫn
base_dir = Path(__file__).parent
csv_path = base_dir / "joint_angles_student.csv"
output_path = base_dir / "joint_angles_plot_student.png"

# Kiểm tra file dữ liệu
if not csv_path.exists():
    raise FileNotFoundError(
        f"Khong tim thay file du lieu: {csv_path}"
    )

df = pd.read_csv(csv_path)

joints = [
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
]

# Kiểm tra các cột cần vẽ
missing_columns = [
    joint for joint in joints
    if joint not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thieu cac cot trong CSV: {missing_columns}"
    )

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))

for joint in joints:
    plt.plot(df["frame"], df[joint], label=joint)

plt.title("Joint Angles Over Time")
plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.ylim(0, 180)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Lưu ảnh thay vì mở cửa sổ
plt.savefig(output_path, dpi=150)
plt.close()

print("Da tao bieu do thanh cong!")
print(f"File anh: {output_path}")
