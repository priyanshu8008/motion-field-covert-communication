import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("experiments/results_real_video_capacity.csv")
OUT_PATH = Path("experiments/fig_capacity_vs_frame.png")

df = pd.read_csv(CSV_PATH)

print("CSV columns:", list(df.columns))

# --------------------------------------------------
# Auto-detect column names
# --------------------------------------------------
if "frame_idx" in df.columns:
    x = df["frame_idx"]
elif "frame" in df.columns:
    x = df["frame"]
elif "flow_index" in df.columns:
    x = df["flow_index"]
else:
    x = range(len(df))

if "capacity_bits" in df.columns:
    y = df["capacity_bits"]
elif "capacity" in df.columns:
    y = df["capacity"]
else:
    raise ValueError("No capacity column found")

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(x, y, marker="o")
plt.xlabel("Frame Index")
plt.ylabel("Embedding Capacity (bits)")
plt.title("Per-Frame Motion-Based Embedding Capacity")
plt.grid(True)
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=200)
plt.show()

print(f"Saved plot → {OUT_PATH}")
