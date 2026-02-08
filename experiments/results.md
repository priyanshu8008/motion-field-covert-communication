# Experimental Results

This document summarizes the empirical evaluation of the
Motion-Field Covert Communication System.

---

## 1. Experimental Setup

- Optical Flow: RAFT (Things pretrained)
- Modulation: QIM on motion magnitude
- ECC: Reed-Solomon
- Video Codec: H.264
- Execution Device: CPU (no GPU acceleration)
- Frame Resolution: Downscaled where necessary

---

## 2. Experiment 1: Capacity vs Motion Magnitude (Synthetic)

**Goal:**  
Measure embedding capacity as a function of motion strength.

**Setup:**
- Synthetic flow fields of size 64×64
- Uniform horizontal motion
- Motion magnitude varied

**Results:**

| Motion Magnitude | Capacity (bits/frame) |
|-----------------|-----------------------|
| 0.5             | 0                     |
| 1.0             | 4096                  |
| 2.0             | 4096                  |
| 5.0             | 4096                  |
| 10.0            | 4096                  |

**Conclusion:**  
There exists a minimum motion threshold below which embedding is impossible.
Above this threshold, capacity saturates.

---

## 3. Experiment 2: Synthetic End-to-End Roundtrip

**Goal:**  
Verify lossless recovery under ideal motion conditions.

**Payload:** `HELLO_COVERT`

**Results:**
- Bits embedded: 2040
- Bits extracted: 2040
- BER: 0.0
- ECC decode: SUCCESS

**Conclusion:**  
The modulation + demodulation pipeline is functionally correct.

---

## 4. Experiment 3: Real Video Capacity Profiling

**Goal:**  
Measure usable capacity across real video frames.

**Method:**
- Extract optical flow from consecutive frames
- Compute per-frame capacity
- Skip low-motion frames

**Output:**
- CSV: `results_real_video_capacity.csv`
- Plot: `fig_capacity_vs_frame.png`

**Conclusion:**  
Capacity varies significantly over time and is scene-dependent.
Adaptive embedding is mandatory.

---

## 5. Key Findings

1. Motion magnitude directly governs payload capacity
2. Static scenes provide zero capacity
3. QIM embedding is perfectly reversible under ideal conditions
4. Real videos require frame-level adaptivity

---

## 6. Limitations Observed

- RAFT is memory-intensive on CPU
- Capacity is highly non-uniform
- No robustness testing under compression yet

---

## 7. Overall Assessment

The system functions as designed and validates the proposed architecture.
All failures observed are expected and align with theoretical assumptions.
