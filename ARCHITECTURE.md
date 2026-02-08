# Motion-Field Covert Communication System
## Complete System Architecture & Design Document

**Version:** 1.0  
**Status:** Design Frozen  
**Last Updated:** <today’s date>

This document represents a frozen system architecture.  
All assumptions, constraints, and module boundaries are finalized.


---

## 1. SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRANSMITTER PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Plaintext Message                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────┐                                                   │
│  │ Crypto Module    │  Argon2 KDF + ChaCha20-Poly1305                   │
│  │ (Module 3)       │  Output: [nonce|ciphertext|tag]                   │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐                                                   │
│  │ ECC Encoder      │  Reed-Solomon / LDPC                              │
│  │ (Module 4)       │  Output: redundant bitstream                      │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐         ┌─────────────────┐                      │
│  │ Motion Field     │◄────────│ Optical Flow    │                      │
│  │ Modulation       │         │ Extraction      │                      │
│  │ (Module 5)       │         │ (Module 2)      │                      │
│  └────────┬─────────┘         └─────────────────┘                      │
│           │                            ▲                                │
│           │                            │                                │
│           ▼                            │                                │
│  ┌──────────────────┐         ┌───────┴─────────┐                      │
│  │ Video Recon      │◄────────│ Video I/O       │                      │
│  │ (Module 6)       │         │ (Module 1)      │                      │
│  └────────┬─────────┘         └─────────────────┘                      │
│           │                                                              │
│           ▼                                                              │
│     Stego Video                                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
                            
┌─────────────────────────────────────────────────────────────────────────┐
│                            CHANNEL                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  • H.264/H.265 compression (CRF 18-28)                                  │
│  • Re-encoding attacks                                                   │
│  • Gaussian noise injection                                              │
│  • Motion estimation errors                                              │
│  • Frame drops (optional)                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                         RECEIVER PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stego Video (possibly corrupted)                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────┐                                                   │
│  │ Video I/O        │  Load & preprocess                                │
│  │ (Module 1)       │                                                    │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐                                                   │
│  │ Motion Field     │  Re-extract optical flow                          │
│  │ Re-Extraction    │                                                    │
│  │ (Module 2)       │                                                    │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐                                                   │
│  │ Motion Field     │  Demodulate embedded bits                         │
│  │ Demodulation     │                                                    │
│  │ (Module 5)       │                                                    │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐                                                   │
│  │ ECC Decoder      │  Error correction                                 │
│  │ (Module 4)       │                                                    │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐                                                   │
│  │ Crypto Module    │  Authenticated decryption                         │
│  │ (Module 3)       │  Verify tag, decrypt                              │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  Plaintext Message (or AUTHENTICATION FAILURE)                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │   ADVERSARIAL SYSTEM     │
                    │      (Parallel)          │
                    └──────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEGANALYSIS ATTACKER                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Video (clean OR stego)                                                  │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────┐                                                   │
│  │ Feature          │  Extract motion statistics:                       │
│  │ Extraction       │  • Histogram features                             │
│  │ (Module 8)       │  • Temporal residuals                             │
│  └────────┬─────────┘  • Motion coherence metrics                       │
│           │            • First-order statistics                         │
│           ▼                                                              │
│  ┌──────────────────┐                                                   │
│  │ CNN Classifier   │  Binary classification                            │
│  │ (Module 8)       │  Output: P(stego | video)                         │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  Detection Decision + Confidence Score                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## 2. FOLDER STRUCTURE

```
motion_covert_comm/
│
├── README.md                          # Project overview
├── ARCHITECTURE.md                    # This document
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── config/                            # Configuration files
│   ├── default_config.yaml           # Default system parameters
│   ├── embedding_params.yaml         # Modulation parameters
│   ├── crypto_params.yaml            # Cryptographic settings
│   └── steganalysis_params.yaml      # Detector configuration
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── module1_video_io/             # MODULE 1: Video I/O
│   │   ├── __init__.py
│   │   ├── video_loader.py           # Load/normalize videos
│   │   ├── video_writer.py           # Write stego videos
│   │   └── preprocessing.py          # Frame normalization
│   │
│   ├── module2_motion_extraction/    # MODULE 2: Optical Flow
│   │   ├── __init__.py
│   │   ├── raft_wrapper.py           # RAFT model wrapper
│   │   ├── flow_extractor.py         # Dense flow extraction
│   │   ├── flow_utils.py             # Flow visualization
│   │   └── models/                   # Pretrained weights
│   │       └── raft-things.pth
│   │
│   ├── module3_crypto/               # MODULE 3: Cryptography
│   │   ├── __init__.py
│   │   ├── key_derivation.py         # Argon2 KDF
│   │   ├── aead_cipher.py            # ChaCha20-Poly1305
│   │   ├── framing.py                # Nonce + tag framing
│   │   └── crypto_utils.py           # Helper functions
│   │
│   ├── module4_ecc/                  # MODULE 4: Error Correction
│   │   ├── __init__.py
│   │   ├── reed_solomon.py           # RS encoder/decoder
│   │   ├── ldpc.py                   # LDPC (optional)
│   │   └── ecc_utils.py              # BER calculations
│   │
│   ├── module5_modulation/           # MODULE 5: Motion Modulation (CORE)
│   │   ├── __init__.py
│   │   ├── qim_modulator.py          # Quantization Index Modulation
│   │   ├── vector_perturbation.py    # Magnitude/direction embedding
│   │   ├── constraint_enforcer.py    # Smoothness + perceptual limits
│   │   ├── demodulator.py            # Bit recovery
│   │   └── embedding_params.py       # ε, payload, thresholds
│   │
│   ├── module6_video_recon/          # MODULE 6: Video Reconstruction
│   │   ├── __init__.py
│   │   ├── frame_warping.py          # Warp frames with modified flow
│   │   ├── temporal_consistency.py   # Enforce coherence
│   │   └── video_builder.py          # Assemble final video
│   │
│   ├── module7_receiver/             # MODULE 7: Receiver (Orchestrator)
│   │   ├── __init__.py
│   │   ├── decoder_pipeline.py       # Full decode pipeline
│   │   └── integrity_check.py        # AEAD verification
│   │
│   ├── module8_steganalysis/         # MODULE 8: Adversarial Detector
│   │   ├── __init__.py
│   │   ├── feature_extractor.py      # Motion statistics
│   │   ├── cnn_detector.py           # Binary classifier
│   │   ├── training.py               # Detector training
│   │   └── evaluation.py             # ROC/AUC computation
│   │
│   ├── math_models/                  # Mathematical modeling
│   │   ├── __init__.py
│   │   ├── channel_model.py          # Y = X + N modeling
│   │   ├── capacity_theory.py        # Shannon capacity bounds
│   │   ├── snr_analysis.py           # SNR computations
│   │   └── information_theory.py     # Entropy, mutual information
│   │
│   └── utils/                        # Common utilities
│       ├── __init__.py
│       ├── metrics.py                # BER, PSNR, SSIM
│       ├── visualization.py          # Plotting helpers
│       ├── logging_config.py         # Structured logging
│       └── config_loader.py          # YAML parser
│
├── experiments/                       # Experimental scripts
│   ├── transmitter.py                # End-to-end sender
│   ├── receiver.py                   # End-to-end receiver
│   ├── channel_simulator.py          # Compression/noise attacks
│   ├── train_detector.py             # Train steganalysis CNN
│   ├── evaluate_system.py            # Full evaluation suite
│   └── ablation_studies.py           # Ablation experiments
│
├── notebooks/                         # Jupyter analysis (optional)
│   ├── flow_visualization.ipynb
│   ├── capacity_analysis.ipynb
│   └── roc_curves.ipynb
│
├── tests/                             # Unit tests
│   ├── test_crypto.py
│   ├── test_ecc.py
│   ├── test_modulation.py
│   └── test_integration.py
│
├── data/                              # Data storage
│   ├── videos/                       # Input videos
│   │   ├── clean/
│   │   └── stego/
│   ├── motion_fields/                # Extracted flow fields
│   ├── datasets/                     # Training datasets for detector
│   └── results/                      # Experiment outputs
│
├── models/                            # Saved models
│   ├── raft/                         # RAFT pretrained weights
│   └── detector/                     # Trained steganalysis models
│
└── docs/                              # Documentation
    ├── mathematical_framework.pdf    # LaTeX-compiled theory
    ├── design_decisions.md           # Rationale for choices
    ├── security_analysis.md          # Threat model
    └── user_guide.md                 # How to run system
```

---

## 3. MODULE SPECIFICATIONS

### MODULE 1: Video I/O

**Responsibility:** Load, preprocess, and write video files

**Input Contracts:**
- Video file path (str)
- Target frame rate (int, default 30)
- Resolution (tuple, optional)

**Output Contracts:**
- `List[np.ndarray]`: Frames as RGB arrays (H, W, 3), dtype=uint8
- Metadata: `dict` with fps, total_frames, resolution

**Key Functions:**
```python
load_video(path: str, fps: int = 30) -> Tuple[List[np.ndarray], dict]
write_video(frames: List[np.ndarray], path: str, fps: int, codec: str = 'libx264')
normalize_frames(frames: List[np.ndarray]) -> List[np.ndarray]
```

**Dependencies:** OpenCV, FFmpeg

**Math Modeling:** None (pure I/O)

---

### MODULE 2: Motion Field Extraction

**Responsibility:** Extract dense optical flow using RAFT

**Input Contracts:**
- Frame pair: `(frame_t, frame_{t+1})` as `np.ndarray` (H, W, 3)
- RAFT model: pretrained weights path

**Output Contracts:**
- Flow field: `np.ndarray` (H, W, 2), dtype=float32
  - `flow[:,:,0]`: horizontal displacement (dx)
  - `flow[:,:,1]`: vertical displacement (dy)

**Key Functions:**
```python
load_raft_model(weights_path: str) -> nn.Module
extract_flow(frame1: np.ndarray, frame2: np.ndarray, model) -> np.ndarray
visualize_flow(flow: np.ndarray) -> np.ndarray  # HSV colormap
compute_flow_statistics(flow: np.ndarray) -> dict
```

**Dependencies:** PyTorch, RAFT (from official repo)

**Math Modeling:**
- Flow estimation as optimization: `min ||I(x) - I(x + f)||²`
- Statistical properties: mean magnitude, directional entropy

---

### MODULE 3: Cryptographic Pipeline

**Responsibility:** Secure message encryption/decryption with AEAD

**Input Contracts:**
- Plaintext: `bytes`
- Password: `str` (user-provided)
- Salt: `bytes` (16 bytes, generated or provided)

**Output Contracts:**
- Ciphertext frame: `bytes`
  - Structure: `[nonce(12) | ciphertext(variable) | tag(16)]`
- Total length: `len(plaintext) + 28` bytes

**Key Functions:**
```python
derive_key(password: str, salt: bytes, memory_cost: int = 65536) -> bytes
encrypt_message(plaintext: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]  # (nonce, ciphertext, tag)
decrypt_message(ciphertext: bytes, nonce: bytes, tag: bytes, key: bytes) -> bytes
frame_aead(nonce: bytes, ciphertext: bytes, tag: bytes) -> bytes
unframe_aead(frame: bytes) -> Tuple[bytes, bytes, bytes]
```

**Dependencies:** `cryptography` (Argon2, ChaCha20Poly1305)

**Math Modeling:**
- AEAD security: IND-CCA2 + INT-CTXT
- Key derivation: Argon2id parameter selection
- Entropy requirements: nonce uniqueness

---

### MODULE 4: Error Correction Coding

**Responsibility:** Add redundancy to protect against channel errors

**Input Contracts:**
- Data bitstream: `bytes` or `List[int]` (bits)
- ECC parameters: `(n, k)` for RS or parity-check matrix for LDPC

**Output Contracts:**
- Encoded bitstream: `bytes` with redundancy
- Code rate: `k/n` (e.g., 0.8 for 20% redundancy)

**Key Functions:**
```python
rs_encode(data: bytes, nsym: int = 32) -> bytes  # nsym = parity symbols
rs_decode(encoded: bytes, nsym: int = 32) -> Tuple[bytes, int]  # (data, n_errors_corrected)
compute_ber(original: bytes, received: bytes) -> float
estimate_correction_capacity(snr_db: float, code_rate: float) -> float
```

**Dependencies:** `reedsolo`

**Math Modeling:**
- Shannon capacity: `C = B log₂(1 + SNR)`
- Code rate vs correction capability tradeoff
- BER before/after ECC decoding

---

### MODULE 5: Motion-Field Modulation (CORE MODULE)

**Responsibility:** Embed/extract bits into motion vectors

**Payload Allocation:**  
Payload is allocated adaptively on a per-frame basis using motion capacity
estimation. Frames with insufficient motion skip embedding entirely.
No fixed bits-per-frame assumption is made at implementation time.


**Input Contracts:**
- Flow field: `np.ndarray` (H, W, 2)
- Bitstream: `bytes`
- Embedding parameters:
  - `epsilon`: perturbation strength (float, e.g., 0.5 pixels)
  - `quantization_step`: QIM step size (float, e.g., 2.0)
  - `payload_capacity`: adaptive, determined per-frame via motion capacity estimation


**Output Contracts:**
- Modified flow: `np.ndarray` (H, W, 2)
- Embedding map: `np.ndarray` (H, W), dtype=bool (which vectors modified)
- Actual bits embedded: `int`

**Embedding Strategy (QIM):**
```
For each motion vector v = (dx, dy):
1. Compute magnitude: m = sqrt(dx² + dy²)
2. Quantize: q = round(m / Δ)
3. Embed bit b:
   - If b=0: m' = q * Δ
   - If b=1: m' = (q + 0.5) * Δ
4. Scale vector: v' = v * (m' / m)
5. Apply constraints: ||v' - v|| ≤ ε
```

**Constraints:**
- Perceptual: `||v' - v||_∞ ≤ ε_max`
- Smoothness: neighboring vectors differ by ≤ threshold
- Magnitude preservation: `0.8m ≤ m' ≤ 1.2m`

**Key Functions:**
```python
embed_bits_qim(flow: np.ndarray, bits: bytes, params: dict) -> Tuple[np.ndarray, np.ndarray]
extract_bits_qim(flow: np.ndarray, num_bits: int, params: dict) -> bytes
enforce_constraints(original_flow: np.ndarray, modified_flow: np.ndarray, epsilon: float) -> np.ndarray
compute_embedding_distortion(original: np.ndarray, modified: np.ndarray) -> float
```

**Dependencies:** NumPy, SciPy (for spatial filtering)

**Math Modeling:**
- **Channel model:** `Y = X + N`, where:
  - `X`: embedded motion signal
  - `N`: compression noise (Gaussian approximation)
  - `Y`: received signal
- **SNR:** `10 log₁₀(σ_X² / σ_N²)`
- **Capacity:** `C ≈ (H × W) × log₂(1 + SNR) / frame_count`

---

### MODULE 6: Video Reconstruction

**Responsibility:** Generate stego video from modified motion fields

**Clarification:**  
Video reconstruction is treated as motion-compensated frame warping with blending.  
This module does **not** perform full frame synthesis.  
The encoder assumes access to original frames, and quality evaluation measures
degradation due to motion-field perturbations, not generative reconstruction accuracy.


**Input Contracts:**
- Original frames: `List[np.ndarray]`
- Modified flows: `List[np.ndarray]` (one per frame pair)
- Reconstruction method: `'warp'` or `'blend'`

**Output Contracts:**
- Reconstructed frames: `List[np.ndarray]`
- Quality metrics: `dict` with PSNR, SSIM

**Reconstruction Process:**
```
1. For each frame pair (I_t, I_{t+1}) and flow F':
   a. Warp I_t using F': I_t_warp = warp(I_t, F')
   b. Blend: I'_{t+1} = α * I_{t+1} + (1-α) * I_t_warp
   c. Apply temporal consistency filter
```

**Key Functions:**
```python
warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray
reconstruct_video(frames: List[np.ndarray], flows: List[np.ndarray]) -> List[np.ndarray]
compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float
compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float
```

**Dependencies:** OpenCV (remap), scikit-image (SSIM)

**Math Modeling:**
- PSNR: `10 log₁₀(255² / MSE)`
- SSIM: structural similarity index

---

### MODULE 7: Receiver Pipeline

**Responsibility:** Orchestrate full decoding process

**Input Contracts:**
- Stego video path: `str`
- Decryption password: `str`
- ECC parameters: `dict`
- Modulation parameters: `dict`

**Output Contracts:**
- Decrypted plaintext: `bytes` (on success)
- Error report: `dict` with BER, authentication status

**Pipeline:**
```
1. Load stego video (Module 1)
2. Extract motion fields (Module 2)
3. Demodulate bitstream (Module 5)
4. Apply ECC decoding (Module 4)
5. Unframe AEAD structure (Module 3)
6. Verify authentication tag
7. Decrypt ciphertext (Module 3)
8. Return plaintext or raise AuthenticationError
```

**Key Functions:**
```python
decode_video(video_path: str, password: str, config: dict) -> bytes
verify_integrity(received_bits: bytes, expected_tag: bytes) -> bool
```

**Dependencies:** All previous modules

**Math Modeling:** None (orchestration only)

---

### MODULE 8: Steganalysis Attacker

**Responsibility:** Detect presence of hidden data

**Input Contracts:**
- Video (clean or stego): `str` or `List[np.ndarray]`
- Detector model: trained CNN weights

**Output Contracts:**
- Detection probability: `float` ∈ [0, 1]
- Class label: `{0: clean, 1: stego}`
- Confidence score

**Architecture:**
```
Input: Motion statistics features (512-D vector per frame)
│
├─ Temporal residual features (256-D)
├─ Histogram features (128-D)
├─ Co-occurrence matrix features (128-D)
│
▼
3-layer CNN (Conv1D) + Global pooling
│
▼
Dense layers [256, 128, 64]
│
▼
Sigmoid output → P(stego)
```

**Feature Extraction:**
- Motion magnitude histogram (32 bins)
- Directional histogram (16 bins)
- Temporal motion difference statistics
- Local coherence metrics
- First-order statistics (mean, variance, skewness, kurtosis)

**Key Functions:**
```python
extract_features(video: List[np.ndarray]) -> np.ndarray  # (T, 512)
train_detector(clean_videos: List, stego_videos: List, epochs: int = 50)
predict(video: str, model) -> Tuple[float, int]
compute_roc(predictions: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]
```

**Dependencies:** PyTorch, scikit-learn

**Math Modeling:**
- Binary classification loss: Binary Cross-Entropy
- ROC curve: TPR vs FPR at varying thresholds
- AUC: Area under ROC curve
- Detection accuracy at fixed false positive rate (e.g., 1%)

---

## 4. DATA FLOW CONTRACTS

### Transmitter Data Flow

```
Message (bytes)
    ↓
[Module 3: Crypto] → Encrypted frame (bytes: nonce + ciphertext + tag)
    ↓
[Module 4: ECC] → Redundant bitstream (bytes)
    ↓
[Module 1: Video I/O] → Original frames (List[np.ndarray])
    ↓
[Module 2: Flow] → Motion fields (List[np.ndarray], shape (H, W, 2))
    ↓
[Module 5: Modulation] → Modified motion fields (List[np.ndarray])
    ↓
[Module 6: Reconstruction] → Stego frames (List[np.ndarray])
    ↓
[Module 1: Video I/O] → Stego video file (H.264)
```

### Channel Attack

```
Stego video (clean)
    ↓
[FFmpeg compression] → Re-encoded video (CRF 23)
    ↓
[Noise injection] → Noisy video (Gaussian σ=5)
    ↓
Channel output
```

### Receiver Data Flow

```
Stego video (possibly corrupted)
    ↓
[Module 1: Video I/O] → Frames (List[np.ndarray])
    ↓
[Module 2: Flow] → Motion fields (re-extracted)
    ↓
[Module 5: Demodulation] → Noisy bitstream (bytes)
    ↓
[Module 4: ECC Decode] → Corrected encrypted frame (bytes)
    ↓
[Module 3: Crypto] → Authenticated decryption
    ↓
Plaintext (bytes) OR AuthenticationError
```

### Steganalysis Data Flow

```
Video (unknown)
    ↓
[Module 1: Video I/O] → Frames
    ↓
[Module 2: Flow] → Motion fields
    ↓
[Module 8: Feature Extraction] → Feature vectors (T, 512)
    ↓
[Module 8: CNN] → P(stego | video)
    ↓
Detection decision (binary) + confidence
```

---

## 5. MATHEMATICAL FRAMEWORK INTEGRATION

### Where Math Fits vs Code

**Pure Math (LaTeX documents in `docs/`):**
- Theoretical capacity bounds (Shannon limit)
- AEAD security proofs (references)
- QIM theoretical analysis
- Information-theoretic limits

**Math + Code (in `src/math_models/`):**
- **Channel model:** Implement `Y = X + N` simulator
- **SNR analysis:** Compute empirical SNR from data
- **Capacity estimation:** Numerical integration of capacity formula
- **BER vs SNR curves:** Monte Carlo simulations

**Code-only (in module implementations):**
- Actual embedding/extraction algorithms
- Video processing pipelines
- Network training

### Key Mathematical Relationships

**1. Channel Capacity**
```
C = B × log₂(1 + SNR)  [bits/second]

where:
  B = spatial bandwidth = (H × W × fps) / N_frames
  SNR = E[||X||²] / E[||N||²]
```

**Implementation:** `src/math_models/capacity_theory.py`
- Function: `compute_shannon_capacity(snr_db, bandwidth)`
- Function: `estimate_practical_capacity(flows_clean, flows_compressed)`

**2. Embedding Distortion**
```
D = (1 / N) Σ ||v_original - v_modified||²

where N = total motion vectors
```

**Implementation:** `src/module5_modulation/qim_modulator.py`
- Method: `compute_embedding_distortion(flow_orig, flow_mod)`

**3. Bit Error Rate**
```
BER = (# bit errors) / (# total bits)

Related to SNR via:
  BER ≈ Q(sqrt(2 × SNR))  [for BPSK in AWGN]
```

**Implementation:** `src/module4_ecc/ecc_utils.py`
- Function: `compute_ber(sent_bits, received_bits)`

**4. Detection Metrics**
```
TPR = TP / (TP + FN)  [True Positive Rate]
FPR = FP / (FP + TN)  [False Positive Rate]
AUC = ∫ TPR(t) d(FPR(t))
```

**Implementation:** `src/module8_steganalysis/evaluation.py`
- Function: `compute_roc_auc(y_true, y_scores)`

---

## 6. CONFIGURATION MANAGEMENT

### `config/default_config.yaml`

```yaml
system:
  version: "1.0.0"
  random_seed: 42

video:
  fps: 30
  resolution: [640, 480]
  codec: "libx264"
  input_format: "mp4"

optical_flow:
  model: "raft"
  weights_path: "models/raft/raft-things.pth"
  iters: 20  # RAFT iterations

crypto:
  kdf: "argon2id"
  kdf_memory: 65536  # 64 MB
  kdf_iterations: 3
  kdf_parallelism: 4
  cipher: "chacha20-poly1305"
  nonce_size: 12
  tag_size: 16

ecc:
  type: "reed_solomon"
  nsym: 32  # parity symbols (20% redundancy for 255-byte blocks)
  
modulation:
  method: "qim"  # or "vector_perturbation"
  epsilon: 0.5  # max perturbation in pixels
  quantization_step: 2.0
  payload_bpf: 256  # bits per frame
  
channel:
  compression_crf: 23  # H.264 CRF (18=high quality, 28=visible loss)
  noise_sigma: 0.0  # Gaussian noise std dev
  
steganalysis:
  feature_dim: 512
  batch_size: 16
  learning_rate: 0.0001
  epochs: 50
```

---

## 7. INTERFACE CONTRACTS SUMMARY

### Critical Type Definitions

```python
# Core data structures
Frame = np.ndarray  # (H, W, 3), dtype=uint8, RGB
FlowField = np.ndarray  # (H, W, 2), dtype=float32
Bitstream = bytes
EncryptedFrame = bytes  # [nonce | ciphertext | tag]

# Module interfaces
class VideoIO:
    def load(path: str) -> Tuple[List[Frame], Dict]
    def save(frames: List[Frame], path: str, fps: int)

class FlowExtractor:
    def extract(frame1: Frame, frame2: Frame) -> FlowField
    def batch_extract(frames: List[Frame]) -> List[FlowField]

class CryptoEngine:
    def encrypt(plaintext: bytes, password: str) -> EncryptedFrame
    def decrypt(ciphertext: EncryptedFrame, password: str) -> bytes

class ECCCodec:
    def encode(data: bytes) -> bytes
    def decode(encoded: bytes) -> Tuple[bytes, int]

class MotionModulator:
    def embed(flow: FlowField, bits: bytes, params: dict) -> FlowField
    def extract(flow: FlowField, params: dict) -> bytes

class VideoReconstructor:
    def reconstruct(frames: List[Frame], flows: List[FlowField]) -> List[Frame]

class Steganalyzer:
    def extract_features(video: List[Frame]) -> np.ndarray
    def predict(features: np.ndarray) -> Tuple[float, int]
```

---

## 8. EXECUTION FLOW SUMMARY

### End-to-End Transmitter

```python
# Pseudocode
def transmit(cover_video_path, message, password, config):
    # 1. Load video
    frames, metadata = load_video(cover_video_path)
    
    # 2. Encrypt message
    encrypted = encrypt_message(message, password)
    
    # 3. Add error correction
    encoded_bits = ecc_encode(encrypted)
    
    # 4. Extract motion fields
    flows = [extract_flow(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    
    # 5. Embed bits into flows
    stego_flows = [embed_bits(flow, bits_chunk, config) for flow, bits_chunk in zip(flows, chunk_bits(encoded_bits))]
    
    # 6. Reconstruct video
    stego_frames = reconstruct_video(frames, stego_flows)
    
    # 7. Write output
    write_video(stego_frames, "stego.mp4", metadata['fps'])
```

### End-to-End Receiver

```python
def receive(stego_video_path, password, config):
    # 1. Load stego video
    frames, _ = load_video(stego_video_path)
    
    # 2. Re-extract motion
    flows = [extract_flow(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    
    # 3. Demodulate bits
    received_bits = b''.join([extract_bits(flow, config) for flow in flows])
    
    # 4. ECC decode
    corrected_bits, n_errors = ecc_decode(received_bits)
    
    # 5. Decrypt
    try:
        plaintext = decrypt_message(corrected_bits, password)
        return plaintext, n_errors
    except AuthenticationError:
        return None, -1  # Integrity failure
```

---

## 9. DEPENDENCY GRAPH

```
Module 8 (Steganalysis)
    ↓ depends on
Module 2 (Flow Extraction)
    ↓
Module 1 (Video I/O)

Module 7 (Receiver)
    ↓ depends on
Module 5 (Modulation) + Module 4 (ECC) + Module 3 (Crypto) + Module 2 + Module 1

Module 6 (Reconstruction)
    ↓ depends on
Module 2 (Flow) + Module 1 (Video I/O)

Module 5 (Modulation) — CORE, depends on numpy/scipy only

Module 4 (ECC) — Independent

Module 3 (Crypto) — Independent

Module 2 (Flow) 
    ↓ depends on
Module 1 (Video I/O) + RAFT model

Module 1 (Video I/O) — Independent (base)
```

**Build Order:**
1. Module 1 (Video I/O)
2. Module 2 (Flow Extraction)
3. Module 3 (Crypto) [parallel to 2]
4. Module 4 (ECC) [parallel to 2-3]
5. Module 5 (Modulation)
6. Module 6 (Reconstruction)
7. Module 7 (Receiver)
8. Module 8 (Steganalysis)

---

## 10. TESTING STRATEGY

### Unit Tests (per module)

```python
# tests/test_crypto.py
def test_encryption_decryption_roundtrip()
def test_authentication_failure_on_tamper()
def test_key_derivation_deterministic()

# tests/test_ecc.py
def test_error_correction_under_noise()
def test_ber_calculation()

# tests/test_modulation.py
def test_qim_embedding_extraction()
def test_constraint_enforcement()
def test_capacity_calculation()
```

### Integration Tests

```python
# tests/test_integration.py
def test_full_transmitter_receiver_pipeline()
def test_robustness_to_compression()
def test_detector_training_convergence()
```

---

## 11. EVALUATION METRICS (Final Deliverables)

### Quantitative Outputs Required

1. **Payload Capacity**
   - Bits per frame (bpf)
   - Total bits per video
   - Effective bitrate (bps)

2. **Reliability**
   - BER before ECC
   - BER after ECC
   - Authentication success rate

3. **Robustness**
   - Payload vs CRF (compression)
   - Payload vs noise level
   - BER vs SNR curve

4. **Security**
   - Detection accuracy (ROC/AUC)
   - TPR at 1% FPR
   - Embedding distortion (MSE)

5. **Quality**
   - PSNR (cover vs stego)
   - SSIM (cover vs stego)
   - Visual artifacts (subjective)

### Required Plots

```python
1. Payload (bpf) vs BER
2. Payload vs Detection Probability
3. ROC curves (TPR vs FPR)
4. Capacity tradeoff (payload vs quality vs security)
5. Ablation: With/without ECC
6. Compression robustness (CRF vs BER)
```

---

## 12. OPEN RESEARCH QUESTIONS (To Address)

1. **Optimal QIM quantization step:** How to set Δ for max capacity under compression?
2. **Flow estimation error impact:** How does RAFT error propagate to BER?
3. **Motion coherence constraint:** What spatial smoothness filter minimizes detectability?
4. **Detector feature engineering:** Which motion statistics best reveal embedding?
5. **Capacity limits:** What is the practical Shannon limit for this channel?

---

## 13. SECURITY THREAT MODEL

### Adversary Capabilities (Assumed)

**Passive Attacker:**
- Observes stego videos
- Knows embedding is in motion fields (Kerckhoffs)
- Has access to clean videos for comparison
- Can train steganalysis detectors

**Active Attacker:**
- Can compress/re-encode videos
- Can add noise
- Cannot break ChaCha20-Poly1305 (assumed)
- Cannot forge authentication tags

**Out of Scope:**
- Side-channel attacks (timing, power)
- Cryptanalysis of Argon2/ChaCha20
- Physical access to encoder

### Security Properties (Goals)

1. **Confidentiality:** Message unreadable without key
2. **Integrity:** Tampering detectable via AEAD tag
3. **Undetectability:** AUC ≤ 0.65 for trained detector
4. **Robustness:** Message recoverable after mild compression

---

## 14. LIMITATIONS & FUTURE WORK

### Known Limitations

1. **Capacity:** Limited by motion field complexity (static scenes fail)
2. **Detectability:** High payload → higher detection risk
3. **Compression:** Lossy compression degrades motion fields
4. **Computational cost:** RAFT inference is slow (~0.5 fps on RTX 3090)

### Future Improvements

1. Adaptive payload based on scene motion
2. Learned modulation (train encoder/decoder jointly)
3. Multi-frame temporal embedding
4. Hybrid spatial–temporal motion embedding


---

---

## 15. SYSTEM VALIDATION STATUS

The architecture described in this document has been fully implemented
and validated through unit tests, integration tests, and controlled
experimental evaluation.

Validation includes:
- Deterministic end-to-end transmitter → receiver recovery
- Synthetic motion roundtrip experiments with zero bit error
- Real-video motion capacity profiling
- ECC-protected payload recovery under sufficient motion
- Verified module isolation and interface contracts

All module boundaries, data contracts, and assumptions described above
are consistent with the final implementation.

**Architecture Status:** Frozen & Experimentally Verified

