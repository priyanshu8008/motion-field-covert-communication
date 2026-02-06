# MODULE INTERFACE SPECIFICATIONS
## Motion-Field Covert Communication System

This document defines precise input/output contracts for all modules.

---

## TYPE DEFINITIONS

```python
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CORE DATA TYPES
# ============================================================================

# Video frame: RGB image
Frame = np.ndarray  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]

# Optical flow field: dense motion vectors
FlowField = np.ndarray  # Shape: (H, W, 2), dtype: float32
                        # [:,:,0] = dx (horizontal displacement)
                        # [:,:,1] = dy (vertical displacement)

# Binary data
Bitstream = bytes  # Arbitrary length binary data

# Encrypted message with authentication
@dataclass
class EncryptedFrame:
    nonce: bytes  # 12 bytes
    ciphertext: bytes  # Variable length
    tag: bytes  # 16 bytes
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes: [nonce|ciphertext|tag]"""
        return self.nonce + self.ciphertext + self.tag
    
    @staticmethod
    def from_bytes(data: bytes) -> 'EncryptedFrame':
        """Deserialize from bytes"""
        return EncryptedFrame(
            nonce=data[:12],
            ciphertext=data[12:-16],
            tag=data[-16:]
        )

# Feature vector for steganalysis
FeatureVector = np.ndarray  # Shape: (512,), dtype: float32

# Embedding location map
EmbeddingMap = np.ndarray  # Shape: (H, W), dtype: bool

# ============================================================================
# METADATA STRUCTURES
# ============================================================================

@dataclass
class VideoMetadata:
    """Metadata for loaded video"""
    fps: int
    width: int
    height: int
    num_frames: int
    duration: float  # seconds
    codec: str
    pixel_format: str

@dataclass
class EmbeddingMetadata:
    """Metadata about embedding operation"""
    bits_embedded: int
    bits_requested: int
    embedding_rate: float  # bits embedded / bits requested
    num_vectors_modified: int
    total_vectors_available: int
    avg_perturbation: float  # Average ||v' - v||
    max_perturbation: float
    
@dataclass
class DecodingMetadata:
    """Metadata about decoding operation"""
    bits_received: int
    ber_before_ecc: float
    ber_after_ecc: float
    errors_corrected: int
    authentication_passed: bool
    
@dataclass
class QualityMetrics:
    """Video quality metrics"""
    psnr: float  # Peak Signal-to-Noise Ratio (dB)
    ssim: float  # Structural Similarity Index
    mse: float   # Mean Squared Error

# ============================================================================
# CONFIGURATION STRUCTURES
# ============================================================================

@dataclass
class ModulationConfig:
    """Configuration for motion-field modulation"""
    epsilon: float  # Max perturbation (pixels)
    quantization_step: float  # QIM step size
    max_payload_bits: int  # Upper bound, not fixed allocation
    motion_threshold: float  # Min motion magnitude to embed
    enforce_smoothness: bool
    smoothness_kernel_size: int

    
@dataclass
class CryptoConfig:
    """Configuration for cryptographic operations"""
    kdf_memory: int  # Argon2 memory cost (KB)
    kdf_time: int    # Argon2 time cost 
    kdf_parallelism: int
    salt_length: int
    key_length: int
    nonce_size: int
    tag_size: int
    
@dataclass
class ECCConfig:
    """Configuration for error correction"""
    ecc_type: str  # "reed_solomon" or "ldpc"
    n: int  # Codeword length
    k: int  # Message length
    nsym: int  # Parity symbols

# ============================================================================
# RESULT STRUCTURES
# ============================================================================

@dataclass
class TransmitterResult:
    """Result from transmitter pipeline"""
    stego_video_path: str
    embedding_metadata: EmbeddingMetadata
    quality_metrics: QualityMetrics
    execution_time: float
    
@dataclass
class ReceiverResult:
    """Result from receiver pipeline"""
    plaintext: Optional[bytes]  # None if decryption failed
    decoding_metadata: DecodingMetadata
    execution_time: float
    error_message: Optional[str]
    
@dataclass
class SteganalysisResult:
    """Result from steganalysis detector"""
    prediction: int  # 0=clean, 1=stego
    confidence: float  # P(stego | video)
    features: FeatureVector
    
@dataclass
class EvaluationResult:
    """Complete system evaluation"""
    capacity_bps: float  # Bits per second
    ber_before_ecc: float
    ber_after_ecc: float
    authentication_success_rate: float
    detection_auc: float
    detection_accuracy: float
    avg_psnr: float
    avg_ssim: float
```

---

## MODULE 1: VIDEO I/O

### Interface

```python
class VideoIO:
    """Video input/output operations"""
    
    def load_video(
        self, 
        path: str,
        fps: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[Frame], VideoMetadata]:
        """
        Load video from file.
        
        Args:
            path: Path to video file
            fps: Target frame rate (None = keep original)
            resolution: Target (width, height) (None = keep original)
            
        Returns:
            frames: List of RGB frames
            metadata: Video metadata
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format unsupported
        """
        pass
    
    def write_video(
        self,
        frames: List[Frame],
        path: str,
        fps: int,
        codec: str = "libx264",
        crf: int = 18
    ) -> None:
        """
        Write frames to video file.
        
        Args:
            frames: List of RGB frames
            path: Output video path
            fps: Frame rate
            codec: Video codec
            crf: Constant rate factor (quality)
            
        Raises:
            ValueError: If frames list is empty or inconsistent shapes
            IOError: If write fails
        """
        pass
    
    def normalize_frames(
        self,
        frames: List[Frame]
    ) -> List[Frame]:
        """
        Normalize frames to consistent size/format.
        
        Args:
            frames: List of frames (possibly different sizes)
            
        Returns:
            normalized_frames: All same size, RGB, uint8
        """
        pass
```

### Data Flow
- **Input**: Video file path (str)
- **Output**: List[Frame] + VideoMetadata
- **Side Effects**: None (pure I/O)

---

## MODULE 2: OPTICAL FLOW EXTRACTION

### Interface

```python
class OpticalFlowExtractor:
    """Dense optical flow extraction using RAFT"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        """
        Initialize RAFT model.
        
        Args:
            model_path: Path to RAFT pretrained weights
            device: "cuda" or "cpu"
        """
        pass
    
    def extract_flow(
        self,
        frame1: Frame,
        frame2: Frame
    ) -> FlowField:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: First frame (H, W, 3)
            frame2: Second frame (H, W, 3)
            
        Returns:
            flow: Flow field (H, W, 2)
                  flow[:,:,0] = horizontal displacement
                  flow[:,:,1] = vertical displacement
                  
        Raises:
            ValueError: If frames have different shapes
        """
        pass
    
    def batch_extract(
        self,
        frames: List[Frame]
    ) -> List[FlowField]:
        """
        Extract flow for all consecutive frame pairs.
        
        Args:
            frames: List of N frames
            
        Returns:
            flows: List of (N-1) flow fields
        """
        pass
    
    def visualize_flow(
        self,
        flow: FlowField
    ) -> Frame:
        """
        Visualize flow as HSV color-coded image.
        
        Args:
            flow: Flow field
            
        Returns:
            visualization: RGB image (H, W, 3)
        """
        pass
    
    def compute_statistics(
        self,
        flow: FlowField
    ) -> Dict[str, float]:
        """
        Compute flow statistics.
        
        Returns:
            stats: {
                'mean_magnitude': float,
                'max_magnitude': float,
                'directional_entropy': float,
                ...
            }
        """
        pass
```

### Data Flow
- **Input**: Pair of frames (Frame, Frame)
- **Output**: FlowField (H, W, 2)
- **Dependencies**: RAFT pretrained model

---

## MODULE 3: CRYPTOGRAPHIC PIPELINE

### Interface

```python
class CryptographicEngine:
    """Encryption and authentication"""
    
    def __init__(self, config: CryptoConfig):
        """Initialize with crypto parameters"""
        pass
    
    def derive_key(
        self,
        password: str,
        salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password.
        
        Args:
            password: User password
            salt: Salt for KDF (None = generate new)
            
        Returns:
            key: 32-byte encryption key
            salt: Salt used (for storage)
        """
        pass
    
    def encrypt(
        self,
        plaintext: bytes,
        password: str
    ) -> EncryptedFrame:
        """
        Encrypt message with AEAD.
        
        Args:
            plaintext: Message to encrypt
            password: Encryption password
            
        Returns:
            encrypted: EncryptedFrame with nonce, ciphertext, tag
        """
        pass
    
    def decrypt(
        self,
        encrypted: EncryptedFrame,
        password: str
    ) -> bytes:
        """
        Decrypt and verify message.
        
        Args:
            encrypted: EncryptedFrame structure
            password: Decryption password
            
        Returns:
            plaintext: Decrypted message
            
        Raises:
            AuthenticationError: If tag verification fails
            ValueError: If password incorrect
        """
        pass
```

### Data Flow
- **Input**: Plaintext (bytes) + Password (str)
- **Output**: EncryptedFrame (nonce + ciphertext + tag)
- **Security**: Argon2id + ChaCha20-Poly1305

---

## MODULE 4: ERROR CORRECTION CODING

### Interface

```python
class ECCCodec:
    """Error correction encoding/decoding"""
    
    def __init__(self, config: ECCConfig):
        """Initialize with ECC parameters"""
        pass
    
    def encode(
        self,
        data: bytes
    ) -> bytes:
        """
        Add error correction redundancy.
        
        Args:
            data: Input data
            
        Returns:
            encoded: Data with parity symbols
                     Length: ceil(len(data)/k) * n
        """
        pass
    
    def decode(
        self,
        encoded: bytes
    ) -> Tuple[bytes, int]:
        """
        Decode and correct errors.
        
        Args:
            encoded: Received data (possibly corrupted)
            
        Returns:
            decoded: Corrected data
            n_errors: Number of errors corrected
            
        Raises:
            ECCDecodeError: If too many errors to correct
        """
        pass
    
    def compute_ber(
        self,
        sent: bytes,
        received: bytes
    ) -> float:
        """
        Compute bit error rate.
        
        Returns:
            ber: Bit error rate in [0, 1]
        """
        pass
```

### Data Flow
- **Input**: Data (bytes)
- **Output**: Encoded data (bytes) with redundancy
- **Code Rate**: k/n (e.g., 0.875 for RS(255, 223))

---

## MODULE 5: MOTION-FIELD MODULATION (CORE)

### Interface

```python
class MotionFieldModulator:
    """Embed/extract bits in motion fields"""
    
    def __init__(self, config: ModulationConfig):
        """Initialize with modulation parameters"""
        pass
    
    def embed(
        self,
        flow: FlowField,
        bits: bytes
    ) -> Tuple[FlowField, EmbeddingMap, EmbeddingMetadata]:
        """
        Embed bits into flow field using QIM.
        
        Args:
            flow: Original flow field (H, W, 2)
            bits: Bitstream to embed (embedding stops when capacity is exhausted)
            
        Returns:
            modified_flow: Flow with embedded bits
            embedding_map: Boolean mask of modified vectors
            metadata: Embedding statistics
            
        Raises:
            InsufficientCapacityError: If flow has zero usable capacity
        """
        pass
    
    def extract(
        self,
        flow: FlowField,
        num_bits: int
    ) -> bytes:
        """
        Extract bits from flow field.
        
        Args:
            flow: Flow field with embedded data
            num_bits: Number of bits to extract
            embedding_map: Locations to extract from (None = detect)
            
        Returns:
            bits: Extracted bitstream

        Note:
            The receiver does not assume access to the original embedding map.
            Bit extraction is performed blindly using modulation statistics.
        """
        pass
    
    def compute_capacity(
        self,
        flow: FlowField
    ) -> int:
        """
        Estimate embedding capacity.
        
        Returns:
            capacity: Maximum bits that can be embedded
        """
        pass
    
    def enforce_constraints(
        self,
        original: FlowField,
        modified: FlowField
    ) -> FlowField:
        """
        Apply perceptual and smoothness constraints.
        
        Returns:
            constrained: Modified flow within limits
        """
        pass
```

### QIM Algorithm

```python
def qim_embed_vector(
    v: np.ndarray,  # (2,) motion vector
    bit: int,       # 0 or 1
    delta: float    # Quantization step
) -> np.ndarray:
    """
    Embed one bit into motion vector.
    
    Algorithm:
        1. m = ||v|| = sqrt(v[0]^2 + v[1]^2)
        2. q = round(m / delta)
        3. if bit == 0:
               m' = q * delta
           else:
               m' = (q + 0.5) * delta
        4. v' = v * (m' / m)
        
    Returns:
        v': Modified vector
    """
    pass

def qim_extract_bit(
    v: np.ndarray,  # (2,) motion vector
    delta: float    # Quantization step
) -> int:
    """
    Extract bit from motion vector.
    
    Algorithm:
        1. m = ||v||
        2. q = round(m / delta)
        3. frac = (m / delta) - q
        4. bit = 0 if frac < 0.25 else 1
        
    Returns:
        bit: 0 or 1
    """
    pass
```

### Data Flow
- **Input**: FlowField + Bitstream
- **Output**: Modified FlowField + EmbeddingMap
- **Constraints**: ε, smoothness, magnitude bounds

---

## MODULE 6: VIDEO RECONSTRUCTION

### Interface

```python
class VideoReconstructor:
    """Reconstruct video from modified flows"""
    
    def reconstruct(
        self,
        frames: List[Frame],
        flows: List[FlowField]
    ) -> Tuple[List[Frame], QualityMetrics]:
        """
        Reconstruct video with modified motion.
        
        Args:
            frames: Original frames (N frames)
            flows: Modified flows (N-1 flows)
            
        Returns:
            stego_frames: Reconstructed frames
            metrics: Quality metrics vs original
        """
        pass
    
    def warp_frame(
        self,
        frame: Frame,
        flow: FlowField
    ) -> Frame:
        """
        Warp frame using flow field.
        
        Args:
            frame: Source frame
            flow: Motion field
            
        Returns:
            warped: Frame warped according to flow
        """
        pass
    
    def compute_quality(
        self,
        original: List[Frame],
        reconstructed: List[Frame]
    ) -> QualityMetrics:
        """
        Compute PSNR, SSIM, MSE.
        """
        pass
```

### Data Flow
- **Input**: Frames + Modified Flows
- **Output**: Stego Frames + Quality Metrics

---

## MODULE 7: RECEIVER ORCHESTRATOR

### Interface

```python
class Receiver:
    """Complete decoding pipeline"""
    
    def decode(
        self,
        stego_video_path: str,
        password: str,
        config: dict
    ) -> ReceiverResult:
        """
        Full receiver pipeline.
        
        Pipeline:
            1. Load stego video
            2. Extract motion fields
            3. Demodulate bits
            4. ECC decode
            5. Decrypt and verify
            
        Returns:
            result: ReceiverResult with plaintext or error
        """
        pass
```

### Data Flow
- **Input**: Stego video path + Password
- **Output**: Plaintext (bytes) or AuthenticationError

---

## MODULE 8: STEGANALYSIS ATTACKER

### Interface

```python
class SteganalysisDetector:
    """Neural steganalysis detector"""
    
    def extract_features(
        self,
        flows: List[FlowField]
    ) -> FeatureVector:
        """
        Extract 512-D feature vector.
        
        Features:
            - Magnitude histogram (128-D)
            - Directional histogram (128-D)
            - Temporal residuals (128-D)
            - Co-occurrence matrix (128-D)
        """
        pass
    
    def train(
        self,
        clean_videos: List[str],
        stego_videos: List[str],
        epochs: int = 50
    ) -> None:
        """
        Train binary classifier.
        """
        pass
    
    def predict(
        self,
        video_path: str
    ) -> SteganalysisResult:
        """
        Predict if video contains hidden data.
        
        Returns:
            result: Prediction + confidence + features
        """
        pass
    
    def evaluate_roc(
        self,
        test_videos: List[Tuple[str, int]]  # (path, label)
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute ROC curve and AUC.
        
        Returns:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under curve
        """
        pass
```

### Data Flow
- **Input**: Video (unknown type)
- **Output**: P(stego | video) + prediction

---

## MATHEMATICAL MODELS

### Channel Model

```python
def channel_model(
    signal: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """
    Simulate channel: Y = X + N
    
    Args:
        signal: Embedded signal (X)
        snr_db: Signal-to-noise ratio (dB)
        
    Returns:
        noisy_signal: Y = X + N
    """
    sigma_n = np.sqrt(10 ** (-snr_db / 10))
    noise = np.random.normal(0, sigma_n, signal.shape)
    return signal + noise
```

### Capacity Computation

```python
def shannon_capacity(
    snr_db: float,
    bandwidth_hz: float
) -> float:
    """
    Shannon capacity: C = B log2(1 + SNR)
    
    Returns:
        capacity: bits/second
    """
    snr_linear = 10 ** (snr_db / 10)
    return bandwidth_hz * np.log2(1 + snr_linear)
```

---

## ERROR HANDLING

All modules must handle errors gracefully:

```python
class MotionCovertCommError(Exception):
    """Base exception"""
    pass

class AuthenticationError(MotionCovertCommError):
    """AEAD tag verification failed"""
    pass

class ECCDecodeError(MotionCovertCommError):
    """Too many errors to correct"""
    pass

class InsufficientCapacityError(MotionCovertCommError):
    """Requested payload exceeds available capacity"""
    pass

class FlowExtractionError(MotionCovertCommError):
    """Optical flow extraction failed"""
    pass
```

---

## TESTING CONTRACTS

Each module must provide:

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test module combinations
3. **Validation tests**: Test with real data

Example test signatures:

```python
# tests/test_crypto.py
def test_encryption_decryption_roundtrip()
def test_authentication_failure_on_tamper()
def test_key_derivation_deterministic()

# tests/test_modulation.py
def test_qim_embedding_extraction_lossless()
def test_constraint_enforcement()
def test_capacity_estimation()

# tests/test_integration.py
def test_full_transmitter_receiver_pipeline()
def test_robustness_to_compression()
```

---

## SUMMARY

This specification ensures:
- ✅ Clear input/output types
- ✅ Explicit error handling
- ✅ Testable interfaces
- ✅ Modular design
- ✅ Type safety
- ✅ Documentation

All implementations must conform to these contracts.
