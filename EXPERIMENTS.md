## Experimental Setup
- Synthetic motion fields
- Real video sequences
- RAFT optical flow

## Capacity Analysis
- Zero motion → zero capacity
- Threshold behavior
- High motion → linear growth

## End-to-End Validation
- ECC-protected payloads
- QIM embedding
- Successful extraction

## Failure Modes
- Low motion videos
- Heavy compression


## Experiment 1 — Capacity vs Motion Magnitude

**Objective:**  
Verify that embedding capacity is deterministically gated by motion magnitude threshold.

**Method:**  
Synthetic optical flow fields (64×64) were generated with uniform horizontal motion.
Motion magnitude was swept across values below and above the configured threshold.

**Results:**

| Motion Magnitude | Capacity (bits) |
|------------------|-----------------|
| 0.5              | 0               |
| 1.0              | 4096            |
| 2.0              | 4096            |
| 5.0              | 4096            |
| 10.0             | 4096            |

**Conclusion:**  
Capacity exhibits a hard threshold behavior. Once motion exceeds the threshold,
capacity saturates to the maximum allowed by the embedding configuration.
This confirms deterministic frame eligibility and encoder–decoder synchronization.

## Experiment 2: Synthetic Motion Roundtrip

**Objective:**  
Validate correctness of QIM embedding + extraction + ECC decoding under ideal motion conditions.

**Setup:**
- Flow: uniform horizontal motion (dx = 5.0)
- Resolution: 64×64
- Payload: "HELLO_COVERT"
- ECC: Reed–Solomon

**Results:**
- Bits embedded: 2040
- Bits extracted: 2040
- Payload recovered perfectly

**Conclusion:**  
The modulation–demodulation–ECC pipeline is functionally correct under controlled motion.
