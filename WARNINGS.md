# Known Warnings (Expected)

## PyTorch / RAFT
- torch.load weights_only warning
- torch.cuda.amp.autocast deprecation warnings
Reason: Upstream RAFT implementation.

## Meshgrid Warning
- torch.meshgrid indexing warning
Reason: Future PyTorch API change, non-breaking.

These warnings do NOT affect correctness or security.
