"""
Example: Receiver Usage

Demonstrates how to use the ReceiverEngine to extract data from stego videos.
"""

import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.module8_receiver import ReceiverEngine


def create_example_config():
    """Create example configuration (matches default_config.yaml structure)."""
    return {
        'system': {
            'version': '1.0.0',
            'verbose': True,
            'debug_mode': False,
        },
        'optical_flow': {
            'model': 'raft',
            'preprocessing': {
                'normalize': True,
                'max_flow_magnitude': 100.0,
            }
        },
        'modulation': {
            'method': 'qim',
            'embedding': {
                'epsilon': 0.5,
                'quantization_step': 2.0,
                'max_payload_bits': 4096,
            },
            'selection': {
                'motion_threshold': 1.0,
                'use_high_motion_regions': True,
                'spatial_distribution': 'uniform',
            },
            'demodulation': {
                'decision_boundary': 0.25,
                'use_soft_decisions': False,
            },
            'constraints': {
                'enforce_smoothness': True,
                'smoothness_kernel_size': 5,
                'smoothness_sigma': 0.1,
                'enforce_magnitude_bounds': True,
                'min_magnitude_ratio': 0.8,
                'max_magnitude_ratio': 1.2,
                'enforce_perceptual_limit': True,
                'max_l_infinity_norm': 1.0,
            }
        }
    }


def create_synthetic_video(num_frames=20, height=128, width=128):
    """
    Create synthetic video frames with motion.
    
    In a real system, these would come from Module 1's load_video() function.
    """
    print(f"Creating synthetic video: {num_frames} frames, {height}x{width}")
    
    frames = []
    for i in range(num_frames):
        # Create frame with moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add diagonal stripe pattern that moves
        offset = i * 3
        for y in range(height):
            for x in range(width):
                if (x + y + offset) % 20 < 10:
                    frame[y, x] = [255, 255, 255]
                else:
                    frame[y, x] = [50, 50, 50]
        
        frames.append(frame)
    
    return frames


def example_basic_extraction():
    """Example 1: Basic extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Extraction")
    print("="*70)
    
    # Create configuration
    config = create_example_config()
    
    # Initialize receiver
    print("\nInitializing ReceiverEngine...")
    receiver = ReceiverEngine(config)
    
    # Create synthetic stego video
    # In real usage: frames = video_io.load_video("stego_video.mp4")
    frames = create_synthetic_video(num_frames=20)
    
    # Extract bitstream
    print(f"\nExtracting from {len(frames)} frames...")
    bitstream = receiver.extract(frames)
    
    print(f"\n✓ Extraction complete!")
    print(f"  Output size: {len(bitstream)} bytes")
    print(f"  Bitstream (hex): {bitstream.hex()[:64]}..." if len(bitstream) > 0 else "  Empty bitstream")


def example_extraction_with_metadata():
    """Example 2: Extraction with detailed metadata."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Extraction with Metadata")
    print("="*70)
    
    config = create_example_config()
    receiver = ReceiverEngine(config)
    
    # Create video
    frames = create_synthetic_video(num_frames=15)
    
    # Extract with metadata
    print(f"\nExtracting from {len(frames)} frames...")
    bitstream, metadata = receiver.extract_with_metadata(frames)
    
    print(f"\n✓ Extraction complete!")
    print(f"\nMetadata:")
    print(f"  Total bits extracted: {metadata['total_bits']}")
    print(f"  Total bytes: {metadata['total_bytes']}")
    print(f"  Frames processed: {metadata['num_frames_processed']}")
    print(f"  Frames skipped: {metadata['num_frames_skipped']}")
    print(f"  Average bits/frame: {metadata['avg_bits_per_frame']:.2f}")
    print(f"  Extraction time: {metadata['extraction_time_seconds']:.3f}s")
    
    # Show per-frame statistics
    print(f"\nPer-frame statistics:")
    for stat in metadata['per_frame_stats'][:5]:  # Show first 5
        status = "SKIPPED" if stat['skipped'] else f"{stat['bits_extracted']} bits"
        print(f"  Frame {stat['frame_idx']:2d}: {status}")
    
    if len(metadata['per_frame_stats']) > 5:
        print(f"  ... ({len(metadata['per_frame_stats']) - 5} more frames)")


def example_config_variations():
    """Example 3: Testing different configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Configuration Variations")
    print("="*70)
    
    # Create base config
    config = create_example_config()
    
    # Test different spatial distributions
    distributions = ['uniform', 'adaptive']
    
    frames = create_synthetic_video(num_frames=10)
    
    for dist in distributions:
        print(f"\nTesting spatial_distribution = '{dist}'")
        config['modulation']['selection']['spatial_distribution'] = dist
        
        receiver = ReceiverEngine(config)
        bitstream, metadata = receiver.extract_with_metadata(frames)
        
        print(f"  Total bits: {metadata['total_bits']}")
        print(f"  Frames skipped: {metadata['num_frames_skipped']}/{metadata['num_frames_processed']}")


def example_integration_pipeline():
    """Example 4: Full receiver pipeline integration."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Full Pipeline Integration")
    print("="*70)
    
    print("\nSimulating full receiver pipeline:")
    print("  Stego Video → Flow Extraction → QIM Demod → Raw Bits")
    
    config = create_example_config()
    receiver = ReceiverEngine(config)
    
    # Step 1: Load video (simulated)
    print("\n[Step 1] Loading stego video...")
    frames = create_synthetic_video(num_frames=12)
    print(f"  Loaded {len(frames)} frames")
    
    # Step 2: Extract bitstream
    print("\n[Step 2] Extracting embedded data...")
    bitstream, metadata = receiver.extract_with_metadata(frames)
    print(f"  Extracted {metadata['total_bits']} bits ({len(bitstream)} bytes)")
    
    # Step 3: Pass to ECC decoder (simulated)
    print("\n[Step 3] Would pass to Module 4 (ECC decoder)...")
    print(f"  Input to ECC: {len(bitstream)} bytes")
    print(f"  # decoded_bits = ecc_decoder.decode(bitstream)")
    
    # Step 4: Pass to crypto (simulated)
    print("\n[Step 4] Would pass to Module 3 (crypto decryptor)...")
    print(f"  # plaintext = crypto.decrypt(decoded_bits, password)")
    
    print("\n✓ Pipeline complete!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("RECEIVER ENGINE - USAGE EXAMPLES")
    print("="*70)
    
    try:
        example_basic_extraction()
        example_extraction_with_metadata()
        example_config_variations()
        example_integration_pipeline()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()