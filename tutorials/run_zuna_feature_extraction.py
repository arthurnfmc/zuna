#!/usr/bin/env python3
"""
Zuna Feature Extraction Tutorial

Runs ZUNA encoder-only feature extraction:
1. (Optional) Preprocessing: .fif -> .pt
2. Feature extraction: .pt -> embeddings (.pt per sample)

Edit paths and options below, then run:
    python tutorials/run_zuna_feature_extraction.py
"""

from pathlib import Path
import math

import torch
from zuna import preprocessing, extract_features

# =============================================================================
# PATHS
# =============================================================================

TUTORIAL_DIR = Path(__file__).parent.resolve()
INPUT_DIR = str(TUTORIAL_DIR / "data" / "1_fif_input")
WORKING_DIR = str(TUTORIAL_DIR / "data" / "working")

WORKING_PATH = Path(WORKING_DIR)
PT_INPUT_DIR = str(WORKING_PATH / "2_pt_input")
FEATURE_DIR = str(WORKING_PATH / "5_features")
PREPROCESSED_FIF_DIR = str(WORKING_PATH / "1_fif_filter")

# =============================================================================
# PREPROCESSING OPTIONS
# =============================================================================

RUN_PREPROCESSING = False
INPUT_TYPE = "auto"
EPOCH_DURATION = 5.0
APPLY_NOTCH_FILTER = True
APPLY_HIGHPASS_FILTER = True
APPLY_AVERAGE_REFERENCE = True
TARGET_CHANNEL_COUNT = 40
BAD_CHANNELS = ["Fz", "Cz", "Pz"]

# =============================================================================
# MOCK DATA OPTIONS
# =============================================================================

GENERATE_MOCK_PT_IF_EMPTY = True
MOCK_NUM_FILES = 2
MOCK_EPOCHS_PER_FILE = 8
MOCK_NUM_CHANNELS = 40
MOCK_SAMPLES_PER_EPOCH = 1280

# =============================================================================
# FEATURE EXTRACTION OPTIONS
# =============================================================================

GPU_DEVICE = 0
TOKENS_PER_BATCH = 100000
DATA_NORM = 10.0
POOLING = "mean"                # "mean", "max", "mean_max_concat"
SAVE_TOKEN_EMBEDDINGS = False    # True saves full token-level embeddings


def _make_mock_channel_positions(n_channels: int) -> torch.Tensor:
    """Create deterministic 3D points on a unit sphere."""
    points = []
    phi = (1 + 5 ** 0.5) / 2
    for i in range(n_channels):
        z = 1 - (2 * i + 1) / n_channels
        r = max(0.0, 1 - z * z) ** 0.5
        theta = 2 * math.pi * i / phi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append([x, y, z])
    return 0.1 * torch.tensor(points, dtype=torch.float32)


def _generate_mock_pt_dataset(
    output_dir: str,
    num_files: int,
    epochs_per_file: int,
    n_channels: int,
    n_samples: int,
) -> None:
    """Generate Zuna-compatible mock .pt files for quick feature extraction tests."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chan_positions = _make_mock_channel_positions(n_channels)
    channel_names = [f"E{i+1:02d}" for i in range(n_channels)]

    for file_idx in range(num_files):
        data_list = []
        pos_list = []

        # Stable seed per file for reproducible tests.
        gen = torch.Generator().manual_seed(1234 + file_idx)

        for epoch_idx in range(epochs_per_file):
            t = torch.linspace(0, 5, steps=n_samples)
            base = torch.sin(2 * math.pi * (8 + file_idx) * t / 5.0)
            noise = 0.05 * torch.randn((n_channels, n_samples), generator=gen)
            channel_scale = torch.linspace(0.8, 1.2, steps=n_channels).unsqueeze(1)
            signal = channel_scale * base.unsqueeze(0) + noise

            # Inject sparse missing channels to emulate dropout-like inputs.
            if epoch_idx % 3 == 0:
                signal[0] = 0.0

            data_list.append(signal.to(torch.float32))
            pos_list.append(chan_positions.clone())

        metadata = {
            "n_epochs": epochs_per_file,
            "channel_names": channel_names,
            "avg_channels_per_epoch": float(n_channels),
            "samples_per_epoch": n_samples,
            "sampling_rate": 256.0,
            "resampled_sampling_rate": 256.0,
            "original_filename": f"mock_source_{file_idx:03d}.fif",
            "channels_dropped_no_coords": [],
        }

        filename = (
            f"ds000000_{file_idx:06d}_000001_d00_"
            f"{epochs_per_file:05d}_{n_channels:03d}_{n_samples}.pt"
        )
        payload = {
            "data": data_list,
            "channel_positions": pos_list,
            "metadata": metadata,
        }
        torch.save(payload, out / filename)

    print(f"Generated mock PT dataset in: {out}")


if __name__ == "__main__":
    Path(PT_INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

    if GENERATE_MOCK_PT_IF_EMPTY and not list(Path(PT_INPUT_DIR).glob("*.pt")):
        print("[0/2] Generating mock PT dataset...", flush=True)
        _generate_mock_pt_dataset(
            output_dir=PT_INPUT_DIR,
            num_files=MOCK_NUM_FILES,
            epochs_per_file=MOCK_EPOCHS_PER_FILE,
            n_channels=MOCK_NUM_CHANNELS,
            n_samples=MOCK_SAMPLES_PER_EPOCH,
        )

    if RUN_PREPROCESSING:
        print("[1/2] Preprocessing FIF to PT...", flush=True)
        preprocessing(
            input_dir=INPUT_DIR,
            output_dir=PT_INPUT_DIR,
            input_type=INPUT_TYPE,
            epoch_duration=EPOCH_DURATION,
            apply_notch_filter=APPLY_NOTCH_FILTER,
            apply_highpass_filter=APPLY_HIGHPASS_FILTER,
            apply_average_reference=APPLY_AVERAGE_REFERENCE,
            preprocessed_fif_dir=PREPROCESSED_FIF_DIR,
            target_channel_count=TARGET_CHANNEL_COUNT,
            bad_channels=BAD_CHANNELS,
        )

    print("[2/2] Extracting encoder features...", flush=True)
    extract_features(
        input_dir=PT_INPUT_DIR,
        output_dir=FEATURE_DIR,
        gpu_device=GPU_DEVICE,
        tokens_per_batch=TOKENS_PER_BATCH,
        data_norm=DATA_NORM,
        pooling=POOLING,
        save_token_embeddings=SAVE_TOKEN_EMBEDDINGS,
    )

    print(f"Done. Features saved to: {FEATURE_DIR}")
