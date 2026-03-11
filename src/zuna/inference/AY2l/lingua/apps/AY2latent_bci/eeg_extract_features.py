#!/usr/bin/env python3
"""
Extract ZUNA encoder features from preprocessed EEG .pt files.

Usage (same config/override style as eeg_eval.py):
    CUDA_VISIBLE_DEVICES=0 python3 eeg_extract_features.py \
        config=.../config_infer.yaml \
        data.data_dir=/path/to/2_pt_input \
        data.export_dir=/path/to/features \
        feature_pooling=mean \
        save_token_embeddings=false
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file as safe_load

from lingua.args import dataclass_from_dict

from apps.AY2latent_bci.eeg_data import (
    BCIDatasetArgs,
    EEGProcessor,
    create_dataloader_v2,
)
from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder


def _as_bool(value) -> bool:
    """Parse bool values from OmegaConf/CLI inputs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_config() -> dict:
    cli_cfg = OmegaConf.from_cli()
    if "config" not in cli_cfg:
        raise ValueError("Missing required CLI argument: config=<path/to/config_infer.yaml>")

    base_cfg = OmegaConf.load(cli_cfg.config)
    merged_cfg = OmegaConf.merge(base_cfg, cli_cfg)
    return OmegaConf.to_container(merged_cfg, resolve=True)


def _build_tok_idx(batch: dict, tok_idx_type: str, rope_dim: int) -> torch.Tensor | None:
    if tok_idx_type is None:
        return None
    if tok_idx_type == "t_coarse" and rope_dim == 1:
        return batch["t_coarse"].cpu().unsqueeze(0)
    if tok_idx_type == "chan_id" and rope_dim == 1:
        return batch["chan_id"].cpu().unsqueeze(0)
    if tok_idx_type == "stack_arange_seqlen" and rope_dim == 1:
        seq_lens = list(batch["seq_lens"].cpu().numpy())
        return torch.hstack([torch.arange(sl) for sl in seq_lens]).unsqueeze(0).unsqueeze(-1)
    if tok_idx_type == "{x,y,z,tc}" and rope_dim == 4:
        chan_pos_discrete = batch["chan_pos_discrete"].cpu().unsqueeze(0)
        t_coarse = batch["t_coarse"].cpu().unsqueeze(0)
        return torch.cat((chan_pos_discrete, t_coarse), dim=2)

    raise ValueError(f"Unsupported tok_idx_type={tok_idx_type} with rope_dim={rope_dim}")


def _split_by_seq_lens(enc_out: torch.Tensor, seq_lens: torch.Tensor) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    start = 0
    for seq_len in seq_lens.tolist():
        end = start + int(seq_len)
        chunks.append(enc_out[0, start:end, :])
        start = end
    return chunks


def _pool_features(token_features: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return token_features.mean(dim=0)
    if pooling == "max":
        return token_features.max(dim=0).values
    if pooling == "mean_max_concat":
        mean_feat = token_features.mean(dim=0)
        max_feat = token_features.max(dim=0).values
        return torch.cat([mean_feat, max_feat], dim=0)

    raise ValueError(f"Invalid pooling strategy: {pooling}")


def _make_output_path(export_dir: Path, source_filename: str, sample_idx: int) -> Path:
    stem = Path(source_filename).stem
    return export_dir / f"{stem}__sample_{sample_idx:05d}.pt"


def _load_model(device: torch.device) -> Tuple[EncoderDecoder, DecoderTransformerArgs]:
    repo_id = "Zyphra/ZUNA"
    weights_name = "model-00001-of-00001.safetensors"
    config_name = "config.json"

    config_path = hf_hub_download(repo_id=repo_id, filename=config_name)
    with open(config_path, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    model_args = dataclass_from_dict(DecoderTransformerArgs, model_cfg["model"])

    weights_path = hf_hub_download(repo_id=repo_id, filename=weights_name, token=False)
    state_dict_raw = safe_load(weights_path, device="cpu")
    state_dict = {k.removeprefix("model."): v.to(device) for k, v in state_dict_raw.items()}

    model = EncoderDecoder(model_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_args


def main() -> None:
    cfg = _parse_config()

    data_args = dataclass_from_dict(BCIDatasetArgs, cfg["data"])
    seed = int(cfg.get("seed", 42))
    pooling = str(cfg.get("feature_pooling", "mean"))
    save_token_embeddings = _as_bool(cfg.get("save_token_embeddings", False))

    valid_pooling = {"mean", "max", "mean_max_concat"}
    if pooling not in valid_pooling:
        raise ValueError(f"Invalid feature_pooling='{pooling}'. Use one of {sorted(valid_pooling)}")

    export_dir = Path(data_args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model, model_args = _load_model(device)
    data_loader = create_dataloader_v2(data_args, seed=seed, rank=0)
    data_processor = EEGProcessor(data_args).to(device)

    total_samples = 0
    total_files = set()

    with torch.no_grad():
        for batch in data_loader:
            eeg_signal = batch["eeg_signal"] / data_args.data_norm
            if data_args.data_clip is not None:
                eeg_signal = eeg_signal.clamp(min=-data_args.data_clip, max=data_args.data_clip)

            filenames = batch.pop("filename", None)
            sample_indices = batch.pop("sample_idx", None)
            metadata_list = batch.pop("metadata", None)

            batch = {
                "eeg_signal": eeg_signal,
                "chan_pos": batch["chan_pos"],
                "chan_pos_discrete": batch["chan_pos_discrete"],
                "chan_id": batch["chan_id"],
                "t_coarse": batch["t_coarse"],
                "seq_lens": batch["seq_lens"],
                "max_tc": batch["max_tc"],
                "chan_dropout": batch["chan_dropout"],
            }

            batch = data_processor.process(**batch)
            batch = {
                k: v.to(device, non_blocking=(device.type == "cuda"))
                for k, v in batch.items()
            }

            tok_idx = _build_tok_idx(batch, model_args.tok_idx_type, model_args.rope_dim)

            # Keep shape handling aligned with inference path: [seq_len, dim] -> [1, seq_len, dim].
            encoder_input = batch["encoder_input"].unsqueeze(0)
            do_idx = (encoder_input.sum(dim=2) == 0).squeeze(0)

            enc_out, _ = model.encoder(
                token_values=encoder_input,
                seq_lens=batch["seq_lens"],
                tok_idx=tok_idx,
                do_idx=do_idx,
            )

            token_chunks = _split_by_seq_lens(enc_out.detach().cpu(), batch["seq_lens"].cpu())

            for i, token_features in enumerate(token_chunks):
                source_filename = filenames[i]
                sample_idx = int(sample_indices[i])
                pooled_feature = _pool_features(token_features, pooling)

                payload = {
                    "source_filename": source_filename,
                    "sample_idx": sample_idx,
                    "pooling": pooling,
                    "pooled_feature": pooled_feature.float(),
                    "token_features": token_features.float() if save_token_embeddings else None,
                    "metadata": metadata_list[i],
                }

                out_path = _make_output_path(export_dir, source_filename, sample_idx)
                torch.save(payload, out_path)

                total_samples += 1
                total_files.add(source_filename)

    print(
        f"Feature extraction done. Saved {total_samples} samples from {len(total_files)} source files to {export_dir}",
        flush=True,
    )


if __name__ == "__main__":
    # Keep behavior deterministic across runs.
    torch.manual_seed(42)
    np.random.seed(42)
    main()
