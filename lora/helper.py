"""reve_lora.py — REVE + LoRA on NeuralBench's EEG-to-image task, with stage logging.
 
Run from a terminal (inside tmux):
    python reve_lora.py
Logs go to the screen and to train_log.txt in the current folder.
"""
import os
 
# Must be set before importing anything that loads Hugging Face.
os.environ.setdefault("HF_HOME", "/egr/research-slim/luozishe/hf_cache")
 
import logging
import time
import typing as tp
 
import pandas as pd
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from neuraltrain.models.reve import NtReve
import logging
LOGGER = logging.getLogger("reve_lora")

def build_reve_lora(
    weights_dir: str,
    n_channels: int = 32,
    n_samples: int = 200,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> nn.Module:
    """Build REVE from local weights and inject LoRA into the 22 transformer blocks.
 
    Only the LoRA A/B matrices are left trainable (88 pairs, ~2.88M params for r=16).
    """
    model = NtReve(from_pretrained_name=weights_dir).build(
        n_spatial_locations=n_channels, n_temporal_samples=n_samples, n_outputs=None
    )
 
    # Find targets on the *fresh* model, before injection (to_qkv, to_out, net.1, net.3).
    prefix = next(n for n, m in model.named_modules() if type(m).__name__ == "TransformerBackbone")
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.startswith(prefix + ".")]
 
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets)
    model = inject_adapter_in_model(cfg, model)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
 
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("LoRA targets: %d Linear layers", len(targets))
    LOGGER.info("Params: %.2fM total, %.3fM trainable (%.1f%%)",
                n_all / 1e6, n_train / 1e6, 100 * n_train / n_all)
    return model
 