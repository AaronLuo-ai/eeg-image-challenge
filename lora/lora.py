"""reve_lora.py — REVE + LoRA on NeuralBench's EEG-to-image task, with stage logging.

Run from a terminal (inside tmux), NOT from Jupyter:
    python reve_lora.py --debug     # quick check: 2 epochs x 5 batches
    python reve_lora.py             # real run
Logs go to the screen and to train_log.txt in the current folder.
"""
import os
import sys

# Must be set before neuralbench/tqdm are imported (avoids the rich/tqdm progress-bar clutter).
os.environ.setdefault("TQDM_DISABLE", "1")
sys.setrecursionlimit(10000)

# Must be set before importing anything that loads Hugging Face.
os.environ.setdefault("HF_HOME", "/egr/research-slim/luozishe/hf_cache")

import logging
import time
import typing as tp

import pandas as pd
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from neuraltrain.models.reve import NtReve

# Private helpers taken from the installed neuralbench, so this copy behaves
# exactly like neuralbench.evaluate_model. They are internal API: if a future
# neuralbench release renames them, update these imports.
from neuralbench.evaluate import (
    ConfDict,
    _download_dataset,
    _experiment_configs,
    _external_config,
    _run,
)

LOGGER = logging.getLogger("reve_lora")


def _stage(msg: str, t0: float) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("[%6.0fs]  STAGE: %s", time.time() - t0, msg)
    LOGGER.info("=" * 60)


# ---------------------------------------------------------------------------
# 1. Model: pretrained REVE with LoRA on every Linear in the transformer
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Local copy of neuralbench.evaluate_model, with stage logging
# ---------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    device: str,
    task: str | list[str],
    *,
    dataset: str | list[str] | None = None,
    space: str | list[str] | None = None,
    name: str = "external",
    overrides: tp.Mapping[str, tp.Any] | None = None,
    downstream_wrapper: str | list[str] = "linear_probe_mean",
    download: bool = True,
    prepare: bool = True,
    cluster: str | None = None,
    debug: bool = False,
    **selection: tp.Any,
) -> pd.DataFrame:
    """Same as neuralbench.evaluate_model, plus a log line at each stage.

    The training loop itself (dataloaders, Lightning trainer.fit, test) runs
    inside _run -> BenchmarkAggregator -> Experiment.run (neuralbench/main.py).
    """
    t0 = time.time()

    _stage("1/5  Serialize model into the experiment config", t0)
    config = _external_config(model)
    overlay = {
        "brain_model_name": name,
        # =replace= so the external config supplants the default model.
        "brain_model_config": {"=replace=": True, **config.model_dump()},
        "compatible_spaces": "all",
    }
    phase = dict(
        device=device, task=task, dataset=dataset, space=space,
        overrides=overrides, downstream_wrapper=downstream_wrapper,
        cluster=cluster, debug=debug, **selection,
    )

    _stage("2/5  Build experiment configs (task YAML + overrides)", t0)
    configs = _experiment_configs(overlay, **phase)
    LOGGER.info("%d experiment(s) to run", len(configs))

    if download:
        _stage("3/5  Download datasets", t0)
        for cfg in {c["data.study.source"]["name"]: c for c in configs}.values():
            _download_dataset(cfg)
    else:
        LOGGER.info("Skipping download (download=False)")

    if prepare:
        _stage("4/5  Warm caches (cut-down experiment: preprocessing + DINOv2 targets)", t0)
        warming = _run(_experiment_configs(overlay, prepare=True, **phase), debug)
        if cluster is not None and any(
            exp.infra.status() != "completed" for exp in warming.experiments
        ):
            LOGGER.info("Warming the caches on %s; call again to run.", cluster)
            return pd.DataFrame()
    else:
        LOGGER.info("Skipping cache warm-up (prepare=False)")

    _stage("5/5  Train + validate + test  (Experiment.run)", t0)
    agg = _run(configs, debug)

    _stage("Collect results", t0)
    frame = pd.DataFrame(agg.collect(cached_only=not debug))
    if not frame.empty and overrides:
        flat = ConfDict(dict(overrides)).flat()
        frame["overrides"] = ";".join(f"{k}={v!r}" for k, v in sorted(flat.items()))

    _stage(f"Done ({len(frame)} result row(s))", t0)
    return frame


# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="tiny run: 2 epochs x 5 batches")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_log.txt", mode="w")],
    )
    logging.captureWarnings(True)

    LOGGER.info("debug=%s", args.debug)
    model = build_reve_lora("/egr/research-slim/luozishe/eeg-image-challenge", r=16)

    overrides = {
        "data.study.source.path": "/egr/research-slim/shared/neuralbench",  # parent of Xu2025Alljoined
        "data.neuro.frequency": 200.0,                    # REVE patch_size = 200 samples
        "data.channel_positions.normalize": False,        # REVE expects raw coordinates
        "lightning_optimizer_config.optimizer.lr": 1e-4,
        "trainer_config.monitor": "val/batch_top5_acc",   # checkpoint on the competition metric
        "delete_checkpoints_on_exit": False,              # keep best.ckpt (remove this line if rejected)
    }

    scores = evaluate_model(
        model=model,            # REVE + LoRA backbone
        device="eeg",           # signal type (not the GPU)
        task="image",           # NeuralBench task: EEG-to-image retrieval
        name="reve-lora-r16",
        dataset="xu2025alljoined",
        downstream_wrapper="finetune_mean",
        overrides=overrides,
        download=False,
        prepare=False,      # caches are already warm; set True if you change frequency/dataset
        debug=args.debug,   # python reve_lora.py --debug for a quick check
    )

    if not scores.empty:
        LOGGER.info("Results:\n%s", scores.T.to_string())
        scores.to_csv("scores_reve_lora_r16.csv", index=False)
        LOGGER.info("Saved scores_reve_lora_r16.csv")