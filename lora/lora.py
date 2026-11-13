import os, torch, torch.nn as nn
os.environ["HF_HOME"] = "/egr/research-slim/luozishe/hf_cache"
from neuraltrain.models.reve import NtReve
from peft import LoraConfig, inject_adapter_in_model
from neuralbench import evaluate_model

model = NtReve(from_pretrained_name="/egr/research-slim/luozishe/eeg-image-challenge").build(
    n_spatial_locations=32, n_temporal_samples=200, n_outputs=None)
prefix  = next(n for n, m in model.named_modules() if type(m).__name__ == "TransformerBackbone")
targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith(prefix + ".")]
model = inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets), model)
for n, p in model.named_parameters():
    p.requires_grad = "lora_" in n
print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

overrides = {
    "data.study.source.path": "/egr/research-slim/shared/neuralbench",
    "data.neuro.frequency": 200.0,
    "data.channel_positions.normalize": False,
    "lightning_optimizer_config.optimizer.lr": 1e-4,
    "trainer_config.monitor": "val/batch_top5_acc",
}
scores = evaluate_model(model, "eeg", "image", name="reve-lora-r16",
                        dataset="xu2025alljoined", downstream_wrapper="finetune_mean",
                        overrides=overrides, download=False, debug=True)
print(scores.T if hasattr(scores, "T") else scores)