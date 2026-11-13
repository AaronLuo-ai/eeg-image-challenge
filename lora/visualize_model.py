import numpy as np, torch, matplotlib.pyplot as plt

FLOW = ["0.norm", "0.to_qkv", "0.attend", "0.to_out",     # attention half
        "1.net.0", "1.net.1", "1.net.2", "1.net.3"]       # feed-forward half
LORA = ["0.to_qkv", "0.to_out", "1.net.1", "1.net.3"]
LORA_LABELS = ["to_qkv", "to_out", "net.1", "net.3"]

def _n(m, trainable=None):
    return sum(p.numel() for p in m.parameters()
               if trainable is None or p.requires_grad == trainable)

def show_lora_model(model, block=0):
    reve   = next(m for m in model.modules() if type(m).__name__ == "REVE")
    blocks = reve.transformer.layers

    # 1. parameter table for each part of REVE
    print(f"{'part':22s}{'frozen':>14s}{'trainable':>12s}")
    for name, m in reve.named_children():
        print(f"{name:22s}{_n(m, False):14,}{_n(m, True):12,}")
    own = [p for p in reve.parameters(recurse=False)]
    print(f"{'(REVE own params)':22s}{sum(p.numel() for p in own if not p.requires_grad):14,}"
          f"{sum(p.numel() for p in own if p.requires_grad):12,}")
    print(f"{'TOTAL':22s}{_n(model, False):14,}{_n(model, True):12,}\n")

    # 2. one block in data-flow order
    blk = blocks[block]
    print(f"Block {block} of {len(blocks)} (data-flow order)")
    for name in FLOW:
        m = blk.get_submodule(name)
        if hasattr(m, "base_layer"):
            b, A, B = m.base_layer, m.lora_A["default"], m.lora_B["default"]
            print(f"  {name:9s} W0 {b.in_features:>4}→{b.out_features:<5} frozen {b.weight.numel():>9,}"
                  f" | A {A.in_features}→{A.out_features}, B {B.in_features}→{B.out_features}"
                  f"  trainable {A.weight.numel() + B.weight.numel():>7,}")
        else:
            print(f"  {name:9s} {type(m).__name__:20s} params {_n(m):,}")
        if name in ("0.to_out", "1.net.3"):
            print("  + residual")

    # 3. charts (log scale; zero bars are hidden)
    nz = lambda v: [x if x > 0 else np.nan for x in v]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    parts = list(reve.named_children()); y = np.arange(len(parts))
    ax[0].barh(y + 0.2, nz([_n(m, False) for _, m in parts]), 0.4, color="0.6", label="frozen")
    ax[0].barh(y - 0.2, nz([_n(m, True) for _, m in parts]), 0.4, color="tab:purple", label="trainable (LoRA)")
    ax[0].set_yticks(y, [n for n, _ in parts]); ax[0].invert_yaxis(); ax[0].set_xscale("log")
    ax[0].set_xlabel("parameters"); ax[0].set_title("REVE parts"); ax[0].legend()

    x = np.arange(len(LORA))
    ax[1].bar(x - 0.2, [blk.get_submodule(n).base_layer.weight.numel() for n in LORA], 0.4, color="0.6", label="frozen $W_0$")
    ax[1].bar(x + 0.2, [_n(blk.get_submodule(n), True) for n in LORA], 0.4, color="tab:purple", label="LoRA $A+B$")
    ax[1].set_xticks(x, LORA_LABELS); ax[1].set_yscale("log"); ax[1].set_ylabel("parameters")
    ax[1].set_title(f"Per layer, one block (×{len(blocks)} blocks)"); ax[1].legend()
    plt.tight_layout(); plt.show()