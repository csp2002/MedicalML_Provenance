"""LoRA adapters on the query and value parts of every Hiera block's `attn.qkv` projection (MedSAM2 image encoder);
the rest of the encoder is frozen. `merged_state_dict` folds the adapters back into the weights, so a fine-tuned
checkpoint loads into a plain MedSAM2 model.
"""
import torch
import torch.nn as nn

LORA_KEYS = ("a_q", "b_q", "a_v", "b_v")


class LoRAQKV(nn.Module):
    """Wraps a Hiera `attn.qkv` Linear and adds a low-rank residual to its Q and V slices."""

    def __init__(self, qkv: nn.Linear, r: int = 4, alpha: float = 4.0):
        super().__init__()
        assert qkv.out_features % 3 == 0, "expected a fused qkv projection"
        self.qkv = qkv
        self.d_in = qkv.in_features
        self.d_out = qkv.out_features // 3
        self.scaling = alpha / r
        f = dict(device=qkv.weight.device, dtype=qkv.weight.dtype)
        self.a_q = nn.Linear(self.d_in, r, bias=False, **f)
        self.b_q = nn.Linear(r, self.d_out, bias=False, **f)
        self.a_v = nn.Linear(self.d_in, r, bias=False, **f)
        self.b_v = nn.Linear(r, self.d_out, bias=False, **f)
        nn.init.zeros_(self.b_q.weight)
        nn.init.zeros_(self.b_v.weight)

    def forward(self, x):
        q, k, v = self.qkv(x).split(self.d_out, dim=-1)
        return torch.cat([q + self.scaling * self.b_q(self.a_q(x)),
                          k,
                          v + self.scaling * self.b_v(self.a_v(x))], dim=-1)


def apply_lora(image_encoder, r=4, alpha=4.0):
    """Freeze `image_encoder` and attach rank-r adapters to every Hiera block.

    Call this AFTER the pretrained weights are loaded (the checkpoint uses the unwrapped key names).
    """
    for p in image_encoder.parameters():
        p.requires_grad = False
    blocks = image_encoder.trunk.blocks
    for blk in blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r=r, alpha=alpha)
    for m in image_encoder.modules():
        if isinstance(m, LoRAQKV):
            for sub in (m.a_q, m.b_q, m.a_v, m.b_v):
                sub.weight.requires_grad = True
    return len(blocks)


def is_lora_param(name):
    return any(f".{k}.weight" in name for k in LORA_KEYS)


def merged_state_dict(model):
    """state_dict with every adapter folded into its base qkv weight, under the original key names."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRAQKV):
            continue
        d = mod.d_out
        w = sd.pop(f"{name}.qkv.weight")
        w[:d] += ((mod.b_q.weight @ mod.a_q.weight) * mod.scaling).to(w.dtype)
        w[2 * d:] += ((mod.b_v.weight @ mod.a_v.weight) * mod.scaling).to(w.dtype)
        sd[f"{name}.weight"] = w
        b = sd.pop(f"{name}.qkv.bias", None)
        if b is not None:
            sd[f"{name}.bias"] = b
        for k in LORA_KEYS:
            sd.pop(f"{name}.{k}.weight", None)
    return sd
