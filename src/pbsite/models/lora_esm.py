"""(c) ESM-2 fine-tuned with LoRA for per-residue binding classification.

Wraps a HuggingFace ESM-2 encoder with a PEFT LoRA adapter and a per-token
classification head. Only LoRA params + the head are trained; the base weights
stay frozen, which is what makes this fit small GPUs (150M locally, 650M on Colab).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoraESMTokenClassifier(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/esm2_t30_150M_UR50D",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
        head_dropout: float = 0.2,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_id)
        if grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            # Required so gradients flow to LoRA params through checkpointed layers
            # on an otherwise-frozen base model.
            self.backbone.enable_input_require_grads()
        hidden = self.backbone.config.hidden_size

        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["query", "key", "value", "dense"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """Return per-token logits (B, L) aligned to input_ids (incl. special tokens)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        return self.classifier(self.dropout(h)).squeeze(-1)

    def trainable_parameter_summary(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}
