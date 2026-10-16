"""Model / tokenizer construction for the four classifiers.

Encoders (DistilBERT, ModernBERT): standard sequence-classification head, full fine-tuning.
Decoders (TinyLlama, Qwen3): AutoModelForSequenceClassification puts a linear `score` head on the LAST non-padding
token; only LoRA adapters + the head are trained (peft TaskType.SEQ_CLS keeps the head trainable).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = {0: "no_preference", 1: "preference"}


def resolve_revision(hf_id, revision=None):
    """Exact Hugging Face commit hash for reproducibility (pinned in metrics.json and the run manifest)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(hf_id, revision=revision).sha
    except Exception:
        return revision      # offline: whatever is cached locally


def build(cfg, revision):
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=revision)
    if tok.pad_token is None:                       # Llama tokenizers have no pad token
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.padding_side = "right"
    decoder = cfg["kind"] == "decoder"
    kwargs = dict(cfg.get("model_kwargs") or {})
    if decoder and cfg.get("precision") == "bf16":
        kwargs["dtype"] = torch.bfloat16      # 1-2B weights in bf16 (fits 12 GB with LoRA)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_id"], revision=revision, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()},
        **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    if decoder:
        from peft import LoraConfig, TaskType, get_peft_model
        if cfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"],
                                                 lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    return tok, model


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)
