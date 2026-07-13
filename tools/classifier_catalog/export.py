"""Export a HuggingFace text/token classifier to ONNX for the Lemonade
`onnxruntime` backend (ort-server), and validate it against the original model.

This is the exact script used to produce the artifacts in the lemonade-sdk ONNX
classifier repos. Reproduce with:

    pip install "optimum[onnxruntime]" transformers torch onnxruntime sentencepiece
    python export.py <hf_model_id> <out_dir> [--task text-classification|token-classification] [--trust-remote-code]

Outputs into <out_dir>: model.onnx, the tokenizer files (incl. tokenizer.json),
manifest.json (task/labels/normalization for ort-server), and validation.json
(parity vs the original PyTorch model on fixtures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXTURES = [
    "My name is John Smith and my SSN is 123-45-6789.",
    "URGENT: verify your account at http://secure-login.example to avoid suspension.",
    "Thanks for the notes from today's standup, talk tomorrow.",
    # Longer than any encoder budget, so validation exercises the truncated path.
    "Please review the attached quarterly figures before the meeting. " * 120,
]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def infer_normalization(config, task: str, override: str | None = None) -> str:
    """Map HF problem_type semantics to the ort-server manifest contract.

    Regression models have no label scores in [0,1] and are rejected outright;
    multi-label classification needs independent sigmoids, not softmax.
    problem_type is only a training-time hint — models trained with BCE outside
    the HF Trainer often leave it null — so --normalization can override.
    """
    problem_type = getattr(config, "problem_type", None)
    if problem_type == "regression" or getattr(config, "num_labels", 2) < 2:
        raise SystemExit(
            "unsupported model: regression / single-output heads have no label "
            "scores in [0,1] (problem_type="
            f"{problem_type!r}, num_labels={getattr(config, 'num_labels', None)})"
        )
    if override:
        return override
    if task == "text-classification" and problem_type == "multi_label_classification":
        return "sigmoid"
    return "softmax"


def _valid_length(n) -> int | None:
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    return n if isinstance(n, int) and 2 <= n <= 1_000_000 else None


def resolve_max_length(tokenizer, config) -> int:
    """The model's real token budget, so runtime truncation matches validation.

    The config fallback subtracts 2 because some families (RoBERTa) declare
    max_position_embeddings larger than the usable budget (position ids start
    after padding_idx); losing two tokens is harmless, overflowing the position
    table is not.
    """
    n = _valid_length(getattr(tokenizer, "model_max_length", None))
    if n is None:
        cfg_n = _valid_length(getattr(config, "max_position_embeddings", None))
        n = cfg_n - 2 if cfg_n is not None and cfg_n > 4 else None
    return n if n is not None else 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("out")
    ap.add_argument(
        "--task",
        default="text-classification",
        choices=["text-classification", "token-classification"],
    )
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--normalization",
        choices=["softmax", "sigmoid"],
        help="Override the problem_type-inferred score normalization "
        "(for multi-label models that don't declare problem_type).",
    )
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )
    from optimum.onnxruntime import (
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    import onnxruntime as ort

    seq = args.task == "text-classification"
    trc = args.trust_remote_code
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=trc)
    RefCls = (
        AutoModelForSequenceClassification if seq else AutoModelForTokenClassification
    )
    ref = RefCls.from_pretrained(args.model_id, trust_remote_code=trc).eval()
    ORTCls = (
        ORTModelForSequenceClassification if seq else ORTModelForTokenClassification
    )
    ort_model = ORTCls.from_pretrained(
        args.model_id, export=True, trust_remote_code=trc
    )
    ort_model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    normalization = infer_normalization(ref.config, args.task, args.normalization)
    max_length = resolve_max_length(tokenizer, ref.config)
    normalize = sigmoid if normalization == "sigmoid" else softmax
    id2label = {int(k): v for k, v in ref.config.id2label.items()}
    if len(id2label) < 2:
        raise SystemExit("id2label must have at least 2 labels")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "id2label": id2label,
                "score_normalization": normalization,
                "token_aggregation": None if seq else "max",
                "max_length": max_length,
            },
            indent=2,
        )
    )

    # Validate: ONNX Runtime (CPU) vs the original PyTorch model on fixtures.
    # Scores are compared through the EXACT path ort-server serves (per-token
    # normalize + aggregate for token-cls; normalize for seq-cls). Raw logit
    # drift is also gated: normalized deltas alone are blind to drift in the
    # saturated regime, where softmax(20,-20) ~= softmax(15,-18).
    onnx_files = sorted(out.glob("*.onnx"))
    assert (
        len(onnx_files) == 1
    ), f"expected exactly one .onnx in {out}, found {onnx_files}"
    sess = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}

    def served_scores(logits: np.ndarray) -> np.ndarray:
        if seq:
            return normalize(logits)
        return normalize(logits).max(axis=0)  # ort-server default: max per label

    max_score_delta, max_logit_delta, argmax_agree = 0.0, 0.0, []
    for text in FIXTURES:
        enc_pt = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        with torch.no_grad():
            ref_logits = ref(**enc_pt).logits[0].numpy()
        assert ref_logits.shape[-1] == len(id2label), (
            f"model outputs {ref_logits.shape[-1]} logits but id2label has "
            f"{len(id2label)} entries"
        )
        enc_np = tokenizer(
            text, return_tensors="np", truncation=True, max_length=max_length
        )
        onnx_logits = sess.run(
            None, {k: v for k, v in enc_np.items() if k in in_names}
        )[0][0]
        max_logit_delta = max(
            max_logit_delta, float(np.abs(ref_logits - onnx_logits).max())
        )
        max_score_delta = max(
            max_score_delta,
            float(np.abs(served_scores(ref_logits) - served_scores(onnx_logits)).max()),
        )
        argmax_agree.append(
            float((ref_logits.argmax(-1) == onnx_logits.argmax(-1)).mean())
        )

    validation = {
        "compared_against": args.model_id,
        "fixtures": len(FIXTURES),
        "score_normalization": normalization,
        "max_length": max_length,
        "max_score_delta": round(max_score_delta, 6),
        "max_logit_delta": round(max_logit_delta, 6),
        "argmax_agreement": round(sum(argmax_agree) / len(argmax_agree), 6),
    }
    if not seq:
        validation["token_label_agreement"] = validation["argmax_agreement"]
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
