"""Generate the tiny-tok token-classification fixture (random weights, seeded).

Reuses tiny-clf's tokenizer files; emits a [batch, seq, 3] logits graph.
    conda run -n lmxclf python test/fixtures/make_tiny_tok.py
"""

import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).parent
src = HERE / "tiny-clf"
dst = HERE / "tiny-tok"
dst.mkdir(parents=True, exist_ok=True)
for f in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
):
    shutil.copy(src / f, dst / f)

VOCAB = sum(1 for _ in open(src / "vocab.txt", encoding="utf-8"))


class TinyTok(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.emb = torch.nn.Embedding(VOCAB, 16)
        self.head = torch.nn.Linear(16, 3)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        return self.head(h)


m = TinyTok().eval()

# Make the special tokens ([CLS]=2, [SEP]=3) score ENT_B (label 2) near 1.0,
# while content tokens stay middling. A server that aggregates special-token
# positions — which the HuggingFace pipeline filters out — then reports
# ENT_B≈1.0 and is caught by the golden comparison. Without this the fixture
# cannot tell the two behaviours apart.
with torch.no_grad():
    m.emb.weight[:, 0] = 0.0  # feature 0 fires for special tokens ONLY
    for special_id in (2, 3):
        m.emb.weight[special_id].zero_()
        m.emb.weight[special_id][0] = 10.0
    m.head.weight[:, 0] = -5.0
    m.head.weight[2, 0] = 5.0  # feature 0 -> ENT_B

ids = torch.tensor([[2, 5, 9, 3]])
mask = torch.ones_like(ids)
torch.onnx.export(
    m,
    (ids, mask),
    str(dst / "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "b", 1: "s"},
        "attention_mask": {0: "b", 1: "s"},
        "logits": {0: "b", 1: "s"},
    },
    opset_version=17,
    dynamo=False,
)

json.dump(
    {
        "task": "token-classification",
        "id2label": {"0": "O", "1": "ENT_A", "2": "ENT_B"},
        "score_normalization": "softmax",
        "token_aggregation": "max",
        "max_length": 512,
    },
    open(dst / "manifest.json", "w"),
    indent=2,
)

cfg = json.load(open(src / "config.json"))
cfg["architectures"] = ["DistilBertForTokenClassification"]
cfg["id2label"] = {"0": "O", "1": "ENT_A", "2": "ENT_B"}
json.dump(cfg, open(dst / "config.json", "w"), indent=2)
print("tiny-tok fixture written, onnx bytes:", (dst / "model.onnx").stat().st_size)
