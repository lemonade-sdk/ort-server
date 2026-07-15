"""Generate tiny-roberta-tok: token-classification with a RobertaProcessing
post-processor (random weights, seeded).

A stock RoBERTa tokenizer serializes its inserted <s>/</s> as a RobertaProcessing
post-processor, which stores them in `cls`/`sep` fields — not in the
`post_processor.special_tokens` map that BERT's TemplateProcessing uses. A server
that reads only the TemplateProcessing shape never learns those ids are special,
so it leaves <s>/</s> in token-classification aggregation, which HuggingFace's
pipeline drops. This fixture makes <s>/</s> the sole driver of one label
(ENT_B ~ 1.0), so that behaviour fails the golden comparison by a wide margin.

    conda run -n lmxclf python test/fixtures/make_tiny_roberta_tok.py
"""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, pre_tokenizers, processors
from tokenizers.models import WordLevel

HERE = Path(__file__).parent
dst = HERE / "tiny-roberta-tok"
dst.mkdir(parents=True, exist_ok=True)

# RoBERTa id convention: <s>=0, <pad>=1, </s>=2, <unk>=3, <mask>=4.
SPECIALS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
CONTENT = ["hello", "world", "foo", "bar", "verify", "account", "the", "at", "again"]
vocab = {t: i for i, t in enumerate(SPECIALS + CONTENT)}
CLS_ID, SEP_ID = 0, 2

tk = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.add_special_tokens(SPECIALS)
tk.post_processor = processors.RobertaProcessing(sep=("</s>", SEP_ID), cls=("<s>", CLS_ID))
tk.save(str(dst / "tokenizer.json"))

VOCAB = len(vocab)


class TinyTok(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.emb = torch.nn.Embedding(VOCAB, 16)
        self.head = torch.nn.Linear(16, 3)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        return self.head(h)


m = TinyTok().eval()

# feature 0 fires for <s>/</s> ONLY and drives ENT_B (label 2) to ~1.0. A server
# that aggregates those positions — which HF filters out — then reports ENT_B~1.0
# and is caught by the golden comparison.
with torch.no_grad():
    m.emb.weight[:, 0] = 0.0
    for sid in (CLS_ID, SEP_ID):
        m.emb.weight[sid].zero_()
        m.emb.weight[sid][0] = 10.0
    m.head.weight[:, 0] = -5.0
    m.head.weight[2, 0] = 5.0

ids = torch.tensor([[CLS_ID, 5, 6, SEP_ID]])  # <s> hello world </s>
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

json.dump(
    {
        "model_type": "roberta",
        "architectures": ["RobertaForTokenClassification"],
        "id2label": {"0": "O", "1": "ENT_A", "2": "ENT_B"},
    },
    open(dst / "config.json", "w"),
    indent=2,
)

# Generic fast tokenizer so make_golden's AutoTokenizer loads from tokenizer.json
# and its return_special_tokens_mask reflects the RobertaProcessing insertions.
json.dump(
    {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "cls_token": "<s>",
        "sep_token": "</s>",
    },
    open(dst / "tokenizer_config.json", "w"),
    indent=2,
)

print("tiny-roberta-tok written, onnx bytes:", (dst / "model.onnx").stat().st_size)
