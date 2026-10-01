"""Generate tiny-router: a zero-shot-classification fixture (random weights, seeded).

Keeps only what the server must get right:

  * a BYTE-LEVEL tokenizer, so "é" or an emoji becomes several tokens sharing
    one character span; byte-offset handling fails the golden.
  * a post-processor prepending <|startoftext|> at span (0, 0), which both
    pooling grids must exclude.
  * a graph whose logit for label r is <pooled label r, pooled text>, so a
    mis-built or zero-filled grid changes the answer.

    conda run -n lmxclf python test/fixtures/make_tiny_router.py
"""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

HERE = Path(__file__).parent
dst = HERE / "tiny-router"
dst.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 192
BOS = "<|startoftext|>"

# Byte-level BPE with no merges: one token per byte, 257-entry vocab.
alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
vocab = {BOS: 0}
vocab.update({tok: i + 1 for i, tok in enumerate(alphabet)})
VOCAB = len(vocab)

tk = Tokenizer(BPE(vocab, [], unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tk.decoder = decoders.ByteLevel()
tk.add_special_tokens([BOS])
# Mirrors the real tokenizer: one inserted marker, reported at span (0, 0).
tk.post_processor = processors.TemplateProcessing(
    single=f"{BOS} $A", special_tokens=[(BOS, 0)]
)
# As in the real tokenizer.json; the server's default budget comes from here.
tk.enable_truncation(max_length=MAX_LENGTH)
tk.save(str(dst / "tokenizer.json"))

D = 16
SCALE = 3.0


class TinyRouter(torch.nn.Module):
    """logit[r] = scale * <pooled category r, pooled text>."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(23)
        self.emb = torch.nn.Embedding(VOCAB, D)

    def forward(self, input_ids, attention_mask, text_pool, category_pool):
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1).to(torch.float32)
        text_vec = torch.bmm(text_pool, h)          # [b, 1, d]
        cat_vec = torch.bmm(category_pool, h)       # [b, r, d]
        return (cat_vec * text_vec).sum(-1) * SCALE  # [b, r]


model = TinyRouter().eval()


# --- the reference algorithm, transcribed ----------------------------------
def build_inputs(text, routes):
    """Lfm2BidirForSequenceRouting.route(), up to the forward pass."""
    body = "\n".join(f"- {r}" for r in routes)
    prefix = f"Categories:\n{body}\n\nText:\n"

    ranges = []
    pos = len("Categories:\n")
    for r in routes:
        start = pos + 2  # skip "- "
        end = start + len(r)
        ranges.append((start, end))
        pos = end + 1  # the "\n"

    text_start = len(prefix)
    enc = tk.encode(prefix + text)
    offsets = enc.offsets
    n = len(enc.ids)

    text_idxs = [i for i in range(n)
                 if offsets[i][1] > text_start and offsets[i][0] != offsets[i][1]]
    assert text_idxs, "text was squeezed out of the window"
    text_pool = torch.zeros(1, 1, n)
    for i in text_idxs:
        text_pool[0, 0, i] = 1.0 / len(text_idxs)

    category_pool = torch.zeros(1, len(routes), n)
    for r, (start, end) in enumerate(ranges):
        idxs = [i for i in range(n)
                if offsets[i][0] < end and offsets[i][1] > start
                and offsets[i][0] != offsets[i][1]]
        assert idxs, f"route {routes[r]!r} was squeezed out of the window"
        for i in idxs:
            category_pool[0, r, i] = 1.0 / len(idxs)

    return (torch.tensor([enc.ids]), torch.ones(1, n, dtype=torch.int64),
            text_pool, category_pool)


def route(text, routes):
    with torch.no_grad():
        logits = model(*build_inputs(text, routes))
    probs = torch.softmax(logits[0], dim=-1)
    return {r: float(probs[i]) for i, r in enumerate(routes)}


ids, mask, tp, cp = build_inputs("hello world", ["code", "chat"])
torch.onnx.export(
    model,
    (ids, mask, tp, cp),
    str(dst / "model.onnx"),
    input_names=["input_ids", "attention_mask", "text_pool", "category_pool"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "b", 1: "s"},
        "attention_mask": {0: "b", 1: "s"},
        "text_pool": {0: "b", 2: "s"},
        "category_pool": {0: "b", 1: "r", 2: "s"},
        "logits": {0: "b", 1: "r"},
    },
    opset_version=17,
    dynamo=False,
)

json.dump(
    {"task": "zero-shot-classification", "max_length": MAX_LENGTH},
    open(dst / "manifest.json", "w"),
    indent=2,
)
json.dump(
    {
        "model_type": "lfm2",
        "architectures": ["Lfm2BidirForSequenceRouting"],
        # As in the real config: far larger than the tokenizer's window.
        "max_position_embeddings": 128000,
    },
    open(dst / "config.json", "w"),
    indent=2,
)
json.dump(
    {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": BOS,
     "model_max_length": int(1e30)},
    open(dst / "tokenizer_config.json", "w"),
    indent=2,
)

CASES = [
    # Plain ASCII, two labels.
    ("hello world", ["code", "chat"]),
    # Three labels, one a prefix of another: the ranges are what separates them,
    # not substring search.
    ("write a sorting function", ["code", "code generation", "small talk"]),
    # Non-ASCII and emoji in BOTH the labels and the text: every one of these
    # characters is several tokens sharing one character span.
    ("déjà vu 🚀 again", ["café", "🚀 launch", "plain"]),
    # Punctuation that survives byte-level pre-tokenization unsplit.
    ("fix my segfault", ["C++", "python", "rust"]),
    # Trailing and leading spaces are part of the label and its range.
    ("spaces matter", ["trailing ", " leading", "neither"]),
    # A single label: softmax over one logit is 1.0 whatever the model says.
    ("anything", ["only"]),
    # Text that looks like the template itself must not confuse the ranges --
    # membership comes from offsets, not from searching for "Text:".
    ("Categories:\n- decoy\n\nText:\nreal", ["decoy", "genuine"]),
]

json.dump(
    {"cases": [{"text": t, "labels": ls, "scores": route(t, ls)} for t, ls in CASES]},
    open(dst / "golden.json", "w"),
    indent=2,
    ensure_ascii=False,
)

print("tiny-router written, onnx bytes:", (dst / "model.onnx").stat().st_size)
for t, ls in CASES:
    print(f"  {t[:28]!r:32} -> " +
          ", ".join(f"{k}={v:.3f}" for k, v in route(t, ls).items()))
