// ort-server — generic ONNX Runtime model server for Lemonade (see README).
//
// CPU EP. Serves three shapes of model from a plain exported ONNX graph plus
// the model's own HuggingFace tokenizer.json:
//
//   text-classification   input_ids/attention_mask[/token_type_ids] -> [1, L]
//   token-classification  same inputs                              -> [1, T, L]
//   zero-shot-classification
//                         input_ids/attention_mask/text_pool/category_pool
//                                                                  -> [1, R]
//
// The first two take their label list from manifest.json (or infer the whole
// contract from a stock export's config.json) and answer /classify with a score
// per label. The third takes its labels from the REQUEST — the caller sends a
// `labels` array and the server builds the two pooling grids the model needs to
// score them (docs/ZERO-SHOT-ROUTER.md).

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

// third_party/tok_ffi: our own C ABI over the HuggingFace `tokenizers` crate.
// It returns character offsets alongside the ids, which the zero-shot path uses
// to work out which tokens belong to which label.
#include "tok_ffi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// A bad request, as opposed to a server/model fault: mapped to 400.
struct InvalidInput : std::runtime_error {
    using std::runtime_error::runtime_error;
};

constexpr const char* kZeroShot = "zero-shot-classification";

constexpr size_t kDefaultMaxGridCells = 32u << 20;

// Smallest number of tokens a label can occupy in the prompt
constexpr int kMinTokensPerLabel = 4;

// Tokens held back for the user's text when the label block is budgeted.
constexpr size_t kMinTextTokens = 8;

// Floor for the vocabulary-derived "characters one token can cover" bound.
constexpr size_t kMinTokenCharsBound = 32;

// Written in place of the middle of a text that did not fit, so the model can
// see that something is missing.
constexpr const char* kElision = "\n...\n";

// Share of the text budget given to the head; the rest goes to the tail.
constexpr size_t kTextHeadDivisor = 3;

struct Manifest {
    // "text-classification" | "token-classification" | "zero-shot-classification"
    std::string task;
    std::vector<std::string> id2label;  // index -> label; EMPTY for zero-shot
    std::string score_normalization = "softmax";  // "softmax" | "sigmoid"
    std::string token_aggregation = "max";        // token-cls only; "max" | "mean"
    int max_length = 512;               // token budget; longer inputs are truncated

    // Zero-shot request limits. 0 means "derive from max_length"
    int max_labels = 0;
    int max_label_chars = 0;
    long long max_grid_cells = 0;

    bool zero_shot() const { return task == kZeroShot; }
};

void apply_zero_shot_limits(Manifest& m) {
    if (!m.zero_shot()) return;
    if (m.max_labels <= 0) {
        m.max_labels = std::max(1, m.max_length / kMinTokensPerLabel);
    }
    // One label cannot be longer than the whole window.
    if (m.max_label_chars <= 0) m.max_label_chars = m.max_length;
    if (m.max_grid_cells <= 0) {
        m.max_grid_cells = static_cast<long long>(kDefaultMaxGridCells);
    }
}

struct LimitOverrides {
    int max_labels = 0;
    int max_label_chars = 0;
    long long max_grid_cells = 0;
};

struct Args {
    std::string model_path;
    int port = 0;
    bool verbose = false;
    LimitOverrides limits;
};

long long positive_arg(const std::string& flag, const char* raw,
                       long long hi = std::numeric_limits<long long>::max()) {
    try {
        size_t used = 0;
        const std::string s(raw);
        const long long v = std::stoll(s, &used);
        if (used != s.size()) throw std::invalid_argument("trailing characters");
        if (v < 1 || v > hi) throw std::out_of_range("out of range");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(flag + " needs an integer between 1 and " +
                                 std::to_string(hi) + ", got '" +
                                 std::string(raw) + "'");
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "--model-path" && i + 1 < argc) a.model_path = argv[++i];
        else if (f == "--port" && i + 1 < argc) a.port = std::stoi(argv[++i]);
        else if (f == "--verbose") a.verbose = true;
        else if (f == "--max-labels" && i + 1 < argc)
            a.limits.max_labels = static_cast<int>(
                positive_arg(f, argv[++i], std::numeric_limits<int>::max()));
        else if (f == "--max-label-chars" && i + 1 < argc)
            a.limits.max_label_chars = static_cast<int>(
                positive_arg(f, argv[++i], std::numeric_limits<int>::max()));
        else if (f == "--max-grid-cells" && i + 1 < argc)
            a.limits.max_grid_cells = positive_arg(f, argv[++i]);
    }
    if (a.model_path.empty() || a.port == 0) {
        throw std::runtime_error(
            "usage: ort-server --model-path <dir> --port <n> [--verbose]\n"
            "       [--max-labels <n>] [--max-label-chars <n>] [--max-grid-cells <n>]\n"
            "  The three limits apply to zero-shot-classification models only. Each\n"
            "  overrides the matching manifest.json field; omit to take the manifest's\n"
            "  value, or the default derived from the model's token window.");
    }
    return a;
}

// Number of Unicode characters in a UTF-8 string: count every byte that is not
// a continuation byte (10xxxxxx)
size_t utf8_len(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++n;
    return n;
}

// Byte offset of character index `chars`, or s.size() if the string is shorter.
size_t utf8_byte_at(const std::string& s, size_t chars) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
            if (n == chars) return i;
            ++n;
        }
    }
    return s.size();
}

std::string utf8_range(const std::string& s, size_t from, size_t to) {
    if (to <= from) return {};
    const size_t b = utf8_byte_at(s, from);
    const size_t e = utf8_byte_at(s, to);
    return s.substr(b, e - b);
}

std::string utf8_head_tail(const std::string& s, size_t max_chars) {
    const size_t total = utf8_len(s);
    if (total <= max_chars) return s;
    const size_t head = max_chars / kTextHeadDivisor;
    const size_t tail = max_chars - head;
    return utf8_range(s, 0, head) + utf8_range(s, total - tail, total);
}

void parse_id2label(const json& id2label, Manifest& m, const std::string& origin) {
    if (!id2label.is_object() || id2label.empty()) {
        throw std::runtime_error("id2label is missing or empty in " + origin);
    }
    m.id2label.resize(id2label.size());
    std::vector<bool> seen(id2label.size(), false);
    for (auto it = id2label.begin(); it != id2label.end(); ++it) {
        size_t pos = 0;
        unsigned long idx = 0;
        try {
            idx = std::stoul(it.key(), &pos);
        } catch (const std::exception&) {
            pos = 0;
        }
        if (pos != it.key().size()) {
            throw std::runtime_error("id2label key '" + it.key() + "' is not an index in " + origin);
        }
        if (idx >= m.id2label.size() || seen[idx]) {
            throw std::runtime_error("id2label keys must be unique and contiguous 0..n-1 in " + origin);
        }
        if (!it.value().is_string()) {
            throw std::runtime_error("id2label values must be strings in " + origin);
        }
        seen[idx] = true;
        m.id2label[idx] = it.value().get<std::string>();
    }
}

json read_json_if_present(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return json::object();
    try {
        json j; f >> j;
        return j.is_object() ? j : json::object();
    } catch (const std::exception&) {
        return json::object();
    }
}

// TWO allowlists, selected by task. The classifier tasks feed the model a
// single sequence with a fabricated all-ones attention mask and all-zero
// token_type_ids, and truncate by keeping the trailing token. That is exactly
// right for BERT-family single-sequence encoders and wrong for architectures
// with different segment/special-token conventions (XLNet puts its classifier
// token last with a distinct segment id). The zero-shot task shares none of
// them, so listing its architectures here would silently promise them the
// encoder treatment. See docs/ZERO-SHOT-ROUTER.md.
const std::set<std::string>& supported_model_types(const std::string& task) {
    static const std::set<std::string> kEncoder = {
        "albert", "bert",     "camembert",  "deberta", "deberta-v2",
        "distilbert", "electra", "roberta", "xlm-roberta", "modernbert",
        "openai_privacy_filter", "pii_masking"
    };
    static const std::set<std::string> kZeroShotTypes = {"lfm2"};
    return task == kZeroShot ? kZeroShotTypes : kEncoder;
}

void validate_model_family(const json& config, const fs::path& dir, const std::string& task) {
    // A manifest describes the OUTPUT contract (labels, normalization). It says
    // nothing about the INPUT convention — attention mask, segment ids, special
    // tokens — which is what this server hardcodes. So a manifest cannot excuse
    // a missing config.json: without it we cannot know the architecture, and an
    // unchecked one would be served with a fabricated mask that may not fit.
    if (config.empty()) {
        throw std::runtime_error(
            "config.json is missing or unreadable in " + dir.string() +
            ". It is required (even alongside a manifest.json) to confirm the "
            "model uses an input convention this server implements.");
    }
    std::string model_type;
    if (config.contains("model_type") && config["model_type"].is_string()) {
        model_type = config["model_type"].get<std::string>();
    }
    if (model_type.empty()) {
        throw std::runtime_error("config.json in " + dir.string() +
                                 " declares no model_type; cannot verify that this "
                                 "architecture uses an input convention ort-server "
                                 "implements");
    }
    const auto& allowed = supported_model_types(task);
    if (!allowed.count(model_type)) {
        std::string supported;
        for (const auto& t : allowed) supported += (supported.empty() ? "" : ", ") + t;
        if (task == kZeroShot) {
            throw std::runtime_error(
                "unsupported model_type '" + model_type + "' for task " + task +
                ". This task feeds the model explicit text_pool/category_pool "
                "grids built from tokenizer character offsets, which is valid "
                "for: " + supported +
                ". Encoder classifiers belong under text-classification.");
        }
        throw std::runtime_error(
            "unsupported model_type '" + model_type +
            "'. ort-server implements the single-sequence encoder convention "
            "(all-ones attention mask, all-zero token_type_ids, trailing-token "
            "truncation), which is valid for: " + supported +
            ". Other architectures need their own mask/segment handling.");
    }
}

// max_length precedence mirrors the exporter: the tokenizer's declared budget,
// then the model's position table (less 2 — RoBERTa-family configs declare
// max_position_embeddings larger than the usable budget), then 512.
void apply_inferred_max_length(const json& tokenizer_config, const json& config, Manifest& m) {
    auto valid = [](const json& j, const char* key) -> int {
        // HF writes a huge sentinel (1e30) when the tokenizer has no real limit;
        // that parses as a double and is skipped by the integer check.
        if (!j.contains(key) || !j[key].is_number_integer()) return 0;
        auto n = j[key].get<long long>();
        return (n >= 2 && n <= 1000000) ? static_cast<int>(n) : 0;
    };
    if (int n = valid(tokenizer_config, "model_max_length")) {
        m.max_length = n;
        return;
    }
    if (int n = valid(config, "max_position_embeddings")) {
        m.max_length = n > 4 ? n - 2 : n;
        return;
    }
    m.max_length = 512;
}

int tokenizer_truncation_max_length(const json& tokenizer_json) {
    if (!tokenizer_json.contains("truncation") || !tokenizer_json["truncation"].is_object()) {
        return 0;
    }
    const json& t = tokenizer_json["truncation"];
    if (!t.contains("max_length") || !t["max_length"].is_number_integer()) return 0;
    auto n = t["max_length"].get<long long>();
    return (n >= 2 && n <= 1000000) ? static_cast<int>(n) : 0;
}

void apply_zero_shot_max_length(const json& tokenizer_json, const fs::path& dir, Manifest& m) {
    if (int n = tokenizer_truncation_max_length(tokenizer_json)) {
        m.max_length = n;
        return;
    }
    throw std::runtime_error(
        "cannot determine the token budget for a " + std::string(kZeroShot) +
        " model in " + dir.string() +
        ": tokenizer.json declares no truncation.max_length. Set \"max_length\" "
        "in manifest.json explicitly");
}

Manifest manifest_from_json(const fs::path& dir, const json& tokenizer_json) {
    std::ifstream f(dir / "manifest.json");
    if (!f) throw std::runtime_error("cannot open manifest.json in " + dir.string());
    json j; f >> j;
    Manifest m;
    m.task = j.at("task").get<std::string>();
    if (m.task != "text-classification" && m.task != "token-classification" &&
        m.task != kZeroShot) {
        throw std::runtime_error("unsupported task in manifest.json: '" + m.task +
                                 "' (expected text-classification, token-classification "
                                 "or " + kZeroShot + ")");
    }
    // Even with an explicit manifest, the tokenization/mask conventions for this
    // task still have to hold for this architecture.
    validate_model_family(read_json_if_present(dir / "config.json"), dir, m.task);

    // Wrong-typed values are errors, not silent fallbacks to defaults.
    if (j.contains("score_normalization")) {
        if (!j["score_normalization"].is_string()) {
            throw std::runtime_error("score_normalization must be a string");
        }
        m.score_normalization = j["score_normalization"].get<std::string>();
    }
    // "none" is rejected: the /classify contract promises label scores in [0,1].
    if (m.score_normalization != "softmax" && m.score_normalization != "sigmoid") {
        throw std::runtime_error("unsupported score_normalization: " + m.score_normalization);
    }
    // token_aggregation is null for sequence-classification; tolerate null/absent,
    // but reject unknown values regardless of task.
    if (j.contains("token_aggregation") && !j["token_aggregation"].is_null()) {
        if (!j["token_aggregation"].is_string()) {
            throw std::runtime_error("token_aggregation must be a string or null");
        }
        m.token_aggregation = j["token_aggregation"].get<std::string>();
        if (m.token_aggregation != "max" && m.token_aggregation != "mean") {
            throw std::runtime_error("unsupported token_aggregation: " + m.token_aggregation);
        }
    }
    if (j.contains("max_length")) {
        if (!j["max_length"].is_number_integer()) {
            throw std::runtime_error("max_length must be an integer");
        }
        const long long v = j["max_length"].get<long long>();
        if (v < 2 || v > std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "max_length must be between 2 and " +
                std::to_string(std::numeric_limits<int>::max()));
        }
        m.max_length = static_cast<int>(v);
    } else if (m.zero_shot()) {
        apply_zero_shot_max_length(tokenizer_json, dir, m);
    } else {
        // Manifest omits the budget: fall back to the model's own metadata
        // rather than a blanket 512, which can overflow a smaller position table.
        apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"),
                                  read_json_if_present(dir / "config.json"), m);
    }

    for (const char* field : {"max_labels", "max_label_chars", "max_grid_cells"}) {
        if (!j.contains(field) || j[field].is_null()) continue;
        if (!m.zero_shot()) {
            throw std::runtime_error(std::string(field) + " only applies to " +
                                     kZeroShot + "; remove it from manifest.json");
        }
        if (!j[field].is_number_integer()) {
            throw std::runtime_error(std::string(field) + " must be an integer");
        }
        const long long v = j[field].get<long long>();
        const bool narrows = std::string(field) != "max_grid_cells";
        const long long hi = narrows
                                 ? static_cast<long long>(std::numeric_limits<int>::max())
                                 : std::numeric_limits<long long>::max();
        if (v < 1 || v > hi) {
            throw std::runtime_error(std::string(field) + " must be between 1 and " +
                                     std::to_string(hi));
        }
        if (std::string(field) == "max_labels") m.max_labels = static_cast<int>(v);
        else if (std::string(field) == "max_label_chars") m.max_label_chars = static_cast<int>(v);
        else m.max_grid_cells = v;
    }

    if (m.zero_shot()) {
        if (j.contains("id2label") && !j["id2label"].is_null()) {
            throw std::runtime_error(
                std::string(kZeroShot) + " takes its labels from each request; "
                "remove id2label from manifest.json");
        }
        if (m.score_normalization != "softmax") {
            throw std::runtime_error(std::string(kZeroShot) +
                                     " scores one label set against one text; "
                                     "score_normalization must be softmax");
        }
        return m;
    }
    parse_id2label(j.at("id2label"), m, "manifest.json");
    return m;
}

// Fallback for a stock HF/Optimum export (no manifest.json): infer the contract
// from config.json (+ tokenizer_config.json), applying HF problem_type semantics.
Manifest manifest_from_hf_config(const fs::path& dir, const json& tokenizer_json) {
    std::ifstream f(dir / "config.json");
    if (!f) {
        throw std::runtime_error("neither manifest.json nor config.json found in " +
                                 dir.string());
    }
    json j; f >> j;
    Manifest m;

    std::string arch;
    if (j.contains("architectures") && j["architectures"].is_array() &&
        !j["architectures"].empty() && j["architectures"][0].is_string()) {
        arch = j["architectures"][0].get<std::string>();
    }
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with(arch, "ForSequenceRouting") || ends_with(arch, "ForZeroShotClassification")) {
        m.task = kZeroShot;
    } else if (ends_with(arch, "ForTokenClassification")) {
        m.task = "token-classification";
    } else if (ends_with(arch, "ForSequenceClassification")) {
        m.task = "text-classification";
    } else {
        throw std::runtime_error("cannot infer task from config.json architecture '" +
                                 arch + "'; provide a manifest.json");
    }
    validate_model_family(j, dir, m.task);

    if (m.zero_shot()) {
        m.score_normalization = "softmax";
        apply_zero_shot_max_length(tokenizer_json, dir, m);
        return m;
    }

    // problem_type is only a training-time hint: models trained with BCE outside
    // the HF Trainer routinely leave it null, so an absent value CANNOT be read
    // as "single-label". Manifest-less inference therefore ASSUMES single-label
    // softmax and says so; a multi-label model must declare it — either via
    // problem_type in its config, or with an explicit manifest.json.
    std::string problem_type;
    if (j.contains("problem_type") && j["problem_type"].is_string()) {
        problem_type = j["problem_type"].get<std::string>();
    }
    if (problem_type == "regression") {
        throw std::runtime_error("regression heads have no label scores in [0,1]");
    }
    if (m.task == "text-classification" && problem_type == "multi_label_classification") {
        m.score_normalization = "sigmoid";
    } else {
        m.score_normalization = "softmax";
        if (m.task == "text-classification" && problem_type.empty()) {
            fprintf(stderr,
                    "ort-server: config.json declares no problem_type; assuming "
                    "SINGLE-LABEL softmax. If this is a multi-label (BCE-trained) "
                    "model, supply a manifest.json with "
                    "\"score_normalization\": \"sigmoid\" — otherwise the scores "
                    "will be wrong.\n");
        }
    }

    parse_id2label(j.at("id2label"), m, "config.json");
    if (m.id2label.size() < 2) {
        throw std::runtime_error("single-output heads have no label scores in [0,1]");
    }
    apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"), j, m);
    return m;
}

// manifest.json (explicit contract, validated strictly) wins; a bare Optimum
// export runs via config.json inference so users need no lemonade tooling.
Manifest load_manifest(const fs::path& dir, const json& tokenizer_json,
                       const LimitOverrides& cli) {
    Manifest m = fs::exists(dir / "manifest.json")
                     ? manifest_from_json(dir, tokenizer_json)
                     : manifest_from_hf_config(dir, tokenizer_json);

    // Command line beats manifest beats derived default.
    const bool any_cli = cli.max_labels > 0 || cli.max_label_chars > 0 ||
                         cli.max_grid_cells > 0;
    if (any_cli && !m.zero_shot()) {
        throw std::runtime_error(
            "--max-labels / --max-label-chars / --max-grid-cells apply to " +
            std::string(kZeroShot) + " models only; this model's task is " + m.task);
    }
    if (cli.max_labels > 0) m.max_labels = cli.max_labels;
    if (cli.max_label_chars > 0) m.max_label_chars = cli.max_label_chars;
    if (cli.max_grid_cells > 0) m.max_grid_cells = cli.max_grid_cells;

    apply_zero_shot_limits(m);
    return m;
}

std::vector<float> softmax(const float* v, size_t n) {
    float mx = *std::max_element(v, v + n);
    std::vector<float> out(n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) { out[i] = std::exp(v[i] - mx); sum += out[i]; }
    for (auto& x : out) x = static_cast<float>(x / sum);
    return out;
}

std::vector<float> normalize(const float* v, size_t n, const std::string& mode) {
    if (mode == "softmax") return softmax(v, n);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-v[i]));
    return out;
}

std::string load_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Collect the token ids the post-processor INSERTS ([CLS]/[SEP], <s>/</s>, …).
// The HF tokenizer serializes these three ways, and a "Sequence" can nest them:
//   TemplateProcessing  -> special_tokens[*].ids   (modern BERT/DistilBERT)
//   BertProcessing      -> cls / sep = [token, id]
//   RobertaProcessing   -> cls / sep = [token, id]  (stock RoBERTa; different shape)
// Reading only TemplateProcessing would silently miss RoBERTa's <s>/</s>, leaving
// them in token-classification aggregation that HF's pipeline drops.
void collect_inserted_special_ids(const json& pp, std::set<int64_t>& out) {
    if (!pp.is_object()) return;
    const std::string type = pp.value("type", "");
    if (type == "Sequence" && pp.contains("processors") && pp["processors"].is_array()) {
        for (const auto& child : pp["processors"]) collect_inserted_special_ids(child, out);
        return;
    }
    if (pp.contains("special_tokens") && pp["special_tokens"].is_object()) {
        for (const auto& entry : pp["special_tokens"]) {
            if (!entry.is_object() || !entry.contains("ids")) continue;
            for (const auto& id : entry["ids"]) {
                if (id.is_number_integer()) out.insert(id.get<int64_t>());
            }
        }
    }
    for (const char* field : {"cls", "sep"}) {
        if (pp.contains(field) && pp[field].is_array() && pp[field].size() == 2 &&
            pp[field][1].is_number_integer()) {
            out.insert(pp[field][1].get<int64_t>());
        }
    }
}

// --- model input binding ----------------------------------------------------
// Everything a single request feeds the graph. Tensors alias these buffers, so
// a Bindings must outlive the Run that uses it.
struct Bindings {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    std::vector<float> text_pool;      // [1, 1, S]
    std::vector<float> category_pool;  // [1, R, S]
    int64_t seq_len = 0;
    int64_t num_labels = 0;            // R; zero-shot only
};

template <typename T>
Ort::Value make_tensor(std::vector<T>& data, std::vector<int64_t> shape,
                       const Ort::MemoryInfo& mem) {
    return Ort::Value::CreateTensor<T>(mem, data.data(), data.size(),
                                       shape.data(), shape.size());
}

using TensorBuilder = Ort::Value (*)(Bindings&, const Ort::MemoryInfo&);

Ort::Value build_input_ids(Bindings& b, const Ort::MemoryInfo& mem) {
    return make_tensor(b.input_ids, {1, b.seq_len}, mem);
}
Ort::Value build_attention_mask(Bindings& b, const Ort::MemoryInfo& mem) {
    return make_tensor(b.attention_mask, {1, b.seq_len}, mem);
}
Ort::Value build_token_type_ids(Bindings& b, const Ort::MemoryInfo& mem) {
    return make_tensor(b.token_type_ids, {1, b.seq_len}, mem);
}
Ort::Value build_text_pool(Bindings& b, const Ort::MemoryInfo& mem) {
    return make_tensor(b.text_pool, {1, 1, b.seq_len}, mem);
}
Ort::Value build_category_pool(Bindings& b, const Ort::MemoryInfo& mem) {
    return make_tensor(b.category_pool, {1, b.num_labels, b.seq_len}, mem);
}

struct InputSpec {
    const char* name;
    bool required;  // absent-but-optional is fine (DistilBERT has no token_type_ids)
    TensorBuilder build;
};

const std::vector<InputSpec>& input_specs(const std::string& task) {
    static const std::vector<InputSpec> kEncoderInputs = {
        {"input_ids", true, build_input_ids},
        {"attention_mask", true, build_attention_mask},
        {"token_type_ids", false, build_token_type_ids},
    };
    static const std::vector<InputSpec> kZeroShotInputs = {
        {"input_ids", true, build_input_ids},
        {"attention_mask", true, build_attention_mask},
        {"text_pool", true, build_text_pool},
        {"category_pool", true, build_category_pool},
    };
    return task == kZeroShot ? kZeroShotInputs : kEncoderInputs;
}

// One tokenization: ids plus the per-token character span they came from.
struct Encoded {
    std::vector<int64_t> ids;
    std::vector<uint32_t> starts;
    std::vector<uint32_t> ends;

    size_t size() const { return ids.size(); }
};

size_t head_chars_for(const Encoded& t, size_t budget) {
    size_t h = std::min(budget, t.size());
    while (h > 0 && h < t.size() && t.ends[h - 1] == t.ends[h]) --h;
    return h > 0 ? static_cast<size_t>(t.ends[h - 1]) : 0;
}

size_t tail_chars_for(const Encoded& t, size_t budget, size_t total_chars) {
    if (budget == 0 || t.size() == 0) return total_chars;
    size_t i = t.size() - std::min(budget, t.size());
    while (i < t.size() && i > 0 && t.starts[i] == t.starts[i - 1]) ++i;
    return i < t.size() ? static_cast<size_t>(t.starts[i]) : total_chars;
}

class Model {
public:
    Model(const fs::path& dir, bool verbose, const LimitOverrides& cli = {})
        : env_(ORT_LOGGING_LEVEL_WARNING, "ort-server"), verbose_(verbose) {
        std::string blob = load_bytes(dir / "tokenizer.json");

        // Parse BEFORE handing the blob to the Rust tokenizer so a truncated or
        // corrupt tokenizer.json (a partial download, say) gets a message that
        // says so. tok_new returns null rather than panicking, but "failed to
        // load" alone would not point at the real cause.
        json tj;
        try {
            tj = json::parse(blob);
        } catch (const std::exception& e) {
            throw std::runtime_error("tokenizer.json in " + dir.string() +
                                     " is not valid JSON (truncated or corrupt "
                                     "download?): " + e.what());
        }

        manifest_ = load_manifest(dir, tj, cli);

        tokenizer_.reset(tok_new(reinterpret_cast<const uint8_t*>(blob.data()), blob.size()));
        if (!tokenizer_) throw std::runtime_error("failed to load tokenizer.json from " + dir.string());

        // Positions the tokenizer INSERTS ([CLS]/[SEP] from the post-processor
        // template, plus padding). They belong in the model's INPUT — sequence
        // classifiers pool [CLS] — but HuggingFace's token-classification
        // pipeline drops them from its OUTPUT, so aggregating them would report
        // entities the reference never does.
        //
        // Deliberately NOT every `added_token` marked special: [UNK] and [MASK]
        // are special *tokens* but appear as ordinary content positions (an
        // out-of-vocabulary word becomes [UNK] and HF still scores it), and
        // HF's special_tokens_mask marks them 0.
        if (tj.contains("post_processor")) {
            collect_inserted_special_ids(tj["post_processor"], special_ids_);
        }

        // Backstop only: tok_new clears padding (see tok_ffi/src/lib.rs).
        if (tj.contains("padding") && tj["padding"].is_object() &&
            tj["padding"].contains("pad_id") &&
            tj["padding"]["pad_id"].is_number_integer()) {
            pad_id_ = tj["padding"]["pad_id"].get<int64_t>();
            special_ids_.insert(pad_id_);
        }

        if (manifest_.zero_shot()) {
            size_t longest = 0;
            if (tj.contains("model") && tj["model"].is_object() &&
                tj["model"].contains("vocab")) {
                const json& v = tj["model"]["vocab"];
                if (v.is_object()) {  // BPE / WordPiece / WordLevel: {token: id}
                    for (auto it = v.begin(); it != v.end(); ++it) {
                        longest = std::max(longest, utf8_len(it.key()));
                    }
                } else if (v.is_array()) {  // Unigram: [[token, score], ...]
                    for (const auto& e : v) {
                        if (e.is_array() && !e.empty() && e[0].is_string()) {
                            longest = std::max(longest, utf8_len(e[0].get<std::string>()));
                        }
                    }
                }
            }
            max_token_chars_ = std::max(longest, kMinTokenCharsBound);

            Encoded empty = encode("", /*add_special_tokens=*/true);
            size_t k = empty.size();
            if (pad_id_ >= 0) {
                while (k > 0 && empty.ids[k - 1] == pad_id_) --k;
            }
            n_specials_ = k;

            elision_tokens_ = encode(kElision, /*add_special_tokens=*/false).size();
        }

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(0);
        session_ = Ort::Session(env_, (dir / "model.onnx").c_str(), opts);

        size_t n_in = session_.GetInputCount();
        for (size_t i = 0; i < n_in; ++i) {
            input_names_.push_back(session_.GetInputNameAllocated(i, alloc_).get());
        }
        size_t n_out = session_.GetOutputCount();
        for (size_t i = 0; i < n_out; ++i) {
            output_names_.push_back(session_.GetOutputNameAllocated(i, alloc_).get());
        }
        bind_inputs();
    }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool zero_shot() const { return manifest_.zero_shot(); }

    // Single entry point for /classify: the presence of a request `labels` list
    // has to agree with what the loaded model is, in both directions.
    json handle(const std::string& text, const std::vector<std::string>* labels, int top_k) {
        if (zero_shot() && !labels) {
            throw InvalidInput(
                "this model scores a label list supplied per request; include a "
                "\"labels\" array in the request body");
        }
        if (!zero_shot() && labels) {
            throw InvalidInput(
                "this model has a fixed label set declared by its manifest; "
                "\"labels\" is not accepted");
        }
        return zero_shot() ? route(text, *labels, top_k) : classify(text, top_k);
    }

private:
    // Check the graph's declared inputs against the task's table once, at load.
    void bind_inputs() {
        const auto& specs = input_specs(manifest_.task);
        std::string known;
        for (const auto& s : specs) known += (known.empty() ? "" : ", ") + std::string(s.name);

        std::set<std::string> declared;
        for (const auto& name : input_names_) {
            const InputSpec* found = nullptr;
            for (const auto& s : specs) {
                if (name == s.name) { found = &s; break; }
            }
            if (!found) {
                throw std::runtime_error(
                    "model declares input '" + name + "', which task " + manifest_.task +
                    " does not know how to build. Expected inputs: " + known);
            }
            bound_.push_back(found);
            declared.insert(name);
        }
        for (const auto& s : specs) {
            if (s.required && !declared.count(s.name)) {
                throw std::runtime_error(
                    "model is missing required input '" + std::string(s.name) +
                    "' for task " + manifest_.task + ". Expected inputs: " + known);
            }
        }
    }

    // One tokenization of `text`, with the character span of every token.
    // Special tokens default ON: encoder classifiers pool [CLS]/use [SEP], the
    // catalog's parity validation runs the HF tokenizer with them enabled, and
    // the router's reference route() leaves the default on too - serving
    // without them would silently diverge.
    // No lock: tokenizers::Tokenizer is Sync and tok_encode borrows it shared.
    Encoded encode(const std::string& text, bool add_special_tokens = true) const {
        TokEncoding enc{};
        if (!tok_encode(tokenizer_.get(), reinterpret_cast<const uint8_t*>(text.data()),
                        text.size(), add_special_tokens, &enc)) {
            throw std::runtime_error("tokenization failed");
        }
        Encoded out;
        out.ids.assign(enc.ids, enc.ids + enc.len);
        out.starts.assign(enc.starts, enc.starts + enc.len);
        out.ends.assign(enc.ends, enc.ends + enc.len);
        tok_encoding_free(&enc);
        return out;
    }

    std::vector<Ort::Value> run(Bindings& b) {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        std::vector<const char*> in_names;
        inputs.reserve(bound_.size());
        for (size_t i = 0; i < bound_.size(); ++i) {
            inputs.push_back(bound_[i]->build(b, mem));
            in_names.push_back(input_names_[i].c_str());
        }
        std::vector<const char*> out_names;
        for (const auto& n : output_names_) out_names.push_back(n.c_str());
        return session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                            inputs.size(), out_names.data(), out_names.size());
    }

    static const float* float_logits(const Ort::Value& v, std::vector<int64_t>& shape) {
        auto type_info = v.GetTensorTypeAndShapeInfo();
        if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "model output must be float32 logits (fp16/quantized-output exports are not supported)");
        }
        shape = type_info.GetShape();
        return v.GetTensorData<float>();
    }

    static json rank(std::vector<std::pair<std::string, float>> ranked, int top_k,
                     bool truncated = false) {
        // stable_sort so equal scores keep the caller's label order, matching
        // the reference's stable sorted(..., reverse=True).
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](auto& a, auto& b) { return a.second > b.second; });
        if (top_k > 0 && static_cast<size_t>(top_k) < ranked.size()) ranked.resize(top_k);
        json labels = json::object();
        for (auto& [label, score] : ranked) labels[label] = score;
        json out{{"labels", labels}};
        if (truncated) out["truncated"] = true;
        return out;
    }

    json classify(const std::string& text, int top_k) {
        Encoded enc = encode(text);
        Bindings b;
        b.input_ids = std::move(enc.ids);

        // Drop the tokenizer's own padding so the model sees exactly what the
        // HuggingFace reference sees (see pad_id_ in the constructor).
        if (pad_id_ >= 0) {
            while (b.input_ids.size() > 1 && b.input_ids.back() == pad_id_) {
                b.input_ids.pop_back();
            }
        }
        if (b.input_ids.empty()) throw std::runtime_error("empty tokenization");

        // Truncate to the manifest's token budget, keeping the trailing token
        // (usually [SEP] / </s>) so the sequence stays well-formed.
        const size_t max_len = static_cast<size_t>(manifest_.max_length);
        bool truncated = false;
        if (b.input_ids.size() > max_len) {
            int64_t last = b.input_ids.back();
            b.input_ids.resize(max_len - 1);
            b.input_ids.push_back(last);
            truncated = true;
        }
        b.seq_len = static_cast<int64_t>(b.input_ids.size());

        // Standard encoder inputs: attention_mask all-ones, token_type_ids all-zeros.
        b.attention_mask.assign(b.input_ids.size(), 1);
        b.token_type_ids.assign(b.input_ids.size(), 0);

        auto outputs = run(b);
        std::vector<int64_t> out_shape;
        const float* logits = float_logits(outputs[0], out_shape);
        const size_t num_labels = manifest_.id2label.size();

        // Guard against a model/manifest mismatch before indexing the buffer.
        if (out_shape.empty() || out_shape.back() < 0 ||
            static_cast<size_t>(out_shape.back()) != num_labels) {
            throw std::runtime_error(
                "model output last dimension (" +
                std::to_string(out_shape.empty() ? -1 : out_shape.back()) +
                ") does not match manifest id2label size (" + std::to_string(num_labels) + ")");
        }

        std::map<std::string, float> scores;
        if (manifest_.task == "token-classification") {
            // out_shape = [1, tokens, labels]; aggregate per-label across
            // tokens per the manifest (a routing-friendly presence signal).
            if (out_shape.size() < 3) {
                throw std::runtime_error("token-classification model must output [batch, tokens, labels]");
            }
            const size_t tokens = static_cast<size_t>(out_shape[out_shape.size() - 2]);
            const bool mean = manifest_.token_aggregation == "mean";
            std::vector<double> agg(num_labels, 0.0);
            size_t counted = 0;
            for (size_t t = 0; t < tokens; ++t) {
                // Skip [CLS]/[SEP]/… : the HuggingFace pipeline filters those
                // positions out of its output, so scoring them would invent
                // entities the reference never reports.
                if (t < b.input_ids.size() && special_ids_.count(b.input_ids[t])) continue;
                ++counted;
                auto p = normalize(logits + t * num_labels, num_labels, manifest_.score_normalization);
                for (size_t l = 0; l < num_labels; ++l) {
                    if (mean) agg[l] += p[l];
                    else agg[l] = std::max(agg[l], static_cast<double>(p[l]));
                }
            }
            if (counted == 0) {
                throw InvalidInput("input has no content tokens to classify "
                                   "(it tokenizes to special tokens only)");
            }
            for (size_t l = 0; l < num_labels; ++l) {
                scores[manifest_.id2label[l]] =
                    static_cast<float>(mean ? agg[l] / counted : agg[l]);
            }
        } else {
            // sequence-classification: normalize the label logits.
            if (out_shape.size() > 2) {
                throw std::runtime_error(
                    "text-classification model must output [batch, labels]; got a rank-" +
                    std::to_string(out_shape.size()) + " tensor (token-classification model?)");
            }
            auto p = normalize(logits, num_labels, manifest_.score_normalization);
            for (size_t l = 0; l < num_labels; ++l) scores[manifest_.id2label[l]] = p[l];
        }
        return rank({scores.begin(), scores.end()}, top_k, truncated);
    }

    // --- zero-shot ----------------------------------------------------------
    // Template, token budget, truncation, pooling, and where this departs from
    // the reference route(): docs/ZERO-SHOT-ROUTER.md.
    json route(const std::string& text, const std::vector<std::string>& labels, int top_k) {
        if (labels.empty()) throw InvalidInput("\"labels\" must contain at least one label");
        if (labels.size() > static_cast<size_t>(manifest_.max_labels)) {
            // A ceiling on the impossible, not a recommended size: a label
            // costs at least kMinTokensPerLabel tokens, so no more than this
            // many can EVER fit.
            throw InvalidInput("too many labels: " + std::to_string(labels.size()) +
                               " (limit " + std::to_string(manifest_.max_labels) +
                               "). A label costs at least " +
                               std::to_string(kMinTokensPerLabel) + " tokens, so this "
                               "model's " + std::to_string(manifest_.max_length) +
                               "-token window cannot hold more, however short they are.");
        }
        std::set<std::string> seen;
        for (const auto& l : labels) {
            if (l.find_first_not_of(" \t\r\n") == std::string::npos) {
                throw InvalidInput("labels must not be empty or whitespace-only");
            }
            if (utf8_len(l) > static_cast<size_t>(manifest_.max_label_chars)) {
                throw InvalidInput("label is too long: " + std::to_string(utf8_len(l)) +
                                   " characters (limit " + std::to_string(manifest_.max_label_chars) +
                                   "; raise it with \"max_label_chars\" in manifest.json "
                                   "or --max-label-chars)");
            }
            for (unsigned char c : l) {
                if (c < 0x20 || c == 0x7F) {
                    throw InvalidInput(
                        "labels must not contain control characters such as newline "
                        "or tab: '" + l + "'");
                }
            }
            if (!seen.insert(l).second) {
                // Identical labels would collide as keys in the response object,
                // silently dropping one of the scores.
                throw InvalidInput("duplicate label: '" + l + "'");
            }
        }

        const size_t max_len = static_cast<size_t>(manifest_.max_length);

        // Build the LABEL BLOCK on its own and record where each label sits, IN
        // CHARACTERS. The text is appended only after the block has been
        // measured and the text has been given what the block left.
        //   Categories:\n- <l0>\n- <l1>\n\nText:\n
        std::string block = "Categories:\n";
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(labels.size());
        size_t pos = utf8_len("Categories:\n");
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i) block += "\n";
            block += "- " + labels[i];
            const size_t start = pos + 2;          // skip "- "
            const size_t end = start + utf8_len(labels[i]);
            ranges.emplace_back(start, end);
            pos = end + 1;                         // the "\n" that follows
        }
        block += "\n\nText:\n";
        const size_t text_start = utf8_len(block);

        // Guard, by characters: refuse a block the window provably cannot hold
        // WITHOUT tokenizing it, so an absurd label set costs one division.
        if (text_start / max_token_chars_ > max_len) {
            throw InvalidInput(
                "the label list is far larger than this model's " +
                std::to_string(max_len) + "-token window (" +
                std::to_string(text_start) + " characters of labels). "
                "Send fewer or shorter labels.");
        }

        // Labels are mandatory: every one must be scored, so
        // a block that does not fit is refused, never cut.
        const size_t label_tokens = encode(block, /*add_special_tokens=*/false).size();
        // kMinTextTokens is a floor on a window big enough to have one.
        const size_t min_text = std::max<size_t>(1, std::min(kMinTextTokens, max_len / 8));
        const size_t reserve = n_specials_ + min_text;
        if (label_tokens + reserve > max_len) {
            throw InvalidInput(
                "the label list needs " + std::to_string(label_tokens) +
                " tokens, and this model's window holds " + std::to_string(max_len) +
                " with " + std::to_string(reserve) +
                " reserved for markers and text. Send fewer or shorter labels.");
        }

        const size_t text_budget = max_len - n_specials_ - label_tokens;
        bool truncated = false;
        std::string body = text;
        size_t mark_from = 0;
        size_t mark_to = 0;

        const size_t char_guard = text_budget * max_token_chars_;
        if (utf8_len(body) > char_guard) {
            body = utf8_head_tail(body, char_guard);
            truncated = true;
        }

        Encoded t = encode(body, /*add_special_tokens=*/false);
        if (t.size() > text_budget) {
            // Pay for the marker out of the budget, but only if the budget can
            // afford it and still keep both ends.
            size_t keep = text_budget;
            std::string mid;
            if (text_budget > elision_tokens_ + 1) {
                keep = text_budget - elision_tokens_;
                mid = kElision;
            }
            const size_t head = std::max<size_t>(1, keep / kTextHeadDivisor);
            const size_t tail = keep - head;
            const size_t total = utf8_len(body);
            const size_t head_end = head_chars_for(t, head);
            size_t tail_start = tail_chars_for(t, tail, total);
            if (tail_start < head_end) tail_start = head_end;
            if (head_end == 0 && tail_start == total) {
                mid.clear();
                tail_start = tail_chars_for(t, text_budget, total);
            }
            body = utf8_range(body, 0, head_end) + mid +
                   utf8_range(body, tail_start, total);
            mark_from = head_end;
            mark_to = head_end + utf8_len(mid);
            truncated = true;
        }

        Encoded enc = encode(block + body);
        // tokens(A + B) can differ from tokens(A) + tokens(B) by a token or
        // two where the two strings meet. Shorten the TEXT and tokenize again
        // rather than trimming ids: a raw id cut can land inside a character,
        // which is exactly what the offsets above exist to prevent. The label
        // block is never touched -- the window check proved it fits, and it
        // sits ahead of the text.
        for (int guard = 0;
             enc.size() > max_len && guard < 8 && !body.empty(); ++guard) {
            const size_t chars = utf8_len(body);
            const size_t drop =
                std::min(chars, std::max<size_t>(1, enc.size() - max_len));
            body = utf8_range(body, 0, chars - drop);
            enc = encode(block + body);
            truncated = true;
        }
        if (enc.size() > max_len) {
            enc.ids.resize(max_len);
            enc.starts.resize(max_len);
            enc.ends.resize(max_len);
            truncated = true;
        }
        const size_t n = enc.size();
        if (n == 0) throw std::runtime_error("empty tokenization");

        if (truncated && verbose_) {
            std::cerr << "[ort-server] text truncated: " << labels.size()
                      << " labels cost " << label_tokens << " tokens of a "
                      << max_len << "-token window, leaving " << text_budget
                      << " for a text that needed " << t.size() << "\n";
        }

        const long long cells = static_cast<long long>(labels.size()) * static_cast<long long>(n);
        if (cells > manifest_.max_grid_cells) {
            throw InvalidInput(
                "label set too large for this model: " + std::to_string(labels.size()) +
                " labels x " + std::to_string(n) + " tokens = " + std::to_string(cells) +
                " pooling-grid cells (limit " + std::to_string(manifest_.max_grid_cells) +
                "; use fewer labels, or raise \"max_grid_cells\" in "
                "manifest.json or --max-grid-cells)");
        }

        Bindings b;
        b.input_ids = std::move(enc.ids);
        b.attention_mask.assign(n, 1);
        b.seq_len = static_cast<int64_t>(n);
        b.num_labels = static_cast<int64_t>(labels.size());

        const size_t mark0 = text_start + mark_from;
        const size_t mark1 = text_start + mark_to;

        // The label and text tests are independent, as in the reference: a token
        // straddling the "\n\nText:\n" header counts for both. The server's own
        // elision marker is kept out of text_pool by position.
        auto scan = [&](auto&& on_label, auto&& on_text) {
            size_t r0 = 0;
            uint32_t prev_start = 0;
            for (size_t i = 0; i < n; ++i) {
                if (enc.starts[i] == enc.ends[i]) continue;
                if (enc.starts[i] < prev_start) r0 = 0;
                prev_start = enc.starts[i];
                while (r0 < ranges.size() && ranges[r0].second <= enc.starts[i]) ++r0;
                for (size_t r = r0; r < ranges.size() && ranges[r].first < enc.ends[i]; ++r) {
                    if (enc.starts[i] < ranges[r].second) on_label(r, i);
                }
                const bool in_marker = enc.starts[i] >= mark0 && enc.ends[i] <= mark1;
                if (enc.ends[i] > text_start && !in_marker) on_text(i);
            }
        };

        std::vector<size_t> counts(labels.size(), 0);
        size_t text_count = 0;
        scan([&](size_t r, size_t) { ++counts[r]; }, [&](size_t) { ++text_count; });

        for (size_t r = 0; r < labels.size(); ++r) {
            if (counts[r] == 0) {
                throw InvalidInput(
                    "label '" + labels[r] + "' holds nothing this tokenizer keeps");
            }
        }
        if (text_count == 0) {
            throw InvalidInput(
                truncated
                    ? "the label list leaves " + std::to_string(text_budget) +
                          " tokens for the text, fewer than one of its characters "
                          "needs. Send fewer or shorter labels."
                    : std::string(
                          "no text tokens: \"text\" holds nothing this tokenizer keeps"));
        }

        b.category_pool.assign(labels.size() * n, 0.0f);
        b.text_pool.assign(n, 0.0f);
        const float text_w = 1.0f / static_cast<float>(text_count);
        scan([&](size_t r, size_t i) {
                 b.category_pool[r * n + i] = 1.0f / static_cast<float>(counts[r]);
             },
             [&](size_t i) { b.text_pool[i] = text_w; });

        auto outputs = run(b);
        std::vector<int64_t> out_shape;
        const float* logits = float_logits(outputs[0], out_shape);
        if (out_shape.empty() || out_shape.back() < 0 ||
            static_cast<size_t>(out_shape.back()) != labels.size()) {
            throw std::runtime_error(
                "model output last dimension (" +
                std::to_string(out_shape.empty() ? -1 : out_shape.back()) +
                ") does not match the " + std::to_string(labels.size()) +
                " labels supplied; the graph may have a fixed label count baked in");
        }
        auto p = softmax(logits, labels.size());
        std::vector<std::pair<std::string, float>> ranked;
        ranked.reserve(labels.size());
        for (size_t r = 0; r < labels.size(); ++r) ranked.emplace_back(labels[r], p[r]);
        return rank(std::move(ranked), top_k, truncated);
    }

    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions alloc_;
    std::unique_ptr<TokHandle, decltype(&tok_free)> tokenizer_{nullptr, tok_free};
    bool verbose_ = false;
    int64_t pad_id_ = -1;  // >= 0 when tokenizer.json enables padding
    std::set<int64_t> special_ids_;
    size_t max_token_chars_ = kMinTokenCharsBound;
    size_t n_specials_ = 0;
    size_t elision_tokens_ = 0;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const InputSpec*> bound_;  // parallel to input_names_
    Manifest manifest_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        Model model(args.model_path, args.verbose, args.limits);

        httplib::Server srv;
        srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status", "ok"},
                                 {"version", ORT_SERVER_VERSION},
                                 {"onnxruntime", ORT_SERVER_ONNXRUNTIME_VERSION}}.dump(),
                            "application/json");
        });
        srv.Post("/classify", [&](const httplib::Request& req, httplib::Response& res) {
            std::string text;
            int top_k = 0;
            std::vector<std::string> labels;
            bool has_labels = false;
            try {
                json body = json::parse(req.body);
                text = body.contains("text") ? body.at("text").get<std::string>()
                                             : body.at("input").get<std::string>();
                top_k = body.value("top_k", 0);
                if (body.contains("labels") && !body["labels"].is_null()) {
                    if (!body["labels"].is_array()) {
                        throw std::runtime_error("\"labels\" must be an array of strings");
                    }
                    for (const auto& l : body["labels"]) {
                        if (!l.is_string()) {
                            throw std::runtime_error("\"labels\" must be an array of strings");
                        }
                        labels.push_back(l.get<std::string>());
                    }
                    has_labels = true;
                }
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
                return;
            }
            try {
                json out = model.handle(text, has_labels ? &labels : nullptr, top_k);
                res.set_content(out.dump(), "application/json");
            } catch (const InvalidInput& e) {
                res.status = 400;  // the request is at fault, not the model
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });

        if (!srv.listen("127.0.0.1", args.port)) {
            fprintf(stderr, "ort-server: failed to bind 127.0.0.1:%d\n", args.port);
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ort-server: %s\n", e.what());
        return 1;
    }
}
