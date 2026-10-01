// ort-server — generic ONNX Runtime model server for Lemonade (see README).
//
// CPU EP. Serves a plain exported ONNX graph plus the model's own HuggingFace
// tokenizer.json, for three tasks:
//
//   text-classification   input_ids/attention_mask[/token_type_ids] -> [1, L]
//   token-classification  same inputs                              -> [1, T, L]
//   zero-shot-classification
//                         input_ids/attention_mask/text_pool/category_pool
//                                                                  -> [1, R]
//
// The first two take their labels from manifest.json (or infer them from
// config.json); zero-shot takes them from each request
// (docs/ZERO-SHOT-ROUTER.md).

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

#include "tok_ffi.h"  // third_party/tok_ffi: C ABI over HF `tokenizers`

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
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

struct Manifest {
    // "text-classification" | "token-classification" | "zero-shot-classification"
    std::string task;
    std::vector<std::string> id2label;  // index -> label; EMPTY for zero-shot
    std::string score_normalization = "softmax";  // "softmax" | "sigmoid"
    std::string token_aggregation = "max";        // token-cls only; "max" | "mean"
    int max_length = 512;               // token budget; longer inputs are truncated

    bool zero_shot() const { return task == kZeroShot; }
};

struct Args {
    std::string model_path;
    int port = 0;
    bool verbose = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "--model-path" && i + 1 < argc) a.model_path = argv[++i];
        else if (f == "--port" && i + 1 < argc) a.port = std::stoi(argv[++i]);
        else if (f == "--verbose") a.verbose = true;
    }
    if (a.model_path.empty() || a.port == 0) {
        throw std::runtime_error("usage: ort-server --model-path <dir> --port <n> [--verbose]");
    }
    return a;
}

// Length in characters: counts every byte that is not a continuation (10xxxxxx).
size_t utf8_len(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++n;
    return n;
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

// TWO allowlists, selected by task
// The classifier tasks feed the model a single sequence with a fabricated
// all-ones attention mask and all-zero token_type_ids, and truncate by keeping
// the trailing token. That is exactly right for BERT-family single-sequence
// encoders and wrong for architectures with different segment/special-token
// conventions (XLNet puts its classifier token last with a distinct segment id).
//
// The zero-shot task shares none of those conventions: it has no segment ids,
// supplies explicit pooling grids, and truncates with a plain cut. Adding lfm2
// to the encoder list would silently promise it the encoder treatment, so the
// lists stay separate
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

// j[key] as a token budget, or 0 if absent or implausible. HF writes a huge
// sentinel (1e30) when the tokenizer has no real limit; that parses as a double
// and is skipped by the integer check.
int budget_field(const json& j, const char* key) {
    if (!j.is_object() || !j.contains(key) || !j[key].is_number_integer()) return 0;
    auto n = j[key].get<long long>();
    return (n >= 2 && n <= 1000000) ? static_cast<int>(n) : 0;
}

// max_length precedence mirrors the exporter: the tokenizer's declared budget,
// then the model's position table (less 2 — RoBERTa-family configs declare
// max_position_embeddings larger than the usable budget), then 512.
void apply_inferred_max_length(const json& tokenizer_config, const json& config, Manifest& m) {
    if (int n = budget_field(tokenizer_config, "model_max_length")) {
        m.max_length = n;
        return;
    }
    if (int n = budget_field(config, "max_position_embeddings")) {
        m.max_length = n > 4 ? n - 2 : n;
        return;
    }
    m.max_length = 512;
}

// Zero-shot takes its default budget from tokenizer.json's truncation rule;
// config.json's position table is far larger than the window the model uses.
void apply_zero_shot_max_length(const json& tokenizer_json, const fs::path& dir, Manifest& m) {
    if (tokenizer_json.contains("truncation")) {
        if (int n = budget_field(tokenizer_json["truncation"], "max_length")) {
            m.max_length = n;
            return;
        }
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
    // A manifest does not excuse the architecture check.
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
        m.max_length = j["max_length"].get<int>();
        if (m.max_length < 2) throw std::runtime_error("max_length must be >= 2");
    } else if (m.zero_shot()) {
        apply_zero_shot_max_length(tokenizer_json, dir, m);
    } else {
        // Manifest omits the budget: fall back to the model's own metadata
        // rather than a blanket 512, which can overflow a smaller position table.
        apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"),
                                  read_json_if_present(dir / "config.json"), m);
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
Manifest load_manifest(const fs::path& dir, const json& tokenizer_json) {
    if (fs::exists(dir / "manifest.json")) return manifest_from_json(dir, tokenizer_json);
    return manifest_from_hf_config(dir, tokenizer_json);
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

class Model {
public:
    Model(const fs::path& dir, bool verbose)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ort-server") {
        (void)verbose;
        std::string blob = load_bytes(dir / "tokenizer.json");

        // Parse first, so a corrupt tokenizer.json (a partial download, say) gets
        // a clearer error than tok_new's bare null.
        json tj;
        try {
            tj = json::parse(blob);
        } catch (const std::exception& e) {
            throw std::runtime_error("tokenizer.json in " + dir.string() +
                                     " is not valid JSON (truncated or corrupt "
                                     "download?): " + e.what());
        }

        manifest_ = load_manifest(dir, tj);

        tokenizer_ = tok_new(reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
        if (!tokenizer_) throw std::runtime_error("failed to load tokenizer.json from " + dir.string());

        // Ids the post-processor INSERTS ([CLS]/[SEP] and the like). The model
        // sees them, but HF's token-classification pipeline drops them from its
        // output, so aggregation must skip them. Deliberately not every special
        // added_token: [UNK] and [MASK] are ordinary content positions to HF.
        // (Zero-shot needs none of this: inserted markers have zero-width spans,
        // which its pooling grids already exclude.)
        if (tj.contains("post_processor")) {
            collect_inserted_special_ids(tj["post_processor"], special_ids_);
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

    ~Model() {
        if (tokenizer_) tok_free(tokenizer_);
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool zero_shot() const { return manifest_.zero_shot(); }

    // /classify entry point: a request must carry `labels` iff the model is zero-shot.
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

    // Special tokens ON, as in every HF reference the scores are validated
    // against. tok_encode is thread-safe, so concurrent requests need no lock.
    Encoded encode(const std::string& text) const {
        TokEncoding enc{};
        if (!tok_encode(tokenizer_, reinterpret_cast<const uint8_t*>(text.data()),
                        text.size(), /*add_special_tokens=*/true, &enc)) {
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

    static json rank(std::vector<std::pair<std::string, float>> ranked, int top_k) {
        // Stable, so ties keep input order as Python's sorted() does.
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](auto& a, auto& b) { return a.second > b.second; });
        if (top_k > 0 && static_cast<size_t>(top_k) < ranked.size()) ranked.resize(top_k);
        json labels = json::object();
        for (auto& [label, score] : ranked) labels[label] = score;
        return json{{"labels", labels}};
    }

    json classify(const std::string& text, int top_k) {
        Encoded enc = encode(text);
        Bindings b;
        b.input_ids = std::move(enc.ids);
        if (b.input_ids.empty()) throw std::runtime_error("empty tokenization");

        // Truncate to the manifest's token budget, keeping the trailing token
        // (usually [SEP] / </s>) so the sequence stays well-formed.
        const size_t max_len = static_cast<size_t>(manifest_.max_length);
        if (b.input_ids.size() > max_len) {
            int64_t last = b.input_ids.back();
            b.input_ids.resize(max_len - 1);
            b.input_ids.push_back(last);
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
        return rank({scores.begin(), scores.end()}, top_k);
    }

    // --- zero-shot ----------------------------------------------------------
    // The model reads the labels and the text as one prompt; two pooling grids
    // tell it which tokens are which. Mirrors the reference route(): same
    // template, character ranges and overlap test.
    json route(const std::string& text, const std::vector<std::string>& labels, int top_k) {
        if (labels.empty()) throw InvalidInput("\"labels\" must contain at least one label");
        std::set<std::string> seen;
        for (const auto& l : labels) {
            if (l.find_first_not_of(" \t\r\n") == std::string::npos) {
                throw InvalidInput("labels must not be empty or whitespace-only");
            }
            if (!seen.insert(l).second) {
                // Would collide as response keys, silently dropping a score.
                throw InvalidInput("duplicate label: '" + l + "'");
            }
        }

        // Build the template and record where each label sits, IN CHARACTERS.
        //   Categories:\n- <l0>\n- <l1>\n\nText:\n<text>
        std::string prompt = "Categories:\n";
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(labels.size());
        size_t pos = utf8_len("Categories:\n");
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i) prompt += "\n";
            prompt += "- " + labels[i];
            const size_t start = pos + 2;          // skip "- "
            const size_t end = start + utf8_len(labels[i]);
            ranges.emplace_back(start, end);
            pos = end + 1;                         // the "\n" that follows
        }
        prompt += "\n\nText:\n";
        const size_t text_start = utf8_len(prompt);
        prompt += text;

        Encoded enc = encode(prompt);
        // Plain cut: the allowlisted template (lfm2) only prepends a marker, so
        // there is no trailing token to keep, unlike the encoder path.
        const size_t max_len = static_cast<size_t>(manifest_.max_length);
        if (enc.size() > max_len) {
            enc.ids.resize(max_len);
            enc.starts.resize(max_len);
            enc.ends.resize(max_len);
        }
        const size_t n = enc.size();
        if (n == 0) throw std::runtime_error("empty tokenization");

        Bindings b;
        b.input_ids = std::move(enc.ids);
        b.attention_mask.assign(n, 1);
        b.seq_len = static_cast<int64_t>(n);
        b.num_labels = static_cast<int64_t>(labels.size());

        // A token belongs to label r if its span overlaps r's range, and to the
        // text if it ends past text_start. Zero-width tokens, like the inserted
        // (0, 0) start marker, belong to neither.
        b.category_pool.assign(labels.size() * n, 0.0f);
        for (size_t r = 0; r < labels.size(); ++r) {
            const auto [start, end] = ranges[r];
            std::vector<size_t> idxs;
            for (size_t i = 0; i < n; ++i) {
                if (enc.starts[i] < end && enc.ends[i] > start &&
                    enc.starts[i] != enc.ends[i]) {
                    idxs.push_back(i);
                }
            }
            if (idxs.empty()) {
                throw InvalidInput(
                    "label '" + labels[r] + "' got no tokens: the label list overran "
                    "the model's " + std::to_string(max_len) +
                    "-token window (try fewer or shorter labels)");
            }
            const float w = 1.0f / static_cast<float>(idxs.size());
            for (size_t i : idxs) b.category_pool[r * n + i] = w;
        }

        b.text_pool.assign(n, 0.0f);
        std::vector<size_t> text_idxs;
        for (size_t i = 0; i < n; ++i) {
            if (enc.ends[i] > text_start && enc.starts[i] != enc.ends[i]) {
                text_idxs.push_back(i);
            }
        }
        if (text_idxs.empty()) {
            throw InvalidInput(
                "no text tokens: \"text\" is empty, or the label list left no room "
                "for it in the model's " + std::to_string(max_len) + "-token window");
        }
        for (size_t i : text_idxs) b.text_pool[i] = 1.0f / static_cast<float>(text_idxs.size());

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
        return rank(std::move(ranked), top_k);
    }

    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions alloc_;
    TokHandle* tokenizer_ = nullptr;
    std::set<int64_t> special_ids_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const InputSpec*> bound_;  // parallel to input_names_
    Manifest manifest_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        Model model(args.model_path, args.verbose);

        httplib::Server srv;
        srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status", "ok"},
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
