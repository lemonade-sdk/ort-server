// ort-server — generic ONNX Runtime model server for Lemonade (see README).
//
// v1: CPU EP, text classification. The model is a plain exported ONNX graph
// (input_ids / attention_mask / token_type_ids -> logits). This process loads
// the model + its HF tokenizer (via tokenizers-cpp), derives the output
// contract from an optional manifest.json (or infers it from the export's own
// config.json), tokenizes the request at /classify, runs the session, and
// shapes the output (normalize for sequence-classification, per-token
// aggregation for token-classification).

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

// tokenizers-cpp loads the model's own HF tokenizer.json and tokenizes in
// process. The C API is used directly: the C++ wrapper's base interface
// hardcodes add_special_tokens=false, but encoder classifiers need [CLS]/[SEP]
// to match the HuggingFace reference they were validated against.
#include "tokenizers_c.h"

#include <mutex>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Manifest {
    std::string task;                   // "text-classification" | "token-classification"
    std::vector<std::string> id2label;  // index -> label
    std::string score_normalization = "softmax";  // "softmax" | "sigmoid"
    std::string token_aggregation = "max";        // token-cls only; "max" | "mean"
    int max_length = 512;               // token budget; longer inputs are truncated
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

// Best-effort: tokenizer_config.json is an optional refinement, so a missing
// or malformed file never aborts startup.
void apply_tokenizer_max_length(const fs::path& dir, Manifest& m) {
    std::ifstream tf(dir / "tokenizer_config.json");
    if (!tf) return;
    try {
        json tj; tf >> tj;
        // HF writes a huge sentinel (1e30) when the tokenizer has no real limit;
        // that parses as a double and is skipped by the integer check.
        if (tj.contains("model_max_length") && tj["model_max_length"].is_number_integer()) {
            auto n = tj["model_max_length"].get<long long>();
            if (n >= 2 && n <= 1000000) m.max_length = static_cast<int>(n);
        }
    } catch (const std::exception&) {
    }
}

Manifest manifest_from_json(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json");
    if (!f) throw std::runtime_error("cannot open manifest.json in " + dir.string());
    json j; f >> j;
    Manifest m;
    m.task = j.at("task").get<std::string>();
    if (m.task != "text-classification" && m.task != "token-classification") {
        throw std::runtime_error("unsupported task in manifest.json: '" + m.task +
                                 "' (expected text-classification or token-classification)");
    }
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
    }
    parse_id2label(j.at("id2label"), m, "manifest.json");
    return m;
}

// Fallback for a stock HF/Optimum export (no manifest.json): infer the contract
// from config.json (+ tokenizer_config.json), applying HF problem_type semantics.
Manifest manifest_from_hf_config(const fs::path& dir) {
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
    if (ends_with(arch, "ForTokenClassification")) {
        m.task = "token-classification";
    } else if (ends_with(arch, "ForSequenceClassification")) {
        m.task = "text-classification";
    } else {
        throw std::runtime_error("cannot infer task from config.json architecture '" +
                                 arch + "'; provide a manifest.json");
    }

    std::string problem_type;
    if (j.contains("problem_type") && j["problem_type"].is_string()) {
        problem_type = j["problem_type"].get<std::string>();
    }
    if (problem_type == "regression") {
        throw std::runtime_error("regression heads have no label scores in [0,1]");
    }
    m.score_normalization = (m.task == "text-classification" &&
                             problem_type == "multi_label_classification")
                                ? "sigmoid" : "softmax";

    parse_id2label(j.at("id2label"), m, "config.json");
    if (m.id2label.size() < 2) {
        throw std::runtime_error("single-output heads have no label scores in [0,1]");
    }
    apply_tokenizer_max_length(dir, m);
    return m;
}

// manifest.json (explicit contract, validated strictly) wins; a bare Optimum
// export runs via config.json inference so users need no lemonade tooling.
Manifest load_manifest(const fs::path& dir) {
    if (fs::exists(dir / "manifest.json")) return manifest_from_json(dir);
    return manifest_from_hf_config(dir);
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

class Model {
public:
    Model(const fs::path& dir, bool verbose)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ort-server"), manifest_(load_manifest(dir)) {
        (void)verbose;
        std::string blob = load_bytes(dir / "tokenizer.json");
        tokenizer_ = tokenizers_new_from_str(blob.data(), blob.size());
        if (!tokenizer_) throw std::runtime_error("failed to load tokenizer.json from " + dir.string());

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
    }

    ~Model() {
        if (tokenizer_) tokenizers_free(tokenizer_);
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    json classify(const std::string& text, int top_k) {
        // Special tokens ON: encoder classifiers pool [CLS]/use [SEP], and the
        // catalog's parity validation runs the HF tokenizer with them enabled —
        // serving without them would silently diverge from the validated scores.
        // The mutex covers the Rust FFI's &mut self contract; tokenization is
        // microseconds next to the session run, which stays concurrent.
        std::vector<int64_t> input_ids;
        {
            std::lock_guard<std::mutex> lock(tokenizer_mutex_);
            TokenizerEncodeResult result;
            tokenizers_encode(tokenizer_, text.data(), text.size(),
                              /*add_special_token=*/1, &result);
            input_ids.assign(result.token_ids, result.token_ids + result.len);
            tokenizers_free_encode_results(&result, 1);
        }
        if (input_ids.empty()) throw std::runtime_error("empty tokenization");

        // Truncate to the manifest's token budget, keeping the trailing token
        // (usually [SEP] / </s>) so the sequence stays well-formed.
        const size_t max_len = static_cast<size_t>(manifest_.max_length);
        if (input_ids.size() > max_len) {
            int64_t last = input_ids.back();
            input_ids.resize(max_len - 1);
            input_ids.push_back(last);
        }
        const int64_t seq_len = static_cast<int64_t>(input_ids.size());

        // Standard encoder inputs: attention_mask all-ones, token_type_ids all-zeros.
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> token_type_ids(input_ids.size(), 0);
        const std::array<int64_t, 2> shape{1, seq_len};

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto make = [&](std::vector<int64_t>& data) {
            return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(),
                                                     shape.data(), shape.size());
        };

        // Feed each declared input by name (models differ: DistilBERT/RoBERTa
        // have no token_type_ids; BERT/DeBERTa do).
        std::vector<Ort::Value> inputs;
        std::vector<const char*> in_names;
        for (const auto& name : input_names_) {
            if (name == "input_ids") inputs.push_back(make(input_ids));
            else if (name == "attention_mask") inputs.push_back(make(attention_mask));
            else if (name == "token_type_ids") inputs.push_back(make(token_type_ids));
            else throw std::runtime_error("unexpected model input: " + name);
            in_names.push_back(name.c_str());
        }

        std::vector<const char*> out_names;
        for (const auto& n : output_names_) out_names.push_back(n.c_str());

        auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                    inputs.size(), out_names.data(), out_names.size());

        auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
        if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "model output must be float32 logits (fp16/quantized-output exports are not supported)");
        }
        const float* logits = outputs[0].GetTensorData<float>();
        auto out_shape = type_info.GetShape();
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
            for (size_t t = 0; t < tokens; ++t) {
                auto p = normalize(logits + t * num_labels, num_labels, manifest_.score_normalization);
                for (size_t l = 0; l < num_labels; ++l) {
                    if (mean) agg[l] += p[l];
                    else agg[l] = std::max(agg[l], static_cast<double>(p[l]));
                }
            }
            for (size_t l = 0; l < num_labels; ++l) {
                scores[manifest_.id2label[l]] =
                    static_cast<float>(mean && tokens > 0 ? agg[l] / tokens : agg[l]);
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

        std::vector<std::pair<std::string, float>> ranked(scores.begin(), scores.end());
        std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if (top_k > 0 && static_cast<size_t>(top_k) < ranked.size()) ranked.resize(top_k);

        json labels = json::object();
        for (auto& [label, score] : ranked) labels[label] = score;
        return json{{"labels", labels}};
    }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions alloc_;
    TokenizerHandle tokenizer_ = nullptr;
    std::mutex tokenizer_mutex_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
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
            try {
                json body = json::parse(req.body);
                text = body.contains("text") ? body.at("text").get<std::string>()
                                             : body.at("input").get<std::string>();
                top_k = body.value("top_k", 0);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
                return;
            }
            try {
                res.set_content(model.classify(text, top_k).dump(), "application/json");
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
