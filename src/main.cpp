// ort-server — generic ONNX Runtime model server for Lemonade (see README).
//
// v1: CPU EP, text classification. The model is a plain exported ONNX graph
// (input_ids / attention_mask / token_type_ids -> logits). This process loads
// the model + its HF tokenizer (via onnxruntime-extensions) + manifest.json,
// tokenizes the request at /classify, runs the session, and shapes the output
// per the manifest (softmax for sequence-classification, per-token aggregation
// for token-classification).

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

// onnxruntime-extensions tokenizer C API (used purely as a tokenizer library;
// the model graph itself uses no custom ops).
#include "ortx_tokenizer.h"
#include "ortx_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Manifest {
    std::string task;                   // "text-classification" | "token-classification"
    std::vector<std::string> id2label;  // index -> label
    std::string token_aggregation;      // token-cls only; "" for seq-cls
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

Manifest load_manifest(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json");
    if (!f) throw std::runtime_error("manifest.json not found in " + dir.string());
    json j; f >> j;
    Manifest m;
    m.task = j.at("task").get<std::string>();
    m.token_aggregation = j.value("token_aggregation", "");
    const auto& id2label = j.at("id2label");
    m.id2label.resize(id2label.size());
    for (auto it = id2label.begin(); it != id2label.end(); ++it) {
        m.id2label[std::stoul(it.key())] = it.value().get<std::string>();
    }
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

// Tokenize one string via onnxruntime-extensions. Returns the token ids.
std::vector<int64_t> tokenize(OrtxTokenizer* tokenizer, const std::string& text) {
    const char* inputs[] = {text.c_str()};
    OrtxTokenId2DArray* ids_2d = nullptr;
    if (OrtxTokenize(tokenizer, inputs, 1, &ids_2d) != kOrtxOK) {
        throw std::runtime_error("tokenization failed");
    }
    const extTokenId_t* ids = nullptr;
    size_t len = 0;
    if (OrtxTokenId2DArrayGetItem(ids_2d, 0, &ids, &len) != kOrtxOK) {
        ORTX_DISPOSE(ids_2d);
        throw std::runtime_error("failed to read token ids");
    }
    std::vector<int64_t> out(ids, ids + len);
    ORTX_DISPOSE(ids_2d);
    return out;
}

class Model {
public:
    Model(const fs::path& dir, bool verbose)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ort-server"), manifest_(load_manifest(dir)) {
        (void)verbose;
        if (OrtxCreateTokenizer(&tokenizer_, dir.string().c_str()) != kOrtxOK || !tokenizer_) {
            throw std::runtime_error("failed to load tokenizer from " + dir.string());
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
    }

    ~Model() {
        if (tokenizer_) ORTX_DISPOSE(tokenizer_);
    }

    json classify(const std::string& text, int top_k) {
        std::vector<int64_t> input_ids = tokenize(tokenizer_, text);
        const int64_t seq_len = static_cast<int64_t>(input_ids.size());
        if (seq_len == 0) throw std::runtime_error("empty tokenization");

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

        const float* logits = outputs[0].GetTensorData<float>();
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const size_t num_labels = manifest_.id2label.size();

        std::map<std::string, float> scores;
        if (manifest_.task == "token-classification") {
            // out_shape = [1, tokens, labels]; report the max probability per
            // label across tokens (a routing-friendly presence signal).
            const size_t tokens = out_shape.size() >= 2 ? static_cast<size_t>(out_shape[out_shape.size() - 2]) : 0;
            for (size_t t = 0; t < tokens; ++t) {
                auto p = softmax(logits + t * num_labels, num_labels);
                for (size_t l = 0; l < num_labels; ++l) {
                    scores[manifest_.id2label[l]] = std::max(scores[manifest_.id2label[l]], p[l]);
                }
            }
        } else {
            // sequence-classification: softmax over the label logits.
            auto p = softmax(logits, num_labels);
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
    OrtxTokenizer* tokenizer_ = nullptr;
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
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        srv.Post("/classify", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                json body = json::parse(req.body);
                std::string text = body.contains("text") ? body.at("text").get<std::string>()
                                                          : body.at("input").get<std::string>();
                int top_k = body.value("top_k", 0);
                res.set_content(model.classify(text, top_k).dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });

        srv.listen("127.0.0.1", args.port);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ort-server: %s\n", e.what());
        return 1;
    }
}
