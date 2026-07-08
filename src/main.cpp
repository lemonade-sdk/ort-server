// ort-server — generic ONNX Runtime model server for Lemonade (see README).
//
// v1: CPU EP, text-classification. The model graph takes a raw string (tokenizer
// baked in via onnxruntime-extensions) and emits logits; this process only picks
// the EP, runs the session, and shapes the output per manifest.json.
//
// STATUS: initial scaffold — API shapes are correct but this has not been built
// yet. Items to verify at first build are marked TODO(build).

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

// onnxruntime-extensions: registers the in-graph tokenizer/custom ops.
// TODO(build): confirm header + symbol name for the pinned extensions tag.
#include <onnxruntime_extensions.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Manifest {
    std::string task;                       // "text-classification" | "token-classification"
    std::vector<std::string> id2label;      // index -> label
    std::string normalization = "softmax";
    std::string token_aggregation;          // empty for seq-cls
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
    m.normalization = j.value("score_normalization", "softmax");
    m.token_aggregation = j.value("token_aggregation", "");
    // id2label is a {"0": "LABEL", ...} object; flatten to an index-ordered vector.
    auto id2label = j.at("id2label");
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

class Model {
public:
    Model(const fs::path& dir, bool verbose)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ort-server"), manifest_(load_manifest(dir)) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(0);  // let ORT choose; matches default CPU EP
        // Register the in-graph tokenizer custom ops (string-in graphs).
        // TODO(build): exact registration call for the pinned extensions version.
        RegisterCustomOps(static_cast<OrtSessionOptions*>(opts), OrtGetApiBase());
        (void)verbose;
        session_ = Ort::Session(env_, (dir / "model.onnx").c_str(), opts);
        input_name_ = session_.GetInputNameAllocated(0, alloc_).get();
    }

    // Returns {label: score}. v1 handles text-classification; token-classification
    // aggregation is a TODO wired off manifest_.token_aggregation.
    json classify(const std::string& text, int top_k) {
        const char* in_names[] = {input_name_.c_str()};
        // Single string input tensor, shape [1].
        Ort::AllocatorWithDefaultOptions alloc;
        std::vector<int64_t> shape{1};
        Ort::Value input = Ort::Value::CreateTensor(
            alloc, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
        const char* strs[] = {text.c_str()};
        input.FillStringTensor(strs, 1);

        Ort::AllocatorWithDefaultOptions out_alloc;
        size_t out_count = session_.GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
        std::vector<const char*> on;
        for (size_t i = 0; i < out_count; ++i) {
            out_name_ptrs.push_back(session_.GetOutputNameAllocated(i, out_alloc));
            on.push_back(out_name_ptrs.back().get());
        }
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, on.data(), on.size());

        float* logits = outputs[0].GetTensorMutableData<float>();
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        size_t n = manifest_.id2label.size();
        (void)info;

        auto scores = softmax(logits, n);
        std::vector<std::pair<std::string, float>> ranked;
        for (size_t i = 0; i < n; ++i) ranked.push_back({manifest_.id2label[i], scores[i]});
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
    std::string input_name_;
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
