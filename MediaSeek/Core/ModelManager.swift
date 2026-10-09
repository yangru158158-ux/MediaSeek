import Foundation
import CoreML
import ZIPFoundation

enum ModelState: Equatable {
    case missing, loading, ready, failed(String)
}

enum ModelSlot: String {
    case text = "TextEmbedder"
    case image = "ImageEmbedder"

    var displayName: String {
        switch self {
        case .text: return "文本模型(EmbeddingGemma)"
        case .image: return "图文模型(SigLIP)"
        }
    }
}

struct ModelMeta: Codable {
    let type: String     // "gemma" | "clip"
    let space: String
    let dim: Int
    let image_size: Int?
    let image_mean: Float?
    let image_std: Float?
}

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var textState: ModelState = .missing
    @Published private(set) var imageState: ModelState = .missing
    @Published private(set) var detailText: String = ""

    private(set) var gemma: GemmaTextEmbedder?
    private(set) var siglip: SigLIPEmbedder?

    var bothReady: Bool { textState == .ready && imageState == .ready }
    var anyReady: Bool { textState == .ready || imageState == .ready }

    private var supportRoot: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MediaSeek/Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func state(for slot: ModelSlot) -> ModelState {
        slot == .text ? textState : imageState
    }

    private func setState(_ s: ModelState, for slot: ModelSlot) {
        if slot == .text { textState = s } else { imageState = s }
    }

    // MARK: - 加载

    func loadAll() async {
        guard textState != .ready, imageState != .ready else { return }
        await loadSlot(.text)
        await loadSlot(.image)
    }

    private func loadSlot(_ slot: ModelSlot) async {
        setState(.loading, for: slot)
        let root = supportRoot
        let bundleDir = Bundle.main.url(forResource: "Models", withExtension: nil)?
            .appendingPathComponent(slot.rawValue)

        let prepared = await Task.detached(priority: .userInitiated) { () -> Result<(URL, URL, ModelMeta), Error> in
            do {
                // 1) 已编译缓存
                var dir = root.appendingPathComponent(slot.rawValue)
                if !FileManager.default.fileExists(atPath: dir.appendingPathComponent("meta.json").path) {
                    // 2) 从随包模型目录编译一次
                    guard let bundled = bundleDir,
                          FileManager.default.fileExists(atPath: bundled.appendingPathComponent("meta.json").path) else {
                        throw MSError("未安装模型包 \(slot.rawValue)")
                    }
                    dir = root.appendingPathComponent(slot.rawValue)
                    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                    for file in try FileManager.default.contentsOfDirectory(atPath: bundled.path) {
                        let src = bundled.appendingPathComponent(file)
                        if src.pathExtension == "mlpackage" {
                            let compiled = try await MLModel.compileModel(at: src)
                            let name = (file as NSString).deletingPathExtension
                            let dest = dir.appendingPathComponent("\(name).mlmodelc")
                            try? FileManager.default.removeItem(at: dest)
                            try FileManager.default.moveItem(at: compiled, to: dest)
                        } else {
                            try FileManager.default.copyItem(at: src, to: dir.appendingPathComponent(file))
                        }
                    }
                }
                let metaURL = dir.appendingPathComponent("meta.json")
                let meta = try JSONDecoder().decode(ModelMeta.self, from: Data(contentsOf: metaURL))
                let tokURL = dir.appendingPathComponent("tokenizer.json")
                guard FileManager.default.fileExists(atPath: tokURL.path) else {
                    throw MSError("缺少 tokenizer.json")
                }
                // swift-transformers 需要 config.json(+tokenizer_config.json)选择分词器类。
                // 注意:SigLIP2 官方就是 GemmaTokenizer,总是覆盖写入以防历史错误值残留
                let cfgURL = dir.appendingPathComponent("config.json")
                let modelType = meta.type == "gemma" ? "gemma3_text" : "siglip"
                try? JSONSerialization.data(withJSONObject: ["model_type": modelType])
                    .write(to: cfgURL)
                let tokCfgURL = dir.appendingPathComponent("tokenizer_config.json")
                try? JSONSerialization.data(withJSONObject: ["tokenizer_class": "GemmaTokenizer"])
                    .write(to: tokCfgURL)
                return .success((dir, tokURL, meta))
            } catch {
                return .failure(error)
            }
        }.value

        switch prepared {
        case .failure(let error):
            setState(.missing, for: slot)
            detailText = "\(slot.displayName):\(error.localizedDescription)"
        case .success(let (dir, tokURL, meta)):
            do {
                switch meta.type {
                case "gemma":
                    let modelURL = dir.appendingPathComponent("GemmaText.mlmodelc")
                    gemma = try await GemmaTextEmbedder(modelURL: modelURL, tokenizerFolder: tokURL.deletingLastPathComponent())
                    setState(.ready, for: slot)
                case "clip":
                    let imgURL = dir.appendingPathComponent("SiglipImage.mlmodelc")
                    let txtURL = dir.appendingPathComponent("SiglipText.mlmodelc")
                    siglip = try await SigLIPEmbedder(
                        textModelURL: txtURL, imageModelURL: imgURL,
                        tokenizerFolder: tokURL.deletingLastPathComponent(),
                        imageSize: meta.image_size ?? 256,
                        mean: meta.image_mean ?? 0.5, std: meta.image_std ?? 0.5)
                    setState(.ready, for: slot)
                default:
                    setState(.failed("未知模型类型 \(meta.type)"), for: slot)
                }
            } catch {
                setState(.failed(error.localizedDescription), for: slot)
            }
        }
    }

    // MARK: - 从 zip 导入模型包

    /// zip 根目录应包含 TextEmbedder/ 与 ImageEmbedder/(各含 .mlpackage、tokenizer.json、meta.json)
    func importModelZip(at url: URL) async throws {
        // 文件选择器给的是安全域 URL,必须先申请访问权
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-import-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let extractDir = tmp.appendingPathComponent("unzipped")
        try FileManager.default.unzipItem(at: url, to: extractDir)

        var found = 0
        for name in try FileManager.default.contentsOfDirectory(atPath: extractDir.path) {
            guard name == ModelSlot.text.rawValue || name == ModelSlot.image.rawValue else { continue }
            let slot: ModelSlot = (name == ModelSlot.text.rawValue) ? .text : .image
            let src = extractDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: src.appendingPathComponent("meta.json").path) else { continue }

            let dest = supportRoot.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
            for file in try FileManager.default.contentsOfDirectory(atPath: src.path) {
                let f = src.appendingPathComponent(file)
                if f.pathExtension == "mlpackage" {
                    let compiled = try await MLModel.compileModel(at: f)
                    let destName = (file as NSString).deletingPathExtension
                    try FileManager.default.moveItem(at: compiled, to: dest.appendingPathComponent("\(destName).mlmodelc"))
                } else {
                    try FileManager.default.copyItem(at: f, to: dest.appendingPathComponent(file))
                }
            }
            found += 1
            setState(.loading, for: slot)
        }
        guard found > 0 else { throw MSError("压缩包里没有找到 TextEmbedder / ImageEmbedder 模型目录") }

        textState = .loading
        imageState = .loading
        await loadAll()
    }

    /// 等待两个模型就绪(用于自动同步前),超时返回是否有任一就绪
    func waitReady(timeout: TimeInterval = 120) async -> Bool {
        let start = Date()
        while !bothReady {
            if Task.isCancelled { return false }
            if case .failed = textState, case .failed = imageState { return false }
            if Date().timeIntervalSince(start) > timeout { return anyReady }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return true
    }
}
