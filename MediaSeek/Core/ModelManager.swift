import Foundation
import CoreML
import ZIPFoundation

enum ModelState: Equatable {
    case missing, loading, ready, failed(String)
}

extension Notification.Name {
    /// 人脸模型指纹变化(导入/升级后):旧脸向量不可用,应触发增量同步重建。
    /// 导入不走 App 重启,自动同步只在启动时跑——不广播的话用户导入 r50 后
    /// 直接去人物归组,聚类的还是旧模型向量(2026-10-13 用户实测「装了 r50
    /// 还是混人」:指纹链路本身是通的,缺的就是导入后的这次触发)
    static let faceModelChanged = Notification.Name("faceModelChanged")
}

enum ModelSlot: String {
    case text = "TextEmbedder"
    case image = "ImageEmbedder"
    case face = "FaceEmbedder"

    var displayName: String {
        switch self {
        case .text: return "文本模型(EmbeddingGemma)"
        case .image: return "图文模型(SigLIP)"
        case .face: return "人脸模型(MobileFaceNet)"
        }
    }
}

struct ModelMeta: Codable {
    let type: String     // "gemma" | "clip" | "face"
    let space: String
    let dim: Int
    let image_size: Int?
    let image_mean: Float?
    let image_std: Float?
    // 人脸模型:ONNX 原始输入/输出名(如 input.1 / 516)
    let input: String?
    let output: String?
    // 人脸模型指纹(ONNX 字节数):换模型时同步前自动清空旧脸向量重嵌
    let model_size: Int?
}

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var textState: ModelState = .missing
    @Published private(set) var imageState: ModelState = .missing
    @Published private(set) var faceState: ModelState = .missing
    @Published private(set) var detailText: String = ""

    private(set) var gemma: GemmaTextEmbedder?
    private(set) var siglip: SigLIPEmbedder?
    private(set) var face: FaceEmbedder?

    var bothReady: Bool { textState == .ready && imageState == .ready }
    var anyReady: Bool { textState == .ready || imageState == .ready }

    private var supportRoot: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MediaSeek/Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func state(for slot: ModelSlot) -> ModelState {
        switch slot {
        case .text: return textState
        case .image: return imageState
        case .face: return faceState
        }
    }

    private func setState(_ s: ModelState, for slot: ModelSlot) {
        switch slot {
        case .text: textState = s
        case .image: imageState = s
        case .face: faceState = s
        }
    }

    // MARK: - 加载

    func loadAll() async {
        if textState != .ready { await loadSlot(.text) }
        if imageState != .ready { await loadSlot(.image) }
        await loadSlot(.face)   // 人脸模型可选:未装时静默降级(人物归组功能不可用)
    }

    private func loadSlot(_ slot: ModelSlot) async {
        setState(.loading, for: slot)
        let root = supportRoot
        let bundleDir = Bundle.main.url(forResource: "Models", withExtension: nil)?
            .appendingPathComponent(slot.rawValue)

        let prepared = await Task.detached(priority: .userInitiated) { () -> Result<(URL, URL?, ModelMeta), Error> in
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
                var tokURL: URL? = nil
                if slot != .face {
                    // 人脸模型无分词器;文本/图文模型必须有
                    let t = dir.appendingPathComponent("tokenizer.json")
                    guard FileManager.default.fileExists(atPath: t.path) else {
                        throw MSError("缺少 tokenizer.json")
                    }
                    tokURL = t
                    // swift-transformers 需要 config.json(+tokenizer_config.json)选择分词器类。
                    // 注意:SigLIP2 官方就是 GemmaTokenizer,总是覆盖写入以防历史错误值残留
                    let cfgURL = dir.appendingPathComponent("config.json")
                    let modelType = meta.type == "gemma" ? "gemma3_text" : "siglip"
                    try? JSONSerialization.data(withJSONObject: ["model_type": modelType])
                        .write(to: cfgURL)
                    let tokCfgURL = dir.appendingPathComponent("tokenizer_config.json")
                    try? JSONSerialization.data(withJSONObject: ["tokenizer_class": "GemmaTokenizer"])
                        .write(to: tokCfgURL)
                }
                return .success((dir, tokURL, meta))
            } catch {
                return .failure(error)
            }
        }.value

        switch prepared {
        case .failure(let error):
            setState(.missing, for: slot)
            // 人脸模型是可选件,缺它不写全局提示(只在设置页自己的行上显示「未安装」)
            if slot != .face {
                detailText = "\(slot.displayName):\(error.localizedDescription)"
            }
        case .success(let (dir, tokURL, meta)):
            do {
                switch meta.type {
                case "gemma":
                    let modelURL = dir.appendingPathComponent("GemmaText.mlmodelc")
                    guard let tokURL else { throw MSError("缺少 tokenizer.json") }
                    gemma = try await GemmaTextEmbedder(modelURL: modelURL, tokenizerFolder: tokURL.deletingLastPathComponent())
                    setState(.ready, for: slot)
                    if textState == .ready && imageState == .ready { detailText = "" }
                case "clip":
                    let imgURL = dir.appendingPathComponent("SiglipImage.mlmodelc")
                    let txtURL = dir.appendingPathComponent("SiglipText.mlmodelc")
                    guard let tokURL else { throw MSError("缺少 tokenizer.json") }
                    siglip = try await SigLIPEmbedder(
                        textModelURL: txtURL, imageModelURL: imgURL,
                        tokenizerFolder: tokURL.deletingLastPathComponent(),
                        imageSize: meta.image_size ?? 256,
                        mean: meta.image_mean ?? 0.5, std: meta.image_std ?? 0.5)
                    setState(.ready, for: slot)
                    if textState == .ready && imageState == .ready { detailText = "" }
                    // 记录当前图像模型维度;IndexingCoordinator 同步前与库内向量维度
                    // 比对,不一致即清空旧 clip 向量全量重建(so400m 等在线升级换模型)
                    UserDefaults.standard.set(meta.dim, forKey: "clipModelDim")
                case "face":
                    let modelURL = dir.appendingPathComponent("Face.mlmodelc")
                    face = try FaceEmbedder(modelURL: modelURL,
                                            inputName: meta.input ?? "input.1",
                                            outputName: meta.output ?? "embedding")
                    setState(.ready, for: slot)
                    // 记录人脸模型指纹;IndexingCoordinator 同步前与库内已嵌指纹比对,
                    // 不一致即清空旧向量重嵌(如 mbf→r50 在线升级,旧嵌入空间不通用)。
                    // 优先 meta.model_size(ONNX 字节数,导出脚本写入);
                    // 旧包缺该字段时退回用已编译模型目录总字节数当指纹(权重占绝对
                    // 大头、同设备上稳定)——已导入旧包的设备升级 App 后,
                    // 无需重新导入也能识别「模型已换」并触发人脸数据重建。
                    // 指纹有变化才广播:同步侧清空重嵌,导入完即刻生效不用重启
                    let prevFingerprint = UserDefaults.standard.integer(forKey: "faceModelSize")
                    let fingerprint = meta.model_size ?? Self.directoryBytes(modelURL)
                    if fingerprint > 0, fingerprint != prevFingerprint {
                        UserDefaults.standard.set(fingerprint, forKey: "faceModelSize")
                        NotificationCenter.default.post(name: .faceModelChanged, object: nil)
                    }
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
            guard let slot = ModelSlot(rawValue: name) else { continue }
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

    /// 目录递归总字节数(旧模型包无 model_size 时的指纹兜底)
    nonisolated private static func directoryBytes(_ url: URL) -> Int {
        guard let en = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]) else { return 0 }
        var total = 0
        for case let f as URL in en {
            total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }
}
