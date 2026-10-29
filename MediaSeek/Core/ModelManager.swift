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

    /// 图像模型切换(so400m ↔ 标准,第四十六批):维度变了旧视觉向量作废,
    /// 广播触发自动同步——同步开始处的维度比对会清空 clip 向量并全量重嵌,
    /// 不用等用户手动点同步或重启 App
    static let clipModelChanged = Notification.Name("clipModelChanged")
}

enum ModelSlot: String {
    case text = "TextEmbedder"
    case image = "ImageEmbedder"
    case imagePro = "ImageEmbedderPro"
    case face = "FaceEmbedder"

    var displayName: String {
        switch self {
        case .text: return "文本模型(EmbeddingGemma)"
        case .image: return "图文模型(SigLIP)"
        case .imagePro: return "图文模型·高精度(SigLIP so400m)"
        case .face: return "人脸模型(ArcFace r50)"
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
    /// 一轮 loadAll 是否进行中:排队未轮到的槽位状态仍是 .missing,设置页
    /// 据此显示「加载中(排队)」而非「未安装」——冷启动时用户看到 SigLIP
    /// 「未安装」以为模型被删了(2026-10-16 用户实测截图),实际在排队
    @Published private(set) var loadPassRunning = false

    // MARK: 图像模型切换(第四十六批:标准 ↔ so400m)
    /// so400m 增强包是否已导入(ImageEmbedderPro 槽位有 meta.json)
    @Published private(set) var proInstalled = false
    /// 切换进行中(设置页禁用交互,防双击并发双加载)
    @Published private(set) var imageSwitching = false

    private(set) var gemma: GemmaTextEmbedder?
    private(set) var siglip: SigLIPEmbedder?
    private(set) var face: FaceEmbedder?

    var bothReady: Bool { textState == .ready && imageState == .ready }
    var anyReady: Bool { textState == .ready || imageState == .ready }

    /// 当前是否用 so400m 高精度图像模型(默认标准)
    var useProImage: Bool {
        UserDefaults.standard.bool(forKey: "imageModelUsePro")
    }
    /// 实际参与加载的图像槽位:选了 pro 但没导入 → 自动退回标准,
    /// 保证 App 永远有可用的图像模型
    var activeImageSlot: ModelSlot {
        useProImage && proInstalled ? .imagePro : .image
    }

    /// SigLIP 分数门槛缩放系数(so400m 校准,2026-10-22 CI 双模型分布实测):
    /// so400m 命中带均值 = base × 0.87(0.0838 vs 0.0963),噪声带同向下移
    /// (0.0635 vs 0.0731)——绝对门槛不可跨模型迁移,按激活槽位整体缩放。
    /// 性别对比线 0.010 是边距制且 so400m 实测边距 女照+0.039/男照-0.047
    /// 余量充足,不缩。搜索侧 SigLIP 门槛全部乘此系数;Gemma 文件门槛
    /// 与图像模型无关,不缩
    var siglipGateScale: Float {
        activeImageSlot == .imagePro ? 0.87 : 1.0
    }

    /// 批95:视觉查询向量按模型分流。依据=2026-10-28 A/B 矩阵(build 248,
    /// so400m 上三族 3:0:海边日落召回+67%且增量恰为最相关橙桶/猫族多真猫
    /// 零误报/驾驶证唯一真证过尾线):so400m 的中文原生理解实测强劲。
    /// **矩阵只在 so400m 上跑过,标准版的原生中文是从未测量的空白格**——
    /// 批71 前科:so400m 结论不跨模型迁移(licence 翻案三批)。故 so400m
    /// 激活时裸搜/语义: 的中文词走模板壳+中文原文;标准版保持词典英译
    /// 原配行为(nativeZh=false 逐字不变)。范围=单词语径与 语义:多词交集径
    /// (runMainSearch;批106 曾扩到平均径,批107 平均退役改交集,原料沿用);
    /// AND 线逐词/英文桥维持模板——各待独立 A/B 后再动。
    /// 原生语义: 前缀不受此条件约束(显式实验意图,任何模型无条件直通)
    var preferNativeZhQuery: Bool {
        activeImageSlot == .imagePro
    }

    init() {
        refreshProInstalled()
    }

    func refreshProInstalled() {
        let meta = supportRoot.appendingPathComponent(ModelSlot.imagePro.rawValue)
            .appendingPathComponent("meta.json")
        proInstalled = FileManager.default.fileExists(atPath: meta.path)
    }

    /// 切换图像模型(标准 ↔ so400m)。切换 = 换视觉「眼睛」:
    /// 释放旧模型内存(两者不可共存,so400m 明显更大)→ 加载新槽位 →
    /// 维度变化时广播 clipModelChanged,AppModel 收到后自动开同步清旧向量全量重嵌
    func setImageModel(pro: Bool) async {
        guard !imageSwitching else { return }
        guard pro != useProImage else { return }
        guard !pro || proInstalled else {
            detailText = "so400m 增强包未导入(设置→模型在线更新)"
            return
        }
        imageSwitching = true
        defer { imageSwitching = false }
        let prevDim = UserDefaults.standard.object(forKey: "clipModelDim") as? Int
        UserDefaults.standard.set(pro, forKey: "imageModelUsePro")
        // 先释放旧 SigLIP 实例再加载新槽位,避免两个图像模型短时同驻内存
        siglip = nil
        setState(.loading, for: .image)
        await loadSlot(activeImageSlot)
        // so400m 加载失败(包损坏/内存不足等):退回标准,保证语义搜索可用
        if siglip == nil, pro {
            UserDefaults.standard.set(false, forKey: "imageModelUsePro")
            await loadSlot(.image)
        }
        if siglip != nil,
           let nowDim = UserDefaults.standard.object(forKey: "clipModelDim") as? Int,
           nowDim != prevDim {
            NotificationCenter.default.post(name: .clipModelChanged, object: nil)
        }
    }

    private var supportRoot: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MediaSeek/Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func state(for slot: ModelSlot) -> ModelState {
        switch slot {
        case .text: return textState
        case .image, .imagePro: return imageState
        case .face: return faceState
        }
    }

    private func setState(_ s: ModelState, for slot: ModelSlot) {
        switch slot {
        case .text: textState = s
        case .image, .imagePro: imageState = s
        case .face: faceState = s
        }
    }

    // MARK: - 加载

    func loadAll() async {
        // 并行加载(第二十四批):iOS 把后台进程杀掉后重开=冷启动,模型必须
        // 从盘重载(进程级无解);此前三槽串行,总等待=三者之和,且排队槽位
        // 显示「未安装」吓人。改并行:总等待≈最慢一个,排队槽位由设置页
        // 按 loadPassRunning 显示「加载中(排队)」
        loadPassRunning = true
        defer { loadPassRunning = false }
        await withTaskGroup(of: Void.self) { group in
            if textState != .ready { group.addTask { await self.loadSlot(.text) } }
            // 只加载「当前选中」的图像模型(标准/so400m 二选一,第四十六批):
            // so400m 体积数倍于标准,两者不可同时驻内存;选了 pro 没导入时
            // activeImageSlot 已自动退回 .image
            if imageState != .ready { group.addTask { await self.loadSlot(self.activeImageSlot) } }
            group.addTask { await self.loadSlot(.face) }   // 人脸模型可选,缺时静默降级
        }
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
                        throw MSError(slot == .imagePro
                            ? "so400m 增强包未导入(设置→模型在线更新 下载后可切换)"
                            : "未安装模型包 \(slot.rawValue)")
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

    // MARK: - 包内容指纹(批89)

    /// 包内容指纹:按包内各槽位 meta.json 的内容派生字段拼装(文本/图像=dim,
    /// 人脸=dim+ONNX 字节数)。这些值随模型内容变、不随 CI 重发变——服务端
    /// 重发同一批模型(版本字符串必变)指纹不变,「与服务端一致,无需重装」
    /// 才成立;真换模型至少 dim/model_size 会变(mbf→r50 字节变,标准→so400m
    /// 维度变,均被覆盖)。字段只用历代包必有的(不用 image_size 等后加字段,
    /// 旧包不错杀);任一槽位缺失(没装全)返回 nil = 需要安装。
    /// 与 export_models 生成 manifest.fingerprint 的拼法同规,改一处须同步另一处
    func installedFingerprint(forPackage key: String) -> String? {
        let slots: [ModelSlot] = key == "so400m" ? [.imagePro] : [.text, .image, .face]
        var parts: [String] = []
        for slot in slots {
            let metaURL = supportRoot.appendingPathComponent(slot.rawValue)
                .appendingPathComponent("meta.json")
            guard let data = try? Data(contentsOf: metaURL),
                  let meta = try? JSONDecoder().decode(ModelMeta.self, from: data) else { return nil }
            switch slot {
            case .text: parts.append("T:\(meta.dim)")
            case .image: parts.append("I:\(meta.dim)")
            case .imagePro: parts.append("P:\(meta.dim)")
            case .face: parts.append("F:\(meta.dim):\(meta.model_size ?? 0)")
            }
        }
        return parts.joined(separator: "|")
    }

    // MARK: - 从 zip 导入模型包

    /// zip 根目录应包含 TextEmbedder/ 与 ImageEmbedder/(各含 .mlpackage、tokenizer.json、meta.json)
    /// - Parameter imageTarget: 显式指定图像包落位槽位(so400m 在线更新传 .imagePro,
    ///   与随包标准模型并存实现切换);nil 时按 meta 维度启发式路由:
    ///   dim ≥ 1024(so400m=1152)视为增强包进 .imagePro,防止手动导入增强包
    ///   覆盖掉标准槽位
    func importModelZip(at url: URL, imageTarget: ModelSlot? = nil) async throws {
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
            guard var slot = ModelSlot(rawValue: name) else { continue }
            let src = extractDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: src.appendingPathComponent("meta.json").path) else { continue }

            // 图像包路由(第四十六批):显式指定优先;否则读 meta.json 的 dim,
            // 高维包(so400m=1152)自动落 .imagePro,不覆盖标准槽
            if slot == .image {
                if imageTarget == .imagePro {
                    slot = .imagePro
                } else if imageTarget == nil,
                          let data = try? Data(contentsOf: src.appendingPathComponent("meta.json")),
                          let meta = try? JSONDecoder().decode(ModelMeta.self, from: data),
                          meta.dim >= 1024 {
                    slot = .imagePro
                }
            }

            let dest = supportRoot.appendingPathComponent(slot.rawValue)
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

        refreshProInstalled()
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
