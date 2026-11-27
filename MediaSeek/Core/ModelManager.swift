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

    /// 图像模型变更:维度变了旧视觉向量作废,广播触发自动同步——同步开始处的
    /// 维度比对会清空 clip 向量并全量重嵌,不用等用户手动点同步或重启 App。
    /// 第四十六批由 setImageModel(标准 ↔ so400m 切换)发出;2.0 起切换逻辑随
    /// Base 停用删除,改由 loadSlot 在维度变化时发出(导入新图像模型包同样触发)
    static let clipModelChanged = Notification.Name("clipModelChanged")
}

enum ModelSlot: String {
    case text = "TextEmbedder"
    case image = "ImageEmbedder"
    case face = "FaceEmbedder"

    var displayName: String {
        switch self {
        case .text: return "文本模型(EmbeddingGemma)"
        case .image: return "图文模型(SigLIP2 so400m)"
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

    // MARK: 图像模型(2.0:唯一模型 so400m)
    /// 2.0 起 Base 标准模型停用,图像模型只有 so400m(用户拍板:so400m 效果更好,
    /// 且删掉 Base 数据源可让新用户少下 0.707GB)。1.0 的双槽位切换(第四十六批)
    /// 与「so400m 加载失败退回 Base」的降级一并删除——门槛系数与中文原生查询都
    /// 按 so400m 标定,拿 Base 顶上是未标定行为。图像模型加载失败时语义搜索不可用,
    /// 文字/标签/人脸不受影响。
    /// 停用判据:图像模型 meta.dim 低于此值即视为 Base(so400m=1152,Base=768;
    /// 沿用 1.0 导入器「dim ≥ 1024 视为增强包」的同一条线,将来换更大的模型仍满足)
    static let minImageModelDim = 1024
    /// 2.0 槽位迁移结果(设置页模型区显示;nil=无迁移发生)
    @Published private(set) var migrationNote: String?

    private(set) var gemma: GemmaTextEmbedder?
    private(set) var siglip: SigLIPEmbedder?
    private(set) var face: FaceEmbedder?
    /// 批157:每个槽位这一轮加载花了多少秒(设置页「已就绪」后面显示)
    @Published private(set) var loadSeconds: [ModelSlot: Double] = [:]
    /// 批157:图像槽两步加载——第一步只开文本塔,图像塔的地址、meta 与这个槽位的起跑时刻
    /// 留给第二步(attachImageTower)
    private var pendingImageTower: (url: URL, meta: ModelMeta, started: Date)?
    /// 批157:SigLIP 文本塔已就绪。单独发一个 @Published,是因为文本塔就绪时图像槽的状态
    /// 还是 .loading(要等图像塔),不发信号的话搜索页的「模型加载中」横幅不会及时消失
    @Published private(set) var siglipTextReady = false

    var bothReady: Bool { textState == .ready && imageState == .ready }
    var anyReady: Bool { textState == .ready || imageState == .ready }
    /// 搜索可用的最低条件:SigLIP 文本塔就绪(照片搜索的查询向量只走它)。批157 起图像塔
    /// 可以晚于它——2GB 的图像模型只给新照片 / 视频帧建索引时用,不该拖住搜索。
    /// 不把 Gemma 算进来:它只管文件名 / 文件内容的语义检索,照片搜索用不到
    var searchReady: Bool { siglipTextReady }
    /// 图像塔是否已挂上(要嵌图的索引路径只认这里;别拿 siglip != nil 当图像模型就绪——
    /// 批157 起文本塔先就绪,图像塔随后才挂上)
    var imageTowerReady: Bool { imageState == .ready && siglip?.imageReady == true }

    /// SigLIP 分数门槛缩放系数(so400m 校准,2026-10-22 CI 双模型分布实测):
    /// so400m 命中带均值 = base × 0.87(0.0838 vs 0.0963),噪声带同向下移
    /// (0.0635 vs 0.0731)——绝对门槛不可跨模型迁移,按激活槽位整体缩放。
    /// 性别对比线 0.010 是边距制且 so400m 实测边距 女照+0.039/男照-0.047
    /// 余量充足,不缩。搜索侧 SigLIP 门槛全部乘此系数;Gemma 文件门槛
    /// 与图像模型无关,不缩。
    /// 2.0:恒为 0.87——此前按激活槽位在 0.87/1.0 间二选一,1.0 那档属于 Base。
    /// 批148 核实(2026-11-26,只读复算,系数不动):0.87 是在 so400m patch16-384
    /// 上测的(校准矩阵当时的型号),产线是 patch14-384。拿批141 跑出的 patch14
    /// 分布对 Base 的两份分布复算(体检原串写法):命中带均值比 0.865~0.871
    /// (patch16 是 0.870),噪声带 0.851~0.854;命中带高出噪声带 1.34 倍
    /// (Base 1.31~1.32、patch16 1.32),两条带的相对位置没变——系数照用。
    /// 逐词比值 0.53~1.21 并不均匀(探针里 微信/身份证 偏低、女人 偏高),
    /// 单一系数是折中,patch16 时同样如此
    var siglipGateScale: Float { 0.87 }

    /// 批95:视觉查询向量按模型分流。依据=2026-10-28 A/B 矩阵(build 248,
    /// so400m 上三族 3:0:海边日落召回+67%且增量恰为最相关橙桶/猫族多真猫
    /// 零误报/驾驶证唯一真证过尾线):so400m 的中文原生理解实测强劲。
    /// **矩阵只在 so400m 上跑过,标准版的原生中文是从未测量的空白格**——
    /// 批71 前科:so400m 结论不跨模型迁移(licence 翻案三批)。故 so400m
    /// 激活时裸搜/语义: 的中文词走模板壳+中文原文;标准版保持词典英译
    /// 原配行为(nativeZh=false 逐字不变)。范围=单词语径与 语义:多词交集径
    /// (runMainSearch;批106 曾扩到平均径,批107 平均退役改交集,原料沿用);
    /// AND 线逐词/英文桥维持模板——各待独立 A/B 后再动。
    /// 原生语义: 前缀不受此条件约束(显式实验意图,任何模型无条件直通)。
    /// 2.0:恒为 true——「标准版保持词典英译」那一半随 Base 停用而消失。siglipQuery
    /// 的 nativeZh 形参与各调用点原样不动:其中一处默认 false(英文桥维持模板)
    /// 是批95 的既有边界,不是遗漏
    var preferNativeZhQuery: Bool { true }

    init() {
        // 批157:模型加载耗时的日志出口(ModelLoadTiming 自己不引用 DiagLog——那个文件
        // 要能被人脸测试等流水线单独编译);要在任何模型加载之前装好
        ModelLoadTiming.setSink { IndexingCoordinator.DiagLog.shared.event($0) }
        migrateImageSlot()
    }

    /// 2.0 图像模型槽位迁移(纯目录状态驱动,幂等,中途被杀下次启动按现状续做)。
    /// 1.0 有两个图像槽位:ImageEmbedder(Base,768 维)与 ImageEmbedderPro
    /// (so400m,1152 维);2.0 只留 so400m,且唯一槽位名=ImageEmbedder(与
    /// so400m 包 zip 内的目录名一致)。按磁盘现状分三种(下方分支顺序为
    /// ①、③、②:先判 Pro 目录在不在,再判 Pro 可不可用):
    ///   ① 两个都在(装过 so400m 的 1.0 设备):删 Base,把 Pro 目录改名为
    ///      ImageEmbedder。此前就在用 so400m 的设备向量库已是 1152 维,维度
    ///      比对不触发,零重嵌;此前选的是标准(库里 768 维)则照常全量重嵌
    ///   ② 只有 Base(只装过 1.0 完整包):Base 停用,删除;图像模型显示未安装,
    ///      提示去下载 so400m。不保留 Base 当降级(理由见 minImageModelDim)
    ///   ③ Pro 目录在但不可用(meta 读不出、维度不达标,或缺已编译的模型目录——
    ///      1.0 里会静默退回 Base):删除;Base 照②处理,用户重下 so400m
    /// ①里改名失败:清掉 Pro 目录收敛到「图像模型未安装」并提示重装,不重试
    /// 本函数只负责清理,「不用 Base」的最终防线在 loadSlot 的 dim 守卫——
    /// 这里删除失败时下次启动再试,期间也不会加载 Base
    private func migrateImageSlot() {
        let fm = FileManager.default
        let root = supportRoot
        let baseDir = root.appendingPathComponent(ModelSlot.image.rawValue)
        let proDir = root.appendingPathComponent("ImageEmbedderPro")

        func storedDim(_ dir: URL) -> Int? {
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("meta.json")) else { return nil }
            return (try? JSONDecoder().decode(ModelMeta.self, from: data))?.dim
        }
        /// meta 可解出维度且达标,且两个已编译模型目录都在。只看 meta 不够:1.0 的
        /// 导入是逐文件复制/编译,中途失败会留下「有 meta.json、没 mlmodelc」的
        /// 半成品——2.0 验收 low 指出,拿它当可用会把 Base 删掉后加载失败且无兜底
        func proDim(_ dir: URL) -> Int? {
            guard let dim = storedDim(dir), dim >= ModelManager.minImageModelDim,
                  fm.fileExists(atPath: dir.appendingPathComponent("SiglipImage.mlmodelc").path),
                  fm.fileExists(atPath: dir.appendingPathComponent("SiglipText.mlmodelc").path)
            else { return nil }
            return dim
        }
        // 目录不存在视为已清;删除失败只记日志,由 loadSlot 守卫兜底
        func wipe(_ dir: URL) -> Bool {
            guard fm.fileExists(atPath: dir.path) else { return true }
            do {
                try fm.removeItem(at: dir)
                return true
            } catch {
                IndexingCoordinator.DiagLog.shared.event(
                    "模型槽位迁移(2.0):删除 \(dir.lastPathComponent) 失败(\(error.localizedDescription))")
                return false
            }
        }

        if let dim = proDim(proDir) {
            // ① Pro 可用:先删 Base(删不掉就原样不动,下次再试),再把 Pro 改名
            if wipe(baseDir) {
                do {
                    try fm.moveItem(at: proDir, to: baseDir)
                    migrationNote = "已升级到 2.0:图像模型统一为 so400m,旧的标准模型已移除"
                    IndexingCoordinator.DiagLog.shared.event(
                        "模型槽位迁移(2.0):Pro(dim=\(dim))已改名为 ImageEmbedder,Base 已删")
                } catch {
                    // 改名失败就清掉失败现场(Pro 目录),收敛到「图像模型未安装」:
                    // 留着它,下次启动会重回本分支——先删掉用户按提示重装进
                    // ImageEmbedder 的 so400m,再改名一次(2.0 验收 low)
                    _ = wipe(proDir)
                    migrationNote = "已升级到 2.0:图像模型目录迁移失败,请在「模型在线更新」重装 so400m"
                    IndexingCoordinator.DiagLog.shared.event(
                        "模型槽位迁移(2.0):改名失败(\(error.localizedDescription)),已清 Pro 目录")
                }
            }
        } else if fm.fileExists(atPath: proDir.path) {
            // ③ Pro 不可用(meta 读不出、维度不达标,或缺已编译模型目录)
            _ = wipe(proDir)
            if wipe(baseDir) {
                migrationNote = "已升级到 2.0:原有图像模型不可用,请在「模型在线更新」下载 so400m"
            }
            IndexingCoordinator.DiagLog.shared.event("模型槽位迁移(2.0):Pro 不可用,已清两个图像目录")
        } else if let dim = storedDim(baseDir), dim < Self.minImageModelDim {
            // ② 只有 Base
            if wipe(baseDir) {
                migrationNote = "已升级到 2.0:标准图像模型已停用并移除,请在「模型在线更新」下载 so400m"
            }
            IndexingCoordinator.DiagLog.shared.event("模型槽位迁移(2.0):仅 Base(dim=\(dim)),已删")
        }
        // 1.0 的槽位选择记账,2.0 无意义
        UserDefaults.standard.removeObject(forKey: "imageModelUsePro")
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
        // 并行加载(第二十四批):iOS 把后台进程杀掉后重开=冷启动,模型必须
        // 从盘重载(进程级无解);此前三槽串行,总等待=三者之和,且排队槽位
        // 显示「未安装」吓人。改并行:总等待≈最慢一个,排队槽位由设置页
        // 按 loadPassRunning 显示「加载中(排队)」
        // 批157(用户:冷启动要等几分钟):分两步——第一步只开两个文本塔,搜索此刻即可用;
        // 第二步(与第一步错开,不抢 IO 和 CPU)才开 2GB 的图像塔与人脸模型——它们只有
        // 给新内容建索引时才用
        loadPassRunning = true
        defer { loadPassRunning = false }
        await withTaskGroup(of: Void.self) { group in
            if textState != .ready { group.addTask { await self.loadSlot(.text) } }
            // 图像槽第一步:准备文件 + 开文本塔(siglip 从此非空,imageState 仍是 loading)
            if imageState != .ready { group.addTask { await self.loadSlot(.image) } }
        }
        if searchReady {
            ModelLoadTiming.log("SigLIP文本", "搜索可用(文本塔就绪)", from: ModelLoadTiming.processStart)
        }
        await withTaskGroup(of: Void.self) { group in
            if pendingImageTower != nil { group.addTask { await self.attachImageTower() } }
            group.addTask { await self.loadSlot(.face) }   // 人脸模型可选,缺时静默降级
        }
    }

    /// 槽位就绪:改状态、记这一轮的秒数(设置页显示)、写一行耗时
    private func markReady(_ slot: ModelSlot, since t0: Date) {
        setState(.ready, for: slot)
        loadSeconds[slot] = Date().timeIntervalSince(t0)
        ModelLoadTiming.log(slot.rawValue, "就绪", from: t0)
    }

    /// 图像槽第二步:把图像塔挂到已就绪的 SigLIP 文本塔上(2GB 的模型只在这里打开)。
    /// MLModel 的初始化是同步调用,放到后台线程里做,不占主线程
    private func attachImageTower() async {
        guard let embedder = siglip, let pending = pendingImageTower else { return }
        pendingImageTower = nil
        let url = pending.url
        let attached = await Task.detached(priority: .userInitiated) { () -> Result<Void, Error> in
            do {
                try embedder.attachImageModel(url)
                return .success(())
            } catch {
                return .failure(error)
            }
        }.value
        switch attached {
        case .failure(let error):
            // 文本塔照常可用(搜索不受影响);要嵌图的索引路径按 imageTowerReady 跳过
            setState(.failed(error.localizedDescription), for: .image)
        case .success:
            let meta = pending.meta
            markReady(.image, since: pending.started)
            if textState == .ready { detailText = "" }
            // 记录当前图像模型维度;IndexingCoordinator 同步前与库内向量维度
            // 比对,不一致即清空旧 clip 向量全量重建(so400m 等在线升级换模型)
            let prevDim = UserDefaults.standard.object(forKey: "clipModelDim") as? Int
            UserDefaults.standard.set(meta.dim, forKey: "clipModelDim")
            // 维度变了=换了图像模型:广播触发自动同步清旧向量全量重嵌。
            // 1.0 由 setImageModel 发(标准↔so400m 切换);2.0 切换逻辑已删,
            // 导入新的图像模型包后在这里发。同模型重装维度不变,不广播
            // (从未记录过维度=0 也算变化,与 1.0 setImageModel 的判定等价)
            if (prevDim ?? 0) != meta.dim {
                NotificationCenter.default.post(name: .clipModelChanged, object: nil)
            }
        }
    }

    private func loadSlot(_ slot: ModelSlot) async {
        setState(.loading, for: slot)
        let t0 = Date()      // 批157:这个槽位这一轮加载的起跑时刻
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
        ModelLoadTiming.log(slot.rawValue, "准备文件", from: t0)

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
                    markReady(.text, since: t0)
                    if textState == .ready && imageState == .ready { detailText = "" }
                case "clip":
                    // 2.0 守卫(「不用 Base」的最终防线):迁移没删掉/塞进了旧包时,
                    // 宁可图像模型停在未安装,也不拿 Base 配 so400m 的门槛系数与
                    // 中文原生查询跑搜索
                    guard meta.dim >= Self.minImageModelDim else {
                        siglip = nil
                        siglipTextReady = false
                        setState(.missing, for: slot)
                        detailText = "标准图像模型(Base)已停用:请在 设置→模型在线更新 下载 so400m"
                        return
                    }
                    let imgURL = dir.appendingPathComponent("SiglipImage.mlmodelc")
                    let txtURL = dir.appendingPathComponent("SiglipText.mlmodelc")
                    guard let tokURL else { throw MSError("缺少 tokenizer.json") }
                    // 批157:两步加载——这里只开文本塔,搜索从此可用;2GB 的图像塔留到
                    // loadAll 第二步(attachImageTower)再开,imageState 到那时才就绪。
                    // 维度记账与换模型广播也挪到那时(要等图像塔真的挂上)
                    let embedder = try await SigLIPEmbedder(
                        textModelURL: txtURL,
                        tokenizerFolder: tokURL.deletingLastPathComponent(),
                        imageSize: meta.image_size ?? 256,
                        mean: meta.image_mean ?? 0.5, std: meta.image_std ?? 0.5)
                    siglip = embedder
                    siglipTextReady = true
                    pendingImageTower = (imgURL, meta, t0)
                case "face":
                    let modelURL = dir.appendingPathComponent("Face.mlmodelc")
                    face = try FaceEmbedder(modelURL: modelURL,
                                            inputName: meta.input ?? "input.1",
                                            outputName: meta.output ?? "embedding")
                    markReady(.face, since: t0)
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
    /// 与 export_models 生成 manifest.fingerprint 的拼法同规,改一处须同步另一处。
    /// 2.0:基础包(full)只含文本+人脸,指纹去掉 I 段(T|F);图像模型包
    /// (key 仍为 so400m)的 P 段改从唯一图像槽位取
    func installedFingerprint(forPackage key: String) -> String? {
        let slots: [ModelSlot] = key == "so400m" ? [.image] : [.text, .face]
        var parts: [String] = []
        for slot in slots {
            let metaURL = supportRoot.appendingPathComponent(slot.rawValue)
                .appendingPathComponent("meta.json")
            guard let data = try? Data(contentsOf: metaURL),
                  let meta = try? JSONDecoder().decode(ModelMeta.self, from: data) else { return nil }
            switch slot {
            case .text: parts.append("T:\(meta.dim)")
            case .image: parts.append("P:\(meta.dim)")
            case .face: parts.append("F:\(meta.dim):\(meta.model_size ?? 0)")
            }
        }
        return parts.joined(separator: "|")
    }

    // MARK: - 从 zip 导入模型包

    /// zip 根目录应包含 TextEmbedder/ ImageEmbedder/ FaceEmbedder/(各含
    /// .mlpackage、tokenizer.json、meta.json),目录名即槽位名。2.0 起图像模型
    /// 只有 so400m 一种:含 Base(dim 低于 minImageModelDim)的 ImageEmbedder
    /// 目录一律跳过不导入——塞进 1.0 时代的旧完整包时,同包的文本/人脸照常导入,
    /// 不整包报错
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
        var skippedBase = false
        for name in try FileManager.default.contentsOfDirectory(atPath: extractDir.path) {
            guard let slot = ModelSlot(rawValue: name) else { continue }
            let src = extractDir.appendingPathComponent(name)
            let metaURL = src.appendingPathComponent("meta.json")
            guard FileManager.default.fileExists(atPath: metaURL.path) else { continue }

            // 2.0:停用的 Base 图像模型不导入(1.0 在此按 dim 把高维包路由到
            // .imagePro 槽;2.0 只有一个图像槽,目录名即槽位,不再需要路由)
            if slot == .image,
               let data = try? Data(contentsOf: metaURL),
               let meta = try? JSONDecoder().decode(ModelMeta.self, from: data),
               meta.dim < Self.minImageModelDim {
                skippedBase = true
                continue
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
        guard found > 0 else {
            throw MSError(skippedBase
                ? "压缩包里只有已停用的标准图像模型(Base);请在「模型在线更新」下载 so400m"
                : "压缩包里没有找到 TextEmbedder / ImageEmbedder / FaceEmbedder 模型目录")
        }
        if skippedBase { detailText = "已跳过停用的标准图像模型(Base)" }

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
            // 2.0:图像模型可能确实没装(Base 停用后尚未下载 so400m;1.0 有 Base
            // 兜底,图像模型总在)。加载轮次已结束仍是「未安装」就不必干等满超时
            if !loadPassRunning, imageState == .missing { return textState == .ready }
            // 批157:图像模型加载失败也干等不到——加载轮次已结束就直接放行(文本侧照常工作)
            if !loadPassRunning, case .failed = imageState { return textState == .ready }
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
