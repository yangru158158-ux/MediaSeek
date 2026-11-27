import Foundation
import CoreML
import Tokenizers   // swift-transformers
import Accelerate

// MARK: - 通用工具

extension MLMultiArray {
    func toFloatArray() -> [Float] {
        let count = count
        switch dataType {
        case .float32:
            let p = dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: p, count: count))
        case .float16:
            let p = dataPointer.assumingMemoryBound(to: UInt16.self)
            return UnsafeBufferPointer(start: p, count: count).map { Float(Float16(bitPattern: $0)) }
        default:
            return (0..<count).map { Float(self[$0].doubleValue) }
        }
    }
}

extension MLModel {
    /// 运行"单输入单输出 embedding"模型,返回归一化向量
    func runEmbedding(inputName: String, input: MLMultiArray,
                      outputName: String = "embedding") throws -> [Float] {
        try runEmbedding(inputs: [inputName: input], outputName: outputName)
    }

    /// 运行"多输入单输出 embedding"模型,返回归一化向量
    func runEmbedding(inputs: [String: MLMultiArray],
                      outputName: String = "embedding") throws -> [Float] {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues {
            MLFeatureValue(multiArray: $0)
        })
        let out = try prediction(from: provider)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw MSError("模型输出缺少 \(outputName)")
        }
        // 统一归一化:图/文嵌入单位化后点积即余弦,检索门槛不随输出尺度漂移
        return VectorMath.normalized(arr.toFloatArray())
    }
}

enum VectorMath {
    static func normalized(_ v: [Float]) -> [Float] {
        let squared = vDSP.dot(v, v)
        let n = max(sqrt(squared), 1e-8)
        return v.map { $0 / n }
    }
}

// MARK: - 分词器

/// 封装 swift-transformers 的 AutoTokenizer,加载本地 tokenizer.json
final class HFTokenizer {
    private let tokenizer: any Tokenizer
    static let maxSequenceGemma = 512
    static let maxSequenceSiglip = 64

    init(folder: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: folder)
    }

    /// 编码并截断。保留末位 token(通常是 EOS/SOS 语义锚点)
    func encode(_ text: String, maxLength: Int) throws -> [Int] {
        var ids = tokenizer.encode(text: text)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength - 1)) + [ids.last!]
        }
        return ids
    }

    /// SigLIP 专用:编码后右填充到定长(pad id=0)。attention_mask 恒填 1
    /// (全部位置参与注意=训练语义)——第四十九批(2026-10-21「搜女人 4217
    /// 全噪声/搜 woman 0」事故):SigLIP2 训练与官方 processor 都不产
    /// attention_mask,pad 参与注意才是训练语义;导出图把掩码做成输入
    /// ((1-mask)×-30000 加性偏置),此前传真实 0/1 掩码把 pad 掩掉,末位
    /// 池化锚点读到偏离训练分布的上下文,文本嵌入内容失明(实测猫照对
    /// "This is a photo of a cat." 仅 -0.0120;全 1 掩码 +0.1126,与官方
    /// 无掩码管线一致度 1.000000)。掩码输入仅为兼容已发布模型包的图形状。
    func encodePadded(_ text: String, maxLength: Int) throws -> (ids: [Int], mask: [Int]) {
        var ids = try encode(text, maxLength: maxLength)
        if ids.count < maxLength {
            ids.append(contentsOf: [Int](repeating: 0, count: maxLength - ids.count))
        }
        return (ids, [Int](repeating: 1, count: maxLength))
    }
}

// MARK: - EmbeddingGemma 文本嵌入器

/// EmbeddingGemma(或未来的 EmbeddingGemma 2,接口一致)。
/// 官方推荐 prompt 模板:查询用 `task: search result | query: …`,文档用 `title: none | text: …`。
final class GemmaTextEmbedder {
    let space = "gemma"
    private let model: MLModel
    private let tokenizer: HFTokenizer

    init(modelURL: URL, tokenizerFolder: URL) async throws {
        // 批157:打开模型与分词器分开计时
        let t0 = Date()
        self.model = try MLModel(contentsOf: modelURL, configuration: ModelLoadTiming.configuration())
        ModelLoadTiming.log("EmbeddingGemma", "打开模型", from: t0)
        let t1 = Date()
        self.tokenizer = try await HFTokenizer(folder: tokenizerFolder)
        ModelLoadTiming.log("EmbeddingGemma", "分词器", from: t1)
    }

    static func queryPrompt(_ text: String) -> String { "task: search result | query: \(text)" }
    static func documentPrompt(_ text: String) -> String { "title: none | text: \(text)" }

    func embedQuery(_ text: String) throws -> [Float] {
        try embed(Self.queryPrompt(text))
    }

    func embedDocument(_ text: String) throws -> [Float] {
        try embed(Self.documentPrompt(text))
    }

    private func embed(_ prompt: String) throws -> [Float] {
        let ids = try tokenizer.encode(prompt, maxLength: HFTokenizer.maxSequenceGemma)
        let arr = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        let p = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, t) in ids.enumerated() { p[i] = Int32(t) }
        let t0 = Date()
        let v = try model.runEmbedding(inputName: "input_ids", input: arr)
        ModelLoadTiming.firstInference("EmbeddingGemma", from: t0)
        return v
    }
}

// MARK: - SigLIP 图文嵌入器(图像与查询文本同一向量空间)

final class SigLIPEmbedder {
    let space = "clip"
    private let textModel: MLModel
    private let tokenizer: HFTokenizer
    private let imageSize: Int
    private let mean: Float
    private let std: Float
    private let needsAttentionMask: Bool
    // 批157(用户:冷启动要等几分钟):图像塔与文本塔分开加载。查询向量只走文本塔
    // (搜索只需要它);图像塔(so400m 包 2GB 的大头)只给新照片 / 视频帧 / 导入文件
    // 建索引时用——原先两者在同一个 init 里一起打开,图像塔没加载完搜索也用不了
    private let imageLock = NSLock()
    private var imageModelBox: MLModel?

    /// 图像塔是否已挂上;没挂上时 embedImage 抛错
    var imageReady: Bool {
        imageLock.lock()
        defer { imageLock.unlock() }
        return imageModelBox != nil
    }

    /// 只开文本塔 + 分词器(搜索可用的最低条件);图像塔由 attachImageModel 随后挂上
    init(textModelURL: URL, tokenizerFolder: URL,
         imageSize: Int = 256, mean: Float = 0.5, std: Float = 0.5) async throws {
        // 批157:打开模型与分词器分开计时
        let t0 = Date()
        self.textModel = try MLModel(contentsOf: textModelURL, configuration: ModelLoadTiming.configuration())
        ModelLoadTiming.log("SigLIP文本", "打开模型", from: t0)
        let t1 = Date()
        self.tokenizer = try await HFTokenizer(folder: tokenizerFolder)
        ModelLoadTiming.log("SigLIP文本", "分词器", from: t1)
        self.imageSize = imageSize
        self.mean = mean
        self.std = std
        // 新版模型包文本塔要求定长 64 双输入;旧包无 attention_mask 输入则走旧路径
        self.needsAttentionMask = textModel.modelDescription
            .inputDescriptionsByName["attention_mask"] != nil
    }

    /// 挂上图像塔(可能很慢:2GB 的模型;调用方放到后台线程里调)
    func attachImageModel(_ url: URL) throws {
        let t0 = Date()
        let model = try MLModel(contentsOf: url, configuration: ModelLoadTiming.configuration())
        ModelLoadTiming.log("SigLIP图像", "打开模型", from: t0)
        imageLock.lock()
        imageModelBox = model
        imageLock.unlock()
    }

    // 批92:查询向量缓存。ANE 浮点推理非确定(Apple 文档:float16 执行路径
    // 差异),同一文本每次 embedQuery 产出微小不同的向量——贴门槛的条目
    // (驾驶证卡对 "driver license card" 在 so400m ≈0.05 贴着地板线)在
    // 线两边横跳,同词连搜时出时不出(245 装机实锤:第一次出、后续不出、
    // 再搜又出;VectorStore 检索本身确定性,唯一变化的是查询向量)。
    // 同文本复用首次向量:会话内同词搜索结果完全可复现;缓存随 embedder
    // 实例生灭,模型切换自动失效。NSLock 防 Task.detached 并发竞态
    private var queryVecCache: [String: [Float]] = [:]
    private let queryCacheLock = NSLock()

    func embedQuery(_ text: String) throws -> [Float] {
        queryCacheLock.lock()
        if let v = queryVecCache[text] {
            queryCacheLock.unlock()
            return v
        }
        queryCacheLock.unlock()
        let t0 = Date()
        let v = try embedQueryCompute(text)
        ModelLoadTiming.firstInference("SigLIP文本", from: t0)
        queryCacheLock.lock()
        if queryVecCache.count > 512 { queryVecCache.removeAll() }   // 会话级防膨胀
        queryVecCache[text] = v
        queryCacheLock.unlock()
        return v
    }

    private func embedQueryCompute(_ text: String) throws -> [Float] {
        if needsAttentionMask {
            let padded = try tokenizer.encodePadded(text, maxLength: HFTokenizer.maxSequenceSiglip)
            let ids = try MLMultiArray(shape: [1, NSNumber(value: padded.ids.count)], dataType: .int32)
            let mask = try MLMultiArray(shape: [1, NSNumber(value: padded.mask.count)], dataType: .int32)
            let pi = ids.dataPointer.assumingMemoryBound(to: Int32.self)
            let pm = mask.dataPointer.assumingMemoryBound(to: Int32.self)
            for (i, t) in padded.ids.enumerated() { pi[i] = Int32(t) }
            // 掩码在 encodePadded 恒填 1(全部位置参与注意=训练语义),直拷;
            // 第四十九批事故与实测证据见 encodePadded 文档
            for (i, m) in padded.mask.enumerated() { pm[i] = Int32(m) }
            return try textModel.runEmbedding(inputs: ["input_ids": ids, "attention_mask": mask])
        }
        let ids = try tokenizer.encode(text, maxLength: HFTokenizer.maxSequenceSiglip)
        let arr = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        let p = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, t) in ids.enumerated() { p[i] = Int32(t) }
        return try textModel.runEmbedding(inputName: "input_ids", input: arr)
    }

    func embedImage(_ cg: CGImage) throws -> [Float] {
        imageLock.lock()
        let model = imageModelBox
        imageLock.unlock()
        guard let model else { throw MSError("图像模型尚未加载完成") }
        let pixels = try ImagePreprocess.normalizedNCHW(cgImage: cg, size: imageSize, mean: mean, std: std)
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: imageSize), NSNumber(value: imageSize)],
                                   dataType: .float32)
        let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
        pixels.withUnsafeBufferPointer { src in
            _ = memcpy(p, src.baseAddress, pixels.count * MemoryLayout<Float>.size)
        }
        let t0 = Date()
        let v = try model.runEmbedding(inputName: "pixel_values", input: arr)
        ModelLoadTiming.firstInference("SigLIP图像", from: t0)
        return v
    }
}

// MARK: - 图像预处理

enum ImagePreprocess {
    /// 缩放到 size×size,归一化后输出 NCHW 排列的浮点像素
    static func normalizedNCHW(cgImage: CGImage, size: Int, mean: Float, std: Float) throws -> [Float] {
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        let ctx = CGContext(data: &buf,
                            width: size, height: size,
                            bitsPerComponent: 8, bytesPerRow: size * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let ctx else { throw MSError("创建图像上下文失败") }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        let plane = size * size
        var out = [Float](repeating: 0, count: plane * 3)
        for p in 0..<plane {
            let i = p * 4
            out[p] = (Float(buf[i]) / 255.0 - mean) / std
            out[plane + p] = (Float(buf[i + 1]) / 255.0 - mean) / std
            out[plane * 2 + p] = (Float(buf[i + 2]) / 255.0 - mean) / std
        }
        return out
    }
}
