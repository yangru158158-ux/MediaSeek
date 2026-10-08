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
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
        let out = try prediction(from: provider)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw MSError("模型输出缺少 \(outputName)")
        }
        return arr.toFloatArray()
    }
}

enum VectorMath {
    static func normalized(_ v: [Float]) -> [Float] {
        var norm: Float = 0
        vDSP.dot(v, v, &norm)
        let n = max(sqrt(norm), 1e-8)
        return v.map { $0 / n }
    }
}

// MARK: - 分词器

/// 封装 swift-transformers 的 AutoTokenizer,加载本地 tokenizer.json
final class HFTokenizer {
    private let tokenizer: any Tokenizer
    static let maxSequenceGemma = 512
    static let maxSequenceSiglip = 64

    init(folder: URL) throws {
        self.tokenizer = try AutoTokenizer.from(modelFolder: folder)
    }

    /// 编码并截断。保留末位 token(通常是 EOS/SOS 语义锚点)
    func encode(_ text: String, maxLength: Int) throws -> [Int] {
        var ids = tokenizer.encode(text: text)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength - 1)) + [ids.last!]
        }
        return ids
    }
}

// MARK: - EmbeddingGemma 文本嵌入器

/// EmbeddingGemma(或未来的 EmbeddingGemma 2,接口一致)。
/// 官方推荐 prompt 模板:查询用 `task: search result | query: …`,文档用 `title: none | text: …`。
final class GemmaTextEmbedder {
    let space = "gemma"
    private let model: MLModel
    private let tokenizer: HFTokenizer

    init(modelURL: URL, tokenizerFolder: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
        self.tokenizer = try HFTokenizer(folder: tokenizerFolder)
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
        return try model.runEmbedding(inputName: "input_ids", input: arr)
    }
}

// MARK: - SigLIP 图文嵌入器(图像与查询文本同一向量空间)

final class SigLIPEmbedder {
    let space = "clip"
    private let textModel: MLModel
    private let imageModel: MLModel
    private let tokenizer: HFTokenizer
    private let imageSize: Int
    private let mean: Float
    private let std: Float

    init(textModelURL: URL, imageModelURL: URL, tokenizerFolder: URL,
         imageSize: Int = 256, mean: Float = 0.5, std: Float = 0.5) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.textModel = try MLModel(contentsOf: textModelURL, configuration: config)
        self.imageModel = try MLModel(contentsOf: imageModelURL, configuration: config)
        self.tokenizer = try HFTokenizer(folder: tokenizerFolder)
        self.imageSize = imageSize
        self.mean = mean
        self.std = std
    }

    func embedQuery(_ text: String) throws -> [Float] {
        let ids = try tokenizer.encode(text, maxLength: HFTokenizer.maxSequenceSiglip)
        let arr = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        let p = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, t) in ids.enumerated() { p[i] = Int32(t) }
        return try textModel.runEmbedding(inputName: "input_ids", input: arr)
    }

    func embedImage(_ cg: CGImage) throws -> [Float] {
        let pixels = try ImagePreprocess.normalizedNCHW(cgImage: cg, size: imageSize, mean: mean, std: std)
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: imageSize), NSNumber(value: imageSize)],
                                   dataType: .float32)
        let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
        pixels.withUnsafeBufferPointer { src in
            _ = memcpy(p, src.baseAddress, pixels.count * MemoryLayout<Float>.size)
        }
        return try imageModel.runEmbedding(inputName: "pixel_values", input: arr)
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
