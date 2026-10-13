// 真实照片端到端对齐工具(macOS CI 编译运行,与 App 同一份 FaceKit.swift):
// 读入照片 → Vision 检测(取最大脸) → ArcFace 对齐 → 导出 112×112 PNG。
// 与 face_align_test.swift 的合成图几何测试互补:合成图只验证几何落位,
// 本工具出的是真实照片上的对齐结果,供 face_e2e_score.py 接 r50 打分断言,
// 对齐图上传 CI 工件可人工目检(眼位/朝向/完整性)。
//
// 用法: face_e2e_align <输出目录> <照片路径>...
import Foundation
import CoreGraphics
import CoreML
import ImageIO
import UniformTypeIdentifiers
import Vision

// FaceKit.swift 引用的错误类型与 MLModel 扩展定义在 App 工程
// (VectorStore.swift / EmbeddingModels.swift,后者依赖 SPM 不便同编)
struct MSError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

extension MLModel {
    func runEmbedding(inputs: [String: MLMultiArray], outputName: String) throws -> [Float] {
        throw MSError("e2e 对齐工具不加载模型,不应走到嵌入")
    }
}

@main
struct FaceE2EAlign {
    static func main() {
        guard CommandLine.arguments.count >= 3 else {
            print("用法: face_e2e_align <输出目录> <照片路径>...")
            exit(2)
        }
        let outDir = URL(fileURLWithPath: CommandLine.arguments[1])
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        var ok = 0
        for arg in CommandLine.arguments.dropFirst(2) {
            let url = URL(fileURLWithPath: arg)
            let name = url.deletingPathExtension().lastPathComponent
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                print("SKIP 无法读取 \(name)")
                continue
            }
            let faces = FaceKit.detect(cg: cg)
            guard let big = faces.max(by: {
                $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
            }) else {
                print("SKIP 未检出人脸 \(name)")
                continue
            }
            guard let aligned = FaceKit.alignedFace(source: cg, observation: big) else {
                print("SKIP 对齐失败 \(name)")
                continue
            }
            let out = outDir.appendingPathComponent("aligned_\(name).png")
            guard let dest = CGImageDestinationCreateWithURL(
                out as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                print("SKIP 输出创建失败 \(name)")
                continue
            }
            CGImageDestinationAddImage(dest, aligned, nil)
            CGImageDestinationFinalize(dest)
            print("OK \(out.lastPathComponent)(检出 \(faces.count) 张脸,取最大)")
            ok += 1
        }
        print("共对齐 \(ok) 张")
        guard ok > 0 else { exit(2) }
    }
}
