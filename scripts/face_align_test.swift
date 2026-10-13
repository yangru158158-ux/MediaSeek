// FaceKit 对齐几何测试(macOS CI 直接编译运行):
// 132 版 bug = CTM 变换复合顺序写反 + 关键点 y 轴未翻转 → 所有对齐输出是黑图
// → 全部人脸嵌入相同 → 36 张脸归一组。本测试用已知眼位的合成图验证:
// ① 顶行朝向(上白下黑条不颠倒) ② 眼标落在 ArcFace 模板位 ③ 斜眼旋转归一
// ④ 不同图案对齐后不塌缩(132 bug 下两条曲线都是全黑、差=0)
import Foundation
import CoreGraphics
import CoreML
import Accelerate

// FaceKit.swift 引用的错误类型与 MLModel 扩展(定义在 App 工程的
// VectorStore.swift / EmbeddingModels.swift,后者依赖 SPM 包不便同编)
struct MSError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

extension MLModel {
    func runEmbedding(inputs: [String: MLMultiArray], outputName: String) throws -> [Float] {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues {
            MLFeatureValue(multiArray: $0)
        })
        let out = try prediction(from: provider)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw MSError("模型输出缺少 \(outputName)")
        }
        let v = arr.toFloatArrayTest()
        let n = max((vDSP.dot(v, v) as Float).squareRoot(), 1e-8)
        return v.map { $0 / n }
    }
}

extension MLMultiArray {
    func toFloatArrayTest() -> [Float] {
        (0..<count).map { Float(self[$0].doubleValue) }
    }
}

@main
struct FaceAlignTest {
    static func main() {
        var failed = 0
        func check(_ name: String, _ ok: Bool, _ detail: String) {
            print((ok ? "PASS " : "FAIL ") + name + " | " + detail)
            if !ok { failed += 1 }
        }

        let w = 400, h = 300
        // 合成图:顶部 y<60 纯白条、底部 y≥240 纯黑条、中间平滑条纹(相位=seed),
        // 眼位画 12×12 纯黑方块
        func makeImage(seed: Double, le: CGPoint, re: CGPoint) -> CGImage {
            var buf = [UInt8](repeating: 0, count: w * h * 4)
            for y in 0..<h {
                for x in 0..<w {
                    let i = (y * w + x) * 4
                    let v: Double
                    if y < 72 { v = 250 }   # 白条须覆盖 row5 的采样点(源 y≈69)
                    else if y >= 240 { v = 5 }
                    else { v = 128 + 60 * sin(Double(x) * .pi / 40 + seed) }
                    let c = UInt8(max(0, min(255, v)))
                    buf[i] = c; buf[i + 1] = c; buf[i + 2] = c; buf[i + 3] = 255
                }
            }
            func mark(_ p: CGPoint) {
                for dy in -6..<6 { for dx in -6..<6 {
                    let x = Int(p.x) + dx, y = Int(p.y) + dy
                    guard x >= 0, x < w, y >= 0, y < h else { continue }
                    let i = (y * w + x) * 4
                    buf[i] = 0; buf[i + 1] = 0; buf[i + 2] = 0
                }}
            }
            mark(le); mark(re)
            var src = buf
            let ctx = CGContext(data: &src, width: w, height: h,
                                bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            return ctx.makeImage()!
        }

        // 读回 112×112 对齐图的 RGBA(row0=显示顶行;identity draw 即不翻,CI 校准)
        func rgba(_ img: CGImage) -> [UInt8] {
            let side = 112
            var buf = [UInt8](repeating: 0, count: side * side * 4)
            let ctx = CGContext(data: &buf, width: side, height: side,
                                bitsPerComponent: 8, bytesPerRow: side * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: side, height: side))
            return buf
        }
        func gray(_ buf: [UInt8], _ x: Int, _ y: Int) -> Double { Double(buf[(y * 112 + x) * 4]) }
        func rowAvg(_ buf: [UInt8], _ y: Int) -> Double {
            (0..<112).map { gray(buf, $0, y) }.reduce(0, +) / 112
        }

        // A:水平眼;B:斜眼(同一 seed=同一图案系);C:同眼位不同条纹相位
        let leA = CGPoint(x: 150, y: 200), reA = CGPoint(x: 250, y: 200)
        let leB = CGPoint(x: 140, y: 220), reB = CGPoint(x: 260, y: 170)
        let imgA = makeImage(seed: 0, le: leA, re: reA)
        let imgB = makeImage(seed: 0, le: leB, re: reB)
        let imgC = makeImage(seed: 2.2, le: leA, re: reA)

        guard let alA = FaceKit.aligned(source: imgA, leftEye: leA, rightEye: reA),
              let alB = FaceKit.aligned(source: imgB, leftEye: leB, rightEye: reB),
              let alC = FaceKit.aligned(source: imgC, leftEye: leA, rightEye: reA) else {
            print("FAIL 对齐返回 nil")
            exit(1)
        }
        let a = rgba(alA), b = rgba(alB), c = rgba(alC)

        // ① 顶行朝向:上白下黑(132 bug 输出全黑,此断言即死)
        let topAvg = rowAvg(a, 5), botAvg = rowAvg(a, 106)
        check("顶行朝向(上白下黑)", topAvg > 180 && botAvg < 80,
              String(format: "row5=%.0f row106=%.0f", topAvg, botAvg))

        // ② 眼标落模板位:(38,52) 与 (73,52) 附近应是黑块
        let eyeL = (gray(a, 38, 52) + gray(a, 39, 52)) / 2
        let eyeR = (gray(a, 73, 52) + gray(a, 74, 52)) / 2
        check("眼标落模板位", eyeL < 80 && eyeR < 80,
              String(format: "L=%.0f R=%.0f", eyeL, eyeR))

        // ③ 旋转归一:斜眼 B 对齐后眼标同样落模板位
        let eyeLB = (gray(b, 38, 52) + gray(b, 39, 52)) / 2
        let eyeRB = (gray(b, 73, 52) + gray(b, 74, 52)) / 2
        check("斜眼旋转归一", eyeLB < 80 && eyeRB < 80,
              String(format: "L=%.0f R=%.0f", eyeLB, eyeRB))

        // ④ 非塌缩:同眼位不同图案,对齐后差异显著(132 bug 两者全黑差=0)
        var diff = 0.0
        for i in 0..<(112 * 112) { diff += abs(Double(a[i * 4]) - Double(c[i * 4])) }
        let meanDiff = diff / Double(112 * 112)
        check("不同图案不塌缩", meanDiff > 15, String(format: "平均差=%.1f", meanDiff))

        print(failed == 0 ? "== 全部通过 ==" : "== \(failed) 项失败 ==")
        exit(failed == 0 ? 0 : 1)
    }
}
