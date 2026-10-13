import Foundation
import CoreML
import Vision
import CoreGraphics
import Accelerate

/// 人脸嵌入器:insightface MobileFaceNet(ArcFace 系,512 维)。
/// 检测/关键点用系统 Vision(免费);本模型只负责「这张脸是谁」的身份向量。
/// 预处理约定(insightface 官方):RGB、(px-127.5)/128、112×112;
/// 输出未归一化,MLModel.runEmbedding 统一归一化后点积即余弦。
final class FaceEmbedder {
    let space = "face"
    private let model: MLModel
    private let inputName: String
    private let outputName: String

    init(modelURL: URL, inputName: String, outputName: String) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
        self.inputName = inputName
        self.outputName = outputName
    }

    /// 已对齐的 112×112 人脸图 → 归一化身份向量。
    /// draw 前翻转 CTM:让 buf 第 0 行=图顶行(张量行主序与标准图像一致;
    /// identity CTM 下 CG 会把源图底行画到 row0=上下颠倒,ArcFace 精度会掉)
    func embed(aligned: CGImage) throws -> [Float] {
        let side = 112
        var buf = [UInt8](repeating: 0, count: side * side * 4)
        guard let ctx = CGContext(data: &buf, width: side, height: side,
                                  bitsPerComponent: 8, bytesPerRow: side * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw MSError("人脸图上下文创建失败")
        }
        ctx.interpolationQuality = .high
        ctx.draw(aligned, in: CGRect(x: 0, y: 0, width: side, height: side))
        let plane = side * side
        var out = [Float](repeating: 0, count: plane * 3)
        for p in 0..<plane {
            let i = p * 4
            out[p] = (Float(buf[i]) - 127.5) / 128.0
            out[plane + p] = (Float(buf[i + 1]) - 127.5) / 128.0
            out[plane * 2 + p] = (Float(buf[i + 2]) - 127.5) / 128.0
        }
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: side), NSNumber(value: side)],
                                   dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        out.withUnsafeBufferPointer { src in
            _ = memcpy(ptr, src.baseAddress, out.count * MemoryLayout<Float>.size)
        }
        return try model.runEmbedding(inputs: [inputName: arr], outputName: outputName)
    }
}

/// 人脸检测与 ArcFace 标准对齐。
/// 对齐用手动相似变换逆映射 + 双线性采样,不经 CGContext CTM——
/// 132 版用 CTM 复合实现,变换顺序写反+关键点 y 轴未翻转,所有对齐输出
/// 都被映射到画布外(纯黑 112×112),全部人脸嵌入相同、36 张脸归一组(用户实锤)。
enum FaceKit {
    /// ArcFace 112×112 五点模板双眼参数(insightface arcface_dst,
    /// 标准图像坐标:原点左上、y 向下)。眼中点 (55.9132, 51.5989),瞳距 35.2372
    private static let eyeMid = CGPoint(x: 55.9132, y: 51.5989)
    private static let eyeDistance: CGFloat = 35.2372

    /// 检测人脸(带关键点)。VNDetectFaceLandmarksRequest 自带检测步,
    /// 结果同时含外接框与五官区域;一张合照最多取 maxFaces 张脸,防群照拖垮索引。
    /// 小脸过滤:外接框宽 < minFacePx 像素的脸跳过——远景小人/文字纹理误检的
    /// 对齐图分辨率不足,嵌入质量差只会把不同人挤成一组(135 用户实测大杂烩组)
    static func detect(cg: CGImage, maxFaces: Int = 8, minFacePx: Int = 64) -> [VNFaceObservation] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            let minW = CGFloat(minFacePx) / CGFloat(max(cg.width, 1))
            return Array((request.results ?? [])
                .filter { $0.boundingBox.width >= minW }
                .prefix(maxFaces))
        } catch {
            return []
        }
    }

    /// 按双眼做相似变换对齐(双眼转水平、瞳距缩放到模板值、眼中点平移到模板位)。
    /// ⚠️ 眼位按 x 坐标排序,不信任 Vision 的 leftEye/rightEye 命名——该命名是
    /// 「人自己的左右眼」还是「图像观察者左右」官方文档不明,拿反了会把脸
    /// 旋转 180° 倒立,ArcFace 只认正立对齐脸、区分度崩塌(135 用户实测
    /// 不同性别证件/文字截图/多人生活照聚成 128 张大杂烩组)。
    /// 无关键点时退化为外接框放大 40% 的中心方形。
    static func alignedFace(source: CGImage, observation: VNFaceObservation) -> CGImage? {
        if let lm = observation.landmarks,
           let le = centroid(lm.leftEye, imageSize: source),
           let re = centroid(lm.rightEye, imageSize: source) {
            return aligned(source: source, leftEye: le, rightEye: re)   // aligned 内部按 x 排序,免疫命名歧义
        }
        let w = CGFloat(source.width), h = CGFloat(source.height)
        let bb = observation.boundingBox   // Vision 归一化坐标,原点左下
        let sidePx = min(max(bb.width * w, bb.height * h) * 1.4, min(w, h))
        let cx = bb.midX * w
        let cy = (1 - bb.midY) * h         // 左下原点 → 顶行原点
        let x0 = cx - sidePx / 2, y0 = cy - sidePx / 2
        guard let (sw, sh, buf) = rgbaBuffer(of: source) else { return nil }
        return warp(width: sw, height: sh, buf: buf) { x, y in
            (x0 + (x + 0.5) / 112 * sidePx, y0 + (y + 0.5) / 112 * sidePx)
        }
    }

    /// 纯几何对齐核心(CI 几何测试直接调用):双眼像素坐标(顶行原点 y 向下)
    /// → 112×112 模板位。逆映射:src = mid + R(θ)·((dst - eyeMid)/s),
    /// s = 模板瞳距/实际瞳距,R 把输出水平基向量转到源图双眼方向。
    /// 眼位先按 x 排序:调用方(或 Vision 命名)把左右眼对调也不影响——
    /// 图像上靠左的眼必须画到模板 x=38 侧,对调会把脸旋转 180° 倒立
    static func aligned(source: CGImage, leftEye le: CGPoint, rightEye re: CGPoint) -> CGImage? {
        guard let (sw, sh, buf) = rgbaBuffer(of: source) else { return nil }
        let eyeL = le.x <= re.x ? le : re
        let eyeR = le.x <= re.x ? re : le
        let mid = CGPoint(x: (eyeL.x + eyeR.x) / 2, y: (eyeL.y + eyeR.y) / 2)
        let dx = eyeR.x - eyeL.x, dy = eyeR.y - eyeL.y
        let dist = max(hypot(dx, dy), 1)
        let s = eyeDistance / dist
        let c = dx / dist, sn = dy / dist
        return warp(width: sw, height: sh, buf: buf) { x, y in
            let ox = (x + 0.5 - eyeMid.x) / s
            let oy = (y + 0.5 - eyeMid.y) / s
            return (mid.x + c * ox - sn * oy, mid.y + sn * ox + c * oy)
        }
    }

    /// 关键点区域质心。Vision normalizedPoints 原点在左下,
    /// 换算成顶行原点像素坐标:y = (1 - ny) * height
    private static func centroid(_ region: VNFaceLandmarkRegion2D?, imageSize: CGImage) -> CGPoint? {
        guard let pts = region?.normalizedPoints, !pts.isEmpty else { return nil }
        var sumX: CGFloat = 0, sumY: CGFloat = 0
        for p in pts { sumX += p.x; sumY += p.y }
        return CGPoint(x: sumX / CGFloat(pts.count) * CGFloat(imageSize.width),
                       y: (1 - sumY / CGFloat(pts.count)) * CGFloat(imageSize.height))
    }

    /// 源图解码为 RGBA 缓冲(行主序 = 顶行起、y 向下,与采样坐标一致)。
    /// 实测(CI 几何测试校准):纯 CGBitmapContext 里 draw 在 identity CTM 下
    /// 本就不翻转(翻转坑在 UIKit 交互场景),再加纠偏翻转反而上下颠倒
    private static func rgbaBuffer(of image: CGImage) -> (Int, Int, [UInt8])? {
        let w = image.width, h = image.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &buf, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (w, h, buf)
    }

    /// 通用重采样:对 112×112 每个输出像素,经 map 逆映射到源图坐标,双线性采样;
    /// 越界填黑(ArcFace 对齐的边界约定)。map 的输入输出均为顶行原点坐标。
    /// out 的 y=0 行即对齐输出的显示顶行;bitmap row0 → CGImage 显示顶行,直接出图
    private static func warp(width: Int, height: Int, buf: [UInt8],
                             map: (CGFloat, CGFloat) -> (CGFloat, CGFloat)) -> CGImage? {
        let side = 112
        var out = [UInt8](repeating: 0, count: side * side * 4)
        let fw = CGFloat(width), fh = CGFloat(height)
        for y in 0..<side {
            for x in 0..<side {
                let (sx, sy) = map(CGFloat(x), CGFloat(y))
                let x0 = floor(sx - 0.5), y0 = floor(sy - 0.5)
                let fx = sx - 0.5 - x0, fy = sy - 0.5 - y0
                var r = Float(0), g = Float(0), b = Float(0), wsum = Float(0)
                for cy2 in 0...1 {
                    for cx2 in 0...1 {
                        let px = x0 + CGFloat(cx2), py = y0 + CGFloat(cy2)
                        guard px >= 0, px < fw, py >= 0, py < fh else { continue }
                        let wgt = Float((cx2 == 0 ? 1 - fx : fx) * (cy2 == 0 ? 1 - fy : fy))
                        let i = (Int(py) * width + Int(px)) * 4
                        r += Float(buf[i]) * wgt
                        g += Float(buf[i + 1]) * wgt
                        b += Float(buf[i + 2]) * wgt
                        wsum += wgt
                    }
                }
                let o = (y * side + x) * 4
                if wsum > 0 {   // 完全越界 = 黑边(0)
                    out[o] = UInt8(r / wsum)
                    out[o + 1] = UInt8(g / wsum)
                    out[o + 2] = UInt8(b / wsum)
                }
                out[o + 3] = 255
            }
        }
        var src = out
        guard let ctx = CGContext(data: &src, width: side, height: side,
                                  bitsPerComponent: 8, bytesPerRow: side * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        return ctx.makeImage()
    }
}
