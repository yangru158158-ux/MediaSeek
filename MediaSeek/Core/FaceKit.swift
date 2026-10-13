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

    /// 已对齐的 112×112 人脸图 → 归一化身份向量
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

/// 人脸检测与 ArcFace 标准对齐
enum FaceKit {
    /// ArcFace 112×112 五点模板的双眼参数(insightface arcface_dst,CG 底左坐标系 y 已翻转)
    private static let eyeMid = CGPoint(x: 55.9132, y: 60.4012)
    private static let eyeDistance: CGFloat = 35.2372

    /// 检测人脸(带关键点)。VNDetectFaceLandmarksRequest 自带检测步,
    /// 结果同时含外接框与五官区域;一张合照最多取 maxFaces 张脸,防群照拖垮索引
    static func detect(cg: CGImage, maxFaces: Int = 8) -> [VNFaceObservation] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            return Array((request.results ?? []).prefix(maxFaces))
        } catch {
            return []
        }
    }

    /// 按双眼连线做相似变换对齐(双眼转水平、瞳距缩放到模板值、眼中点平移到模板位),
    /// 渲染成 112×112;无关键点时退化为外接框放大 40% 的中心方形。
    /// ArcFace 对齐敏感,不按模板对齐会显著掉同人召回。
    static func alignedFace(source: CGImage, observation: VNFaceObservation) -> CGImage? {
        let side = 112
        guard let ctx = CGContext(data: nil, width: side, height: side,
                                  bitsPerComponent: 8, bytesPerRow: side * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        let w = CGFloat(source.width), h = CGFloat(source.height)

        let transform: CGAffineTransform
        if let lm = observation.landmarks,
           let le = centroid(lm.leftEye, imageSize: source),
           let re = centroid(lm.rightEye, imageSize: source) {
            let mid = CGPoint(x: (le.x + re.x) / 2, y: (le.y + re.y) / 2)
            let dx = re.x - le.x, dy = re.y - le.y
            let dist = max(hypot(dx, dy), 1)
            let s = eyeDistance / dist
            // 复合顺序 = 平移眼中点至原点 → 旋平双眼 → 缩放瞳距 → 平移到模板位
            transform = CGAffineTransform(translationX: -mid.x, y: -mid.y)
                .rotated(by: -atan2(dy, dx))
                .scaledBy(x: s, y: s)
                .translatedBy(x: eyeMid.x, y: eyeMid.y)
        } else {
            let bb = observation.boundingBox
            let sidePx = min(max(bb.width * w, bb.height * h) * 1.4, min(w, h))
            let cx = bb.midX * w, cy = bb.midY * h
            let src = CGRect(x: cx - sidePx / 2, y: cy - sidePx / 2, width: sidePx, height: sidePx)
            let s = CGFloat(side) / sidePx
            transform = CGAffineTransform(translationX: -src.minX, y: -src.minY)
                .scaledBy(x: s, y: s)
        }
        // CTM 取逆:把「源坐标→模板坐标」的变换换算成绘制时的画布映射
        ctx.concatenate(transform.inverted())
        ctx.draw(source, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    /// 关键点区域质心(归一化坐标 → 图像像素,底左原点与 CG 一致)
    private static func centroid(_ region: VNFaceLandmarkRegion2D?, imageSize: CGImage) -> CGPoint? {
        guard let pts = region?.normalizedPoints, !pts.isEmpty else { return nil }
        var sumX: CGFloat = 0, sumY: CGFloat = 0
        for p in pts { sumX += p.x; sumY += p.y }
        return CGPoint(x: sumX / CGFloat(pts.count) * CGFloat(imageSize.width),
                       y: sumY / CGFloat(pts.count) * CGFloat(imageSize.height))
    }
}
