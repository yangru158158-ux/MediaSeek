import CoreGraphics

/// 照片主色调分析:8×8 下采样平均色 → HSV → 中文色名。
/// 纯数学计算,无模型参与,结果确定性 100%。
enum PhotoColor {
    static let buckets = ["红", "橙", "黄", "绿", "青", "蓝", "紫", "粉", "黑", "灰", "白"]

    static func bucket(of image: CGImage) -> String? {
        let w = 8, h = 8
        var pix = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pix, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .low
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var r = 0.0, g = 0.0, b = 0.0
        for i in stride(from: 0, to: pix.count, by: 4) {
            r += Double(pix[i]); g += Double(pix[i + 1]); b += Double(pix[i + 2])
        }
        let n = Double(w * h)
        return classify(r: r / n / 255, g: g / n / 255, b: b / n / 255)
    }

    static func classify(r: Double, g: Double, b: Double) -> String {
        let mx = max(r, g, b), mn = min(r, g, b)
        let v = mx
        let s = v == 0 ? 0 : (mx - mn) / mx
        if v < 0.15 { return "黑" }
        if s < 0.13 { return v > 0.78 ? "白" : "灰" }
        let d = mx - mn
        var h: Double
        if mx == r { h = (g - b) / d }
        else if mx == g { h = (b - r) / d + 2 }
        else { h = (r - g) / d + 4 }
        h = h * 60
        if h < 0 { h += 360 }
        switch h {
        case ..<14: return "红"
        case ..<42: return "橙"
        case ..<70: return "黄"
        case ..<165: return "绿"
        case ..<200: return "青"
        case ..<255: return "蓝"
        case ..<290: return "紫"
        case ..<335: return "粉"
        default: return "红"
        }
    }
}
