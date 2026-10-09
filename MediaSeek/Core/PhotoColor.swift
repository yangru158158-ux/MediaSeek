import CoreGraphics

/// 照片主色调分析:8×8 下采样 → 逐像素分档 → 票数最多的色档。
/// 纯数学计算,无模型参与,结果确定性 100%。
/// 不用整图平均:白底截图上的蓝色链接/头像会把平均值带偏成「蓝」。
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
        var votes = [String: Int]()
        for i in stride(from: 0, to: pix.count, by: 4) {
            let name = classify(r: Double(pix[i]) / 255,
                                g: Double(pix[i + 1]) / 255,
                                b: Double(pix[i + 2]) / 255)
            votes[name, default: 0] += 1
        }
        // 固定顺序取票数最高档,平票取靠前者,保证确定性
        var bestName: String?
        var bestCount = -1
        for name in buckets {
            let c = votes[name] ?? 0
            if c > bestCount { bestCount = c; bestName = name }
        }
        return bestName
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
