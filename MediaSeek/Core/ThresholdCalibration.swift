import Foundation

/// 门槛校准测量(批150):在用户自己的相册上,量 SigLIP 分数门槛线实际切在哪里。
///
/// 背景:`ModelManager.siglipGateScale`(0.87)把三条线缩成 一档 0.261 / 二档 0.157 /
/// 三档 0.070。CI 的分布测量只有十来个探针词和探针图,量得出模型之间的相对关系,
/// 量不出这三条线在真实相册上切得对不对——3 (3) 装机日志里 12 个查询的一档命中
/// 全是 0。要回答「系数对不对」,得拿真实相册的数据。
///
/// 做法:固定词表逐词对全部照片打分(查询串与主管线同一个函数),按「照片的标签牌
/// 是否命中该词」分三组——真(牌命中)/ 他(有牌但不命中)/ 无牌——各组记六个
/// 分数带的张数与最高分,写进 diag.log,由开发侧离线汇总。
/// **只读**:不改向量、不改记账、不改任何搜索行为。
enum ThresholdCalibration {

    /// 固定词表:都是标签词典里收了的常见概念(有牌才能当「真命中」的参照),
    /// 覆盖 动物/自然/器物/食物/人物/场合/文档 七类;末尾几个是用户点过名的词
    /// (「身份证」的牌词未必对得上,它的「真」组可能为空,看「他」组即可)
    static let words: [String] = [
        "猫", "狗", "鸟", "鱼", "马", "兔子",
        "海滩", "山", "湖", "雪", "森林", "树", "花", "日落", "夜晚", "天空", "云", "彩虹", "河流", "瀑布",
        "汽车", "自行车", "摩托车", "飞机", "火车", "手机", "电脑", "电视", "键盘", "书", "眼镜", "手表",
        "食物", "蛋糕", "咖啡", "水果", "面条", "火锅",
        "婴儿", "儿童", "女人", "男人", "人群",
        "婚礼", "生日", "游泳", "钓鱼", "演唱会", "运动", "足球", "游泳池", "酒店", "餐厅", "商场", "街道", "桥", "建筑", "寺庙",
        "文档", "屏幕截图", "发票", "护照", "无人机", "热气球", "身份证", "信用卡",
    ]

    /// 一组的六个分数带(自上而下):≥一档线 / 未缩放二档线~一档线 / 二档线~未缩放二档线 /
    /// 尾线低置信线~二档线 / 三档线~尾线低置信线 / <三档线。带界由调用方按现行系数算好传入
    private struct Bands {
        var n = 0
        var c = [Int](repeating: 0, count: 6)
        var maxScore: Float = -1

        mutating func add(_ s: Float, _ edges: [Float]) {
            n += 1
            maxScore = max(maxScore, s)
            var i = 0
            while i < edges.count, s < edges[i] { i += 1 }
            c[i] += 1
        }

        var text: String { c.map { String($0) }.joined(separator: "/") }
    }

    /// 逐词全库打分并写 diag.log。返回给界面看的摘要。
    /// - Parameter gate: 现行门槛系数(ModelManager.siglipGateScale)
    static func run(store: VectorStore, siglip: SigLIPEmbedder, gate: Float) throws -> String {
        let started = Date()
        // 带界(降序):一档 0.30×gate / 未缩放的二档 0.18(系数取 1.0 时的线)/
        // 二档 0.18×gate / 尾线低置信基准 0.13×gate / 三档 0.08×gate
        let edges: [Float] = [0.30 * gate, 0.18, 0.18 * gate, 0.13 * gate, 0.08 * gate]
        var plates: [String: String] = [:]
        for r in try store.allPhotoLabels() where plates[r.refKey] == nil { plates[r.refKey] = r.title }
        var photos = 0
        var lines: [String] = []
        for word in words {
            let text = SearchEngine.siglipQuery(word, nativeZh: true)
            let q = try siglip.embedQuery(text)
            let hits = try store.search(space: siglip.space, kinds: [.photo], query: q,
                                        limit: 1_000_000, minScore: -1)
            photos = hits.count
            let tokens = QueryUnderstanding.labelTokens(word)
            var yes = Bands(), other = Bands(), none = Bands()
            for h in hits {
                if let plate = plates[h.refKey] {
                    if QueryUnderstanding.labelTokensMatchCount(tokens, plate) > 0 {
                        yes.add(h.score, edges)
                    } else {
                        other.add(h.score, edges)
                    }
                } else {
                    none.add(h.score, edges)
                }
            }
            lines.append("校准[\(word)] 串=\(text) 真=\(yes.n):\(yes.text) "
                         + "他=\(other.n):\(other.text) 高=\(fmt(other.maxScore)) "
                         + "无牌=\(none.n):\(none.text) 高=\(fmt(none.maxScore))")
        }
        let edgeText = edges.map { String(format: "%.3f", $0) }.joined(separator: "/")
        IndexingCoordinator.DiagLog.shared.event(
            "校准头 照片=\(photos) 有牌=\(plates.count) 词数=\(words.count) 系数=\(gate) 带=\(edgeText) "
            + "(六带自上而下:≥一档线/未缩放二档线~一档线/二档线~未缩放二档线/尾线低置信线~二档线/"
            + "三档线~尾线低置信线/<三档线;只算照片;真=标签牌命中该词,他=有牌不命中,无牌=没有标签牌)")
        for l in lines { IndexingCoordinator.DiagLog.shared.event(l) }
        let secs = Date().timeIntervalSince(started)
        return "\(words.count) 个词 · 照片 \(photos) 张(有标签牌 \(plates.count) 张) · 用时 \(String(format: "%.1f", secs)) 秒\n"
             + "明细已写入诊断日志:点下方「导出诊断日志」发出"
    }

    private static func fmt(_ s: Float) -> String {
        s < 0 ? "-" : String(format: "%.3f", s)
    }
}
