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
///
/// 批152(用户:「再量一次」):批150 的装机数据表明绝对分分不开真伪(真命中九成落在
/// 0.07~0.157,噪声能到 0.17~0.226),下一步要评估「相对分」——跟对照词比、跟这张
/// 照片对其它词的平均分比、看这个词排第几。六带计数做不了这些,所以测量时把整张
/// 分数表(全部照片 × 词表与对照词)、每张照片的标签牌、以及「牌是否命中该词」的
/// 标记,写成一份制表符分隔的文件(`dataURL`),由设置页「导出校准数据」分享出去。
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

    /// 批152:词表之外的对照词——「这张照片是什么类型」的通用描述,给「相对分」当参照
    static let extras: [String] = ["人物", "风景", "室内", "户外"]

    /// 原始分数表文件(Documents 下,「文件」App 可见;设置页「导出校准数据」分享的就是它)
    static let dataURL: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("MediaSeek-calibration.tsv")

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

    /// 逐词全库打分,写 diag.log(六带计数)与原始分数表文件。返回给界面看的摘要。
    /// - Parameter gate: 现行门槛系数(ModelManager.siglipGateScale)
    static func run(store: VectorStore, siglip: SigLIPEmbedder, gate: Float) throws -> String {
        let started = Date()
        // 带界(降序):一档 0.30×gate / 未缩放的二档 0.18(系数取 1.0 时的线)/
        // 二档 0.18×gate / 尾线低置信基准 0.13×gate / 三档 0.08×gate
        let edges: [Float] = [0.30 * gate, 0.18, 0.18 * gate, 0.13 * gate, 0.08 * gate]
        var plates: [String: String] = [:]
        for r in try store.allPhotoLabels() where plates[r.refKey] == nil { plates[r.refKey] = r.title }
        // 分数表:每个词一行,列=照片(以第一个词的全库结果定照片顺序)
        let all = words + extras
        var scores: [[Float]] = []
        var queries: [String] = []
        var refKeys: [String] = []
        var indexOf: [String: Int] = [:]
        // 牌命中标记:[词 × 照片数 + 照片序号] = 1 表示这张照片的牌命中该词(只对词表内的词)
        var mask = [UInt8]()
        var photos = 0
        var lines: [String] = []
        for (wi, word) in all.enumerated() {
            let isMain = wi < words.count
            let text = isMain ? SearchEngine.siglipQuery(word, nativeZh: true)
                              : QueryUnderstanding.nativeSiglipQuery(word)
            let q = try siglip.embedQuery(text)
            let hits = try store.search(space: siglip.space, kinds: [.photo], query: q,
                                        limit: 1_000_000, minScore: -1)
            if refKeys.isEmpty {
                photos = hits.count
                refKeys = hits.map(\.refKey)
                for (i, k) in refKeys.enumerated() { indexOf[k] = i }
                mask = [UInt8](repeating: 0, count: words.count * photos)
            }
            var row = [Float](repeating: 0, count: photos)
            var yes = Bands(), other = Bands(), none = Bands()
            let tokens: Set<String> = isMain ? QueryUnderstanding.labelTokens(word) : []
            for h in hits {
                guard let i = indexOf[h.refKey] else { continue }
                row[i] = h.score
                guard isMain else { continue }
                if let plate = plates[h.refKey] {
                    if QueryUnderstanding.labelTokensMatchCount(tokens, plate) > 0 {
                        yes.add(h.score, edges)
                        mask[wi * photos + i] = 1
                    } else {
                        other.add(h.score, edges)
                    }
                } else {
                    none.add(h.score, edges)
                }
            }
            scores.append(row)
            queries.append(text)
            if isMain {
                lines.append("校准[\(word)] 串=\(text) 真=\(yes.n):\(yes.text) "
                             + "他=\(other.n):\(other.text) 高=\(fmt(other.maxScore)) "
                             + "无牌=\(none.n):\(none.text) 高=\(fmt(none.maxScore))")
            }
        }
        let edgeText = edges.map { String(format: "%.3f", $0) }.joined(separator: "/")
        IndexingCoordinator.DiagLog.shared.event(
            "校准头 照片=\(photos) 有牌=\(plates.count) 词数=\(words.count) 系数=\(gate) 带=\(edgeText) "
            + "(六带自上而下:≥一档线/未缩放二档线~一档线/二档线~未缩放二档线/尾线低置信线~二档线/"
            + "三档线~尾线低置信线/<三档线;只算照片;真=标签牌命中该词,他=有牌不命中,无牌=没有标签牌)")
        for l in lines { IndexingCoordinator.DiagLog.shared.event(l) }

        // 原始分数表文件
        var out: [String] = []
        out.append("#智搜校准数据 v2\t照片=\(photos)\t词数=\(words.count)\t额外=\(extras.count)\t系数=\(gate)")
        out.append("#词表\t" + all.joined(separator: "\t"))
        out.append("#串\t" + queries.joined(separator: "\t"))
        out.append("#列\t序号\t标签牌\t牌命中(逐词0/1,按词表顺序)\t分数(先词表后额外,各一个)")
        var maskChars = [Character]()
        maskChars.reserveCapacity(words.count)
        for p in 0..<photos {
            maskChars.removeAll(keepingCapacity: true)
            for w in 0..<words.count { maskChars.append(mask[w * photos + p] == 1 ? "1" : "0") }
            var parts: [String] = [String(p), plates[refKeys[p]] ?? "", String(maskChars)]
            parts.reserveCapacity(3 + all.count)
            for w in 0..<all.count { parts.append(String(format: "%.3f", scores[w][p])) }
            out.append(parts.joined(separator: "\t"))
        }
        let body = out.joined(separator: "\n") + "\n"
        try body.write(to: dataURL, atomically: true, encoding: .utf8)
        let kb = body.utf8.count / 1024
        IndexingCoordinator.DiagLog.shared.event(
            "校准表 已写入 \(dataURL.lastPathComponent) 行=\(photos) 列=\(all.count + 3) 大小=\(kb)KB")

        let secs = Date().timeIntervalSince(started)
        return "\(words.count) 个词 · 照片 \(photos) 张(有标签牌 \(plates.count) 张) · 用时 \(String(format: "%.1f", secs)) 秒\n"
             + "明细已写入诊断日志:点下方「导出诊断日志」发出\n"
             + "完整分数表已存成文件(\(String(format: "%.1f", Double(kb) / 1024)) MB):点下方「导出校准数据」发出"
    }

    private static func fmt(_ s: Float) -> String {
        s < 0 ? "-" : String(format: "%.3f", s)
    }
}
