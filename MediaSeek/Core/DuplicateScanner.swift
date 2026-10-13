import Accelerate

/// 向量贪心聚类(照片查重/人脸归组共用):全量载入归一化向量,
/// 每轮取首个存活向量为锚,与全部存活向量做点积(即余弦),≥ threshold 归入同组。
/// 返回组内为 (keys 下标, 对锚相似度) 的分组列表(锚自身恒为 1.0)——
/// 查重 UI 用分数区分「完全相同 0.99+」与「只是相似 0.92 档」。
enum DuplicateScanner {
    static func scan(keys: [String], flat: [Float], dim: Int, threshold: Float,
                     progress: (Int, Int) -> Void) -> [[(index: Int, score: Float)]] {
        let n = keys.count
        guard n > 1, dim > 0, flat.count == n * dim else { return [] }
        var alive = [Bool](repeating: true, count: n)
        var groups: [[(index: Int, score: Float)]] = []
        var done = 0
        while true {
            guard let a = alive.firstIndex(of: true) else { break }
            var group: [(index: Int, score: Float)] = []
            flat.withUnsafeBufferPointer { fb in
                let base = fb.baseAddress!
                let anchor = base + a * dim
                for j in 0..<n where alive[j] {
                    // 逐行 vDSP_dotpr(锚点向量·第 j 行),不用整行 vDSP_mmul:
                    // 132 实测人脸矩阵(n=4687×dim=512)mmul 段错误且 far 恰为 B 基址
                    // (合法堆地址整页未映射),巨型矩阵单次调用对形状敏感;
                    // 逐行点积 FLOPs 相同、性能不变,无矩阵形状/步长敏感性
                    var dot: Float = 0
                    vDSP_dotpr(anchor, 1, base + j * dim, 1, &dot, vDSP_Length(dim))
                    if j == a { group.append((j, 1)) }
                    else if dot >= threshold { group.append((j, dot)) }
                }
            }
            if group.count > 1 { groups.append(group) }
            for (j, _) in group { alive[j] = false }
            done += group.count
            progress(done, n)
        }
        return groups
    }

    /// 组间合并:贪心聚类「一锚定终身」会把同一人拆成多组(锚若是侧脸/油画,
    /// 正脸够不到阈值就另起一组,且组一旦成形永不复合)。
    ///
    /// **组质心连接 + 垃圾剔除**(v3,校准模拟验证:15 子组 120 张的大人物全并
    /// 一组、跨人零串组;141 单连接版会把 A B 经「模糊脸桥」串通成 79% 纯度
    /// 巨组,142 平均连接版又保守到把大人物拆散——用户 142 实测同一人又分
    /// 好几组,「照片特别多的人一组近 2000 张本来就可能是对的」):
    /// ① 两组相似度 = **质心(全组向量均值)的余弦**——质心即「平均脸」,
    ///    同人的各子组质心都逼近本人均值、必然相近;个别模糊/误检脸在几百张
    ///    里被平均掉,拦不住正确合并(142 的教训:代表样本挑最低分几张,
    ///    平均被最难看的样本拖低,大组正确合并全被卡死)。
    /// ② 每轮按质心相似度**降序**贪心合并(最像的先并),逐轮收敛——不设
    ///    「互为最佳」限制,大人物的十几个子组就是靠连续合并聚拢的。
    /// ③ **离群照片吸收(143 用户实测同一人散在 1/2/3 张的小组里)**——
    ///    贪心聚类里够不到任何锚的照片一直是"单例",从没资格参与合并;
    ///    v4 让单例并入质心最像的组(≥absorbThreshold)。真实数据同人最低
    ///    0.48/跨人最高 0.16,中间 0.2~0.45 是空带——连接/吸收阈值放在
    ///    空带下段,跨人不可能串组,同人的跨年代子组与离群照全能并拢。
    /// ④ **垃圾剔除**——合并完成后,与组锚相似 < expelFloor 的成员移出组
    ///    (成为单例):模糊/误检脸混进大组拉低「最弱」显示(141 的 -6%),
    ///    该剔的是它们,而不是拦整个合并。
    ///
    /// - Parameters:
    ///   - linkThreshold: 两组质心相似度达到该值才合并
    ///   - expelFloor: 成员对组锚的相似度下限,低于则剔除
    ///   - coherenceFloor: 簇内内聚底线(0=关)。合并完成后每簇计算质心
    ///     (归一化平均向量),成员对质心的**平均**相似度低于它 → 整簇解散
    ///     (成员回落单例)。148 用户实测「第三组人物众多不知为何合并一组」:
    ///     截图小脸/影视剧照/模糊脸在嵌入空间互相靠近(打印脸效应),质心
    ///     连接把它们链式串成「假人物」巨组(154 张里婴儿/明星/界面截屏全有)。
    ///     真人物的成员围绕平均脸紧密分布(同人中位 ~0.7),假簇平均相似度
    ///     显著偏低——解散比逐个剔除更对,因为错在「簇本身不是一个人」
    static func mergeGroups(_ groups: [[(index: Int, score: Float)]],
                            flat: [Float], dim: Int, linkThreshold: Float, expelFloor: Float,
                            singletons: [Int] = [], absorbThreshold: Float? = nil,
                            maxRounds: Int = 25, coherenceFloor: Float = 0,
                            progress: (Int, Int) -> Void = { _, _ in }) -> [[(index: Int, score: Float)]] {
        guard groups.count > 1, dim > 0 else { return groups }

        // 每组维护:成员下标、向量逐维和(质心=归一化后的和)、锚(成员最多原组的锚)
        var out: [[(index: Int, score: Float)]] = []
        var memberIdx: [[Int]] = groups.map { $0.map(\.index) }
        var anchors: [Int] = groups.map { $0[0].index }
        var sums: [[Float]] = memberIdx.map { idx in
            var s = [Float](repeating: 0, count: dim)
            for i in idx {
                for d in 0..<dim { s[d] += flat[i * dim + d] }
            }
            return s
        }

        flat.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            func dot(_ a: Int, _ b: Int) -> Float {
                var d: Float = 0
                vDSP_dotpr(base + a * dim, 1, base + b * dim, 1, &d, vDSP_Length(dim))
                return d
            }
            func centroidDot(_ i: Int, _ j: Int) -> Float {
                var d: Float = 0
                vDSP_dotpr(sums[i], 1, sums[j], 1, &d, vDSP_Length(dim))
                return d   // 未归一化,仅用于同轮内排序与阈值比较前先归一化
            }

            for round in 0..<maxRounds {
                progress(round, maxRounds)
                let n = memberIdx.count
                if n < 2 { break }
                // 归一化质心两两余弦(同轮内公平可比)
                var norms = [Float](repeating: 0, count: n)
                for i in 0..<n {
                    var ss: Float = 0
                    vDSP_dotpr(sums[i], 1, sums[i], 1, &ss, vDSP_Length(dim))
                    norms[i] = max(ss.squareRoot(), 1e-8)
                }
                var pairs: [(i: Int, j: Int, sim: Float)] = []
                for i in 0..<n {
                    for j in (i + 1)..<n {
                        let sim = centroidDot(i, j) / (norms[i] * norms[j])
                        if sim >= linkThreshold { pairs.append((i, j, sim)) }
                    }
                }
                if pairs.isEmpty { break }
                pairs.sort { $0.sim > $1.sim }

                var used = Set<Int>()
                var nextIdx: [[Int]] = []
                var nextAnchors: [Int] = []
                var nextSums: [[Float]] = []
                for p in pairs {
                    guard !used.contains(p.i), !used.contains(p.j) else { continue }
                    used.insert(p.i); used.insert(p.j)
                    nextIdx.append(memberIdx[p.i] + memberIdx[p.j])
                    // 锚沿用成员较多一侧的(大组锚更代表「平均脸」)
                    nextAnchors.append(memberIdx[p.i].count >= memberIdx[p.j].count ? anchors[p.i] : anchors[p.j])
                    var s = [Float](repeating: 0, count: dim)
                    vDSP_vadd(sums[p.i], 1, sums[p.j], 1, &s, 1, vDSP_Length(dim))
                    nextSums.append(s)
                }
                guard !nextIdx.isEmpty else { break }
                for i in 0..<n where !used.contains(i) {
                    nextIdx.append(memberIdx[i])
                    nextAnchors.append(anchors[i])
                    nextSums.append(sums[i])
                }
                memberIdx = nextIdx
                anchors = nextAnchors
                sums = nextSums
            }

            // 离群照片吸收:单例并入质心最像的组。
            // 单例与单例之间不合并——模糊脸互相抱团成"混合桥"的通道就是它
            if !singletons.isEmpty {
                let absorb = absorbThreshold ?? (linkThreshold - 0.03)
                var norms = sums.map { s -> Float in
                    var ss: Float = 0
                    vDSP_dotpr(s, 1, s, 1, &ss, vDSP_Length(dim))
                    return max(ss.squareRoot(), 1e-8)
                }
                for s in singletons {
                    var bestG = -1
                    var bestSim: Float = -2
                    for g in 0..<memberIdx.count {
                        var d: Float = 0
                        vDSP_dotpr(sums[g], 1, base + s * dim, 1, &d, vDSP_Length(dim))
                        let sim = d / norms[g]
                        if sim > bestSim { bestSim = sim; bestG = g }
                    }
                    if bestG >= 0, bestSim >= absorb {
                        memberIdx[bestG].append(s)
                        for d in 0..<dim { sums[bestG][d] += flat[s * dim + d] }
                        var ss: Float = 0
                        vDSP_dotpr(sums[bestG], 1, sums[bestG], 1, &ss, vDSP_Length(dim))
                        norms[bestG] = max(ss.squareRoot(), 1e-8)
                    }
                }
            }

            // 垃圾剔除 + 按锚重算分数
            out = memberIdx.enumerated().compactMap { gi, idx in
                let anchor = anchors[gi]
                var scored: [(index: Int, score: Float)] = []
                for i in idx where i != anchor {
                    let d = dot(anchor, i)
                    if d >= expelFloor { scored.append((i, d)) }
                }
                guard !scored.isEmpty else { return nil }   // 剔完只剩锚=单例,不算组
                scored.sort { $0.score > $1.score }
                return [(index: anchor, score: Float(1))] + scored
            }

            // 簇内内聚守卫:假人物簇整簇解散(见 coherenceFloor 文档)。
            // 每簇质心 = 成员向量和归一化;成员对质心的平均相似度过低即解散
            if coherenceFloor > 0 {
                out = out.filter { g in
                    var c = [Float](repeating: 0, count: dim)
                    for m in g { for d in 0..<dim { c[d] += flat[m.index * dim + d] } }
                    var norm: Float = 0
                    vDSP_dotpr(c, 1, c, 1, &norm, vDSP_Length(dim))
                    norm = max(norm.squareRoot(), 1e-8)
                    var total: Float = 0
                    for m in g {
                        var d: Float = 0
                        vDSP_dotpr(c, 1, base + m.index * dim, 1, &d, vDSP_Length(dim))
                        total += d / norm
                    }
                    return total / Float(g.count) >= coherenceFloor
                }
            }
        }
        return out.sorted { $0.count > $1.count }
    }
}
