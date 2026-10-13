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

    /// 组间合并(单连接层次聚类的一轮收敛):贪心聚类「一锚定终身」会把同一人
    /// 拆成多组——锚若是侧脸/戴口罩照,正脸照够不到阈值就另起一组,且组一旦
    /// 成形永不复合(140 用户实测同一人分好几组)。本函数按「**任意成员对**
    /// 相似度 ≥ linkThreshold」把组连通起来:同一人的侧脸组与正脸组只要有
    /// 一对照片够像就并成一组(单连接,人脸簇天然链式——同人不同角度靠中间
    /// 角度的照片串起来)。合并后分数相对新组锚(组内最大组的锚)重算。
    ///
    /// linkThreshold 应略高于入组阈值:入组是「与锚像」,连接是「两组各有
    /// 一张互相像」,后者更容易偶然命中,过松会把不同人连成一串。
    static func mergeGroups(_ groups: [[(index: Int, score: Float)]],
                            flat: [Float], dim: Int, linkThreshold: Float,
                            progress: (Int, Int) -> Void = { _, _ in }) -> [[(index: Int, score: Float)]] {
        let g = groups.count
        guard g > 1, dim > 0 else { return groups }
        // 并查集:组号连通分量
        var parent = Array(0..<g)
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { parent[r] = parent[parent[r]]; r = parent[r] }
            return r
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
        }
        // 每组取代表样本(锚 + 组内分数最低的若干张:最低分样本是该组的「边缘
        // 角度」,正是与其他组连接的桥梁);全对全比较成本按代表数平方,
        // 代表数封顶保证大库可用
        let maxReps = 6
        let reps: [[Int]] = groups.map { grp in
            let sorted = grp.sorted { $0.score < $1.score }
            var picked = [grp[0].index]                 // 锚
            for m in sorted.prefix(maxReps - 1) where m.index != grp[0].index {
                picked.append(m.index)
            }
            return picked
        }
        flat.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for i in 0..<g {
                progress(i, g)
                for j in (i + 1)..<g where find(i) != find(j) {
                    var linked = false
                    for ri in reps[i] {
                        for rj in reps[j] {
                            var dot: Float = 0
                            vDSP_dotpr(base + ri * dim, 1, base + rj * dim, 1, &dot, vDSP_Length(dim))
                            if dot >= linkThreshold { linked = true; break }
                        }
                        if linked { break }
                    }
                    if linked { union(i, j) }
                }
            }
        }        // 按连通分量汇总成员,去重后以「成员最多的原组的锚」为新锚重算分数
        var buckets: [Int: [Int]] = [:]          // 根 → 原组号
        for i in 0..<g { buckets[find(i), default: []].append(i) }
        var out: [[(index: Int, score: Float)]] = []
        flat.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            for (_, members) in buckets {
                let anchorGroup = members.max { groups[$0].count < groups[$1].count } ?? members[0]
                let anchor = groups[anchorGroup][0].index
                var seen = Set<Int>()
                var merged: [(index: Int, score: Float)] = []
                for m in members {
                    for e in groups[m] where seen.insert(e.index).inserted {
                        if e.index == anchor { continue }
                        var dot: Float = 0
                        vDSP_dotpr(base + anchor * dim, 1, base + e.index * dim, 1,
                                   &dot, vDSP_Length(dim))
                        merged.append((e.index, dot))
                    }
                }
                merged.sort { $0.score > $1.score }
                out.append([(anchor, 1)] + merged)
            }
        }
        return out.sorted { $0.count > $1.count }
    }
}
