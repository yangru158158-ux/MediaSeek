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
    /// **三重防链式塌缩**(141 用户实测单连接版把 1047 组并成 119 组、
    /// 首组 1900 张、组内最弱 -6% = 组里混进了完全不是一个人的脸):
    /// ① **平均连接**而非单连接——两组相似度取「跨组成员对的平均」,不是
    ///    「任意一对够像」;单连接下 A像B、B像C 就把 A C 串起来,一条链能把
    ///    几百人连成一坨(链式塌缩,人脸库里尤其致命)。
    /// ② **互为最佳**——每轮只合并「彼此都是对方最相似组」的一对,逐轮收敛;
    ///    防止一个大组像磁铁一样把所有组吸走。
    /// ③ **组内底线守卫**——合并后若跨组最低成员对低于 floor 就撤销本次合并,
    ///    硬性保证组里不会出现互相完全不像的两张脸(-6% 这类不可能再出现)。
    ///
    /// - Parameters:
    ///   - linkThreshold: 两组平均相似度达到该值才考虑合并
    ///   - floor: 跨组最低成员对的下限,低于则拒绝合并
    static func mergeGroups(_ groups: [[(index: Int, score: Float)]],
                            flat: [Float], dim: Int, linkThreshold: Float, floor: Float,
                            maxRounds: Int = 12,
                            progress: (Int, Int) -> Void = { _, _ in }) -> [[(index: Int, score: Float)]] {
        guard groups.count > 1, dim > 0 else { return groups }

        // 代表样本:锚 + 组内最低分几张(边缘角度正是与其他组的连接点)。
        // 平均/最低都在代表样本上算,成本按代表数平方封顶
        let maxReps = 5
        func representatives(of group: [(index: Int, score: Float)]) -> [Int] {
            let anchor = group[0].index
            let weakest = group.sorted { $0.score < $1.score }
                .prefix(maxReps - 1)
                .map(\.index)
                .filter { $0 != anchor }
            return [anchor] + weakest
        }

        var current: [[Int]] = groups.map(representatives)
        var members: [[(index: Int, score: Float)]] = groups

        flat.withUnsafeBufferPointer { fb in
            guard let base = fb.baseAddress else { return }
            func dot(_ a: Int, _ b: Int) -> Float {
                var d: Float = 0
                vDSP_dotpr(base + a * dim, 1, base + b * dim, 1, &d, vDSP_Length(dim))
                return d
            }

            for round in 0..<maxRounds {
                progress(round, maxRounds)
                let n = current.count
                if n < 2 { break }

                var bestPeer = [Int](repeating: -1, count: n)
                var bestAvg = [Float](repeating: -2, count: n)
                var lowPair = [[Int: Float]](repeating: [:], count: n)
                for i in 0..<n {
                    for j in (i + 1)..<n {
                        var sum: Float = 0, low: Float = 1
                        var count = 0
                        for a in current[i] {
                            for b in current[j] {
                                let d = dot(a, b)
                                sum += d
                                low = min(low, d)
                                count += 1
                            }
                        }
                        let avg = count > 0 ? sum / Float(count) : -1
                        lowPair[i][j] = low
                        if avg > bestAvg[i] { bestAvg[i] = avg; bestPeer[i] = j }
                        if avg > bestAvg[j] { bestAvg[j] = avg; bestPeer[j] = i }
                    }
                }

                var used = Set<Int>()
                var nextCurrent: [[Int]] = []
                var nextMembers: [[(index: Int, score: Float)]] = []
                var changed = false
                for i in 0..<n where !used.contains(i) {
                    let j = bestPeer[i]
                    guard j >= 0, !used.contains(j), bestPeer[j] == i,
                          bestAvg[i] >= linkThreshold,
                          (lowPair[min(i, j)][max(i, j)] ?? -1) >= floor else { continue }
                    used.insert(i); used.insert(j)
                    changed = true
                    // 新组锚 = 成员多的那组的锚;全员分数相对新锚重算
                    let bigger = members[i].count >= members[j].count ? i : j
                    let anchor = members[bigger][0].index
                    var seen: Set<Int> = [anchor]
                    var merged: [(index: Int, score: Float)] = []
                    for e in members[i] + members[j] where seen.insert(e.index).inserted {
                        merged.append((e.index, dot(anchor, e.index)))
                    }
                    merged.sort { $0.score > $1.score }
                    let combined = [(index: anchor, score: Float(1))] + merged
                    nextMembers.append(combined)
                    nextCurrent.append(representatives(of: combined))
                }
                for i in 0..<n where !used.contains(i) {
                    nextCurrent.append(current[i])
                    nextMembers.append(members[i])
                }
                current = nextCurrent
                members = nextMembers
                if !changed { break }
            }
        }
        return members.sorted { $0.count > $1.count }
    }
}
