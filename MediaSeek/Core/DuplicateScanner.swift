import Accelerate

/// 向量贪心聚类(照片查重/人脸归组共用):全量载入归一化向量,
/// 每轮取首个存活向量为锚,与全部存活向量做点积(即余弦),≥ threshold 归入同组。
/// 返回组内为 keys 下标的分组列表。
enum DuplicateScanner {
    static func scan(keys: [String], flat: [Float], dim: Int, threshold: Float,
                     progress: (Int, Int) -> Void) -> [[Int]] {
        let n = keys.count
        guard n > 1, dim > 0, flat.count == n * dim else { return [] }
        var alive = [Bool](repeating: true, count: n)
        var groups: [[Int]] = []
        var done = 0
        while true {
            guard let a = alive.firstIndex(of: true) else { break }
            var group: [Int] = []
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
                    if j == a || dot >= threshold { group.append(j) }
                }
            }
            if group.count > 1 { groups.append(group) }
            for j in group { alive[j] = false }
            done += group.count
            progress(done, n)
        }
        return groups
    }
}
