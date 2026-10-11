import Accelerate

/// 照片查重:全量载入 clip 向量后贪心聚类——余弦 ≥ threshold 归入同组。
/// 每轮用一次 vDSP_mmul 给锚点照片对全量向量打分(1.4 万张×768 维约 10ms),
/// 整体秒级完成;返回组内为 keys 下标的分组列表。
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
            var c = [Float](repeating: 0, count: n)
            flat.withUnsafeBufferPointer { fb in
                c.withUnsafeMutableBufferPointer { cb in
                    // C(1×N) = A(1×dim) × B(dim×N),B 的第 n 列 = 第 n 张照片向量
                    vDSP_mmul(fb.baseAddress! + a * dim, 1,
                              fb.baseAddress!, vDSP_Length(dim),
                              cb.baseAddress!, 1,
                              1, vDSP_Length(n), vDSP_Length(dim))
                }
            }
            var group: [Int] = []
            for j in 0..<n where alive[j] {
                if j == a || c[j] >= threshold { group.append(j) }
            }
            if group.count > 1 { groups.append(group) }
            for j in group { alive[j] = false }
            done += group.count
            progress(done, n)
        }
        return groups
    }
}
