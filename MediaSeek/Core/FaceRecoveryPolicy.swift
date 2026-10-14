import Foundation

enum FaceRecoveryPolicy {
    struct Match {
        let cluster: Int
        let score: Float
        let strong: Bool
        let embedding: [Float]
    }

    static func vetoedIndices(
        in matches: [Match],
        recognizedClusters: Set<Int>,
        pairSimilarityFloor: Float = 0.48
    ) -> Set<Int> {
        var topByCluster: [Int: Float] = [:]
        for match in matches where recognizedClusters.contains(match.cluster) {
            topByCluster[match.cluster] = max(topByCluster[match.cluster] ?? -2, match.score)
        }

        var vetoed = Set<Int>()
        for (index, match) in matches.enumerated() where recognizedClusters.contains(match.cluster) {
            let top = topByCluster[match.cluster] ?? match.score
            if match.score < top && (top - match.score > 0.15 || match.score < 0.45) {
                vetoed.insert(index)
            }
        }

        for cluster in recognizedClusters.sorted() {
            let candidates = matches.indices.filter {
                matches[$0].cluster == cluster && matches[$0].strong && !vetoed.contains($0)
            }
            let ordered = candidates.sorted { lhs, rhs in
                let left = matches[lhs].score
                let right = matches[rhs].score
                return left == right ? lhs < rhs : left > right
            }
            var accepted: [Int] = []
            for candidate in ordered {
                let consistent = accepted.allSatisfy { prior in
                    guard let similarity = cosineSimilarity(
                        matches[candidate].embedding, matches[prior].embedding
                    ) else { return true }
                    return similarity >= pairSimilarityFloor
                }
                if consistent {
                    accepted.append(candidate)
                } else {
                    vetoed.insert(candidate)
                }
            }
        }

        return vetoed
    }

    /// 躯干兜底裁决:照片只认出一张主角脸、但画面里有 ≥2 个主角级躯干
    /// (宽度 ≥ 最大躯干 30%,与脸的 30% 规则同构)→ 同行者的脸大概率被
    /// 检测层漏检(侧脸/遮挡/动感模糊),不能按纯单人处理。
    /// 远处路人/背景人身体小不过 30% 线,不触发;多主角脸照片不走此兜底
    /// (由脸匹配路径裁决)。
    static func companionSuspectedByBody(detectedFaces: Int, significantBodies: Int) -> Bool {
        detectedFaces == 1 && significantBodies >= 2
    }

    private static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float? {
        guard !lhs.isEmpty, lhs.count == rhs.count else { return nil }
        var dot: Float = 0
        var lhsNorm: Float = 0
        var rhsNorm: Float = 0
        for i in lhs.indices {
            dot += lhs[i] * rhs[i]
            lhsNorm += lhs[i] * lhs[i]
            rhsNorm += rhs[i] * rhs[i]
        }
        let denominator = (lhsNorm * rhsNorm).squareRoot()
        guard denominator.isFinite, denominator > 1e-8 else { return nil }
        return dot / denominator
    }
}
