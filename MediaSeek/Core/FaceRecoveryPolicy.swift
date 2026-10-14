import Foundation
import CoreGraphics

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

    /// 躯干兜底裁决:照片只认出 ≤1 张主角脸、但画面里有 ≥2 个同伴级躯干
    /// → 同行者的脸大概率被检测层漏检(侧脸/遮挡/动感模糊/远景小脸),
    /// 不能按纯单人处理。
    /// 0 张脸(复检零检出,168 实测新径)与 1 张脸同样适用——两人背影/
    /// 远景合照连一张脸都检不出,躯干仍可作同行证据。
    /// 多主角脸照片不走此兜底(由脸匹配路径裁决)。
    static func companionSuspectedByBody(detectedFaces: Int, significantBodies: Int) -> Bool {
        detectedFaces <= 1 && significantBodies >= 2
    }

    /// 同伴级躯干计数:宽度 ≥ 最宽躯干 18% 的身体框个数。
    /// 168-170 实测 30% 线漏掉真同行者——『躯干3个/主角级1』= 第二人躯干
    /// 被检出但卡在 30% 线下(自拍/合影同行者常比主角瘦小或站后半步)。
    /// 18% 的依据:走到躯干层的同行者都是与主角同画面尺度的(脸若被复检
    /// 看到早走脸路径了),躯干比例通常 ≥20%;远处路人/背景人身体小
    /// (<15%),仍在线外。诊断带逐躯干宽度百分比,线按实测数据再校。
    static func significantBodyCount(_ bodies: [CGRect], ratio: CGFloat = 0.18) -> Int {
        guard let hero = bodies.max(by: { $0.width < $1.width }), hero.width > 0 else { return 0 }
        return bodies.filter { $0.width >= hero.width * ratio }.count
    }

    /// SigLIP 计数兜底:脸检(双通道并集)与躯干(双模式+同伴级)都只数出
    /// 1 时的最后一层证据——整图语义对「两人」vs「一人」的相对偏好
    /// (照片向量已在索引库,零图像推理)。贴脸拥抱/两人被检测器框成一个
    /// 整体时,脸与躯干几何通道全部失效,整图语义是唯一证据。
    /// 判据保守:真余弦尺度上命中带才 0.10-0.14,要求 two ≥ 0.10 且领先
    /// one ≥ 0.025(两人解读明确胜出才判);诊断必带两个分数,误伤时
    /// 一张截图即可校准 margin。
    static func companionSuspectedBySigLIP(twoScore: Float, oneScore: Float,
                                           minTwo: Float = 0.10, margin: Float = 0.025) -> Bool {
        twoScore >= minTwo && (twoScore - oneScore) >= margin
    }

    /// 同框先验(Y,第九批):171 实测 rect 并集已把合影第二脸找回,但糊/侧脸
    /// 对质心只有 0.22-0.27(双判据线 0.32 下)全判 U,890 张堆在「与其他人」。
    /// 未用的强证据:最像的簇与本照片已认出主角历史高频同框(杨如×程玉婷
    /// 数百张)——陌生人/低频路人成不了主角的高频同框者。三重门:
    /// 同框 ≥20 张 + 地板 0.22 + 领先第二名 0.06(跨人误认被同框约束+边距
    /// 双保险)。诊断字母 Y,误伤一张截图即可校准。
    static func priorRecoveredByCoOccurrence(best: Float, second: Float, coCount: Int,
                                             minBest: Float = 0.22, minLead: Float = 0.06,
                                             minCoCount: Int = 20) -> Bool {
        coCount >= minCoCount && best >= minBest && (best - second) >= minLead
    }

    /// 近邻复核(K,第九批):质心是全簇平均脸,把糊/侧脸往簇中心拉、系统性
    /// 压分;改与簇内库内成员逐脸比 top-5 均值——糊脸对「同角度/同糊历史脸」
    /// 的分高于对平均脸(InsightFace 标准匹配即 KNN)。0.30 地板+领先第二名
    /// 簇近邻 0.06;无第二簇竞争(nil)时达标即认。
    static func priorRecoveredByKNN(bestKNN: Float, secondKNN: Float?) -> Bool {
        bestKNN >= 0.30 && bestKNN - (secondKNN ?? -1) >= 0.06
    }

    /// 聚合转正(J,第十批):单张证据不够线的 U 脸,若在 ≥8 次观测里稳定
    /// best 指向同一簇 → 跨照片一致性即身份铁证(嵌入噪声方向随机,真
    /// 同伴的糊脸方向稳定)。地板 0.18 排除纯噪声,领先 0.02 排除二义脸
    /// (172 实测 c1@0.20>c2@0.19 领先 0.01 不该赌)。救 Y 的鸡生蛋盲区:
    /// 中频同伴的合影从未有一张被认出 → 历史同框计数恒 0 → Y 永不触发
    /// (172 实测 c9@0.27>c6@0.20U,分数边距全过 Y 门却卡在同框 0)。
    static func aggregatedRecovered(pairCount: Int, best: Float, second: Float,
                                    minPair: Int = 8, minBest: Float = 0.18,
                                    minLead: Float = 0.02) -> Bool {
        pairCount >= minPair && best >= minBest && (best - second) >= minLead
    }

    /// 从裁决 detail 解析逐脸段(第十批 J 通道二轮统计用):
    /// 「c9@0.27>c6@0.20U」→(ci:9, best:0.27, secondCi:6, second:0.20, tag:"U")。
    /// 与 work() 的 detailParts 生成端共同约定格式;非脸段(诊断前缀/
    /// 嵌入失败U)解析不出自动跳过。detail 即结构化载体,聚合免改缓存表。
    static func parseFaceSegments(_ detail: String)
        -> [(ci: Int, best: Float, secondCi: Int, second: Float, tag: String)] {
        var faces: [(ci: Int, best: Float, secondCi: Int, second: Float, tag: String)] = []
        for raw in detail.components(separatedBy: " | ") {
            // 首段可能与诊断前缀以「; 」相连,取最后一个 "; " 之后
            let s = raw.range(of: "; ", options: .backwards).map { String(raw[$0.upperBound...]) } ?? raw
            guard s.hasPrefix("c"), let at1 = s.firstIndex(of: "@"),
                  let gt = s.firstIndex(of: ">") else { continue }
            let afterGt = s[s.index(after: gt)...]
            guard afterGt.hasPrefix("c"), let at2 = afterGt.firstIndex(of: "@") else { continue }
            let tail = s[afterGt.index(after: at2)...]
            guard tail.count >= 5,   // "0.20"+单字母 tag(%.2f 恒两位小数)
                  let ci = Int(s[s.index(after: s.startIndex)..<at1]),
                  let secondCi = Int(afterGt[afterGt.index(after: afterGt.startIndex)..<at2]),
                  let best = Float(s[s.index(after: at1)..<gt]),
                  let second = Float(tail.dropLast()),
                  ["S", "W", "V", "U", "Y", "K"].contains(String(tail.suffix(1))) else { continue }
            faces.append((ci, best, secondCi, second, String(tail.suffix(1))))
        }
        return faces
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
