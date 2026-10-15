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
        // 每簇最高分认领的行号(嵌入一致性豁免的对照对象)
        var topIndexByCluster: [Int: Int] = [:]
        for (index, match) in matches.enumerated() where recognizedClusters.contains(match.cluster) {
            if let cur = topIndexByCluster[match.cluster] {
                if matches[cur].score < match.score { topIndexByCluster[match.cluster] = index }
            } else {
                topIndexByCluster[match.cluster] = index
            }
        }

        var vetoed = Set<Int>()
        for (index, match) in matches.enumerated() where recognizedClusters.contains(match.cluster) {
            guard let topIndex = topIndexByCluster[match.cluster], topIndex != index else { continue }
            let top = matches[topIndex].score
            guard match.score < top, top - match.score > 0.15 || match.score < 0.45 else { continue }
            // 分差/低分只是嫌疑不是定罪:与最高分认领嵌入一致(≥0.48,同人
            // 下限 0.484/跨人上限 0.16 的 CI 标定带)= 同人双脸(镜像/海报/
            // 水面倒影,第二脸因更糊/角度更偏天然压分)→ 豁免。173 实测大量
            // 「见2脸·均判本人」的单人照被 V 凭空造出「未识别同行脸」推进
            // 「与其他人」。同行者的脸是跨人嵌入,到不了 0.48 线,照 veto。
            if let sim = cosineSimilarity(match.embedding, matches[topIndex].embedding),
               sim >= pairSimilarityFloor { continue }
            vetoed.insert(index)
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

    /// 同伴级躯干计数:宽度 ≥ 最宽躯干 18% 且面积 ≥ 其 15%。
    /// 168-170 实测 30% 线漏掉真同行者——『躯干3个/主角级1』= 第二人躯干
    /// 被检出但卡在 30% 线下(自拍/合影同行者常比主角瘦小或站后半步)。
    /// 18% 的依据:走到躯干层的同行者都是与主角同画面尺度的(脸若被复检
    /// 看到早走脸路径了),躯干比例通常 ≥20%;远处路人/背景人身体小
    /// (<15%),仍在线外。诊断带逐躯干宽度百分比,线按实测数据再校。
    /// 面积门(第十三批):175 实测「很远很小」的远景路人宽度能过 18% 线
    /// 但整体很小(高也矮,30%宽×35%高=面积10%);171 的瘦小同行者是
    /// 「窄但高相近」(22%宽→面积 ~20%)——面积把两者分开,已知同行
    /// 案例(22%/27%/35%宽)全部保住。诊断带逐躯干 宽%/面% 可校准。
    static func significantBodyCount(_ bodies: [CGRect], ratio: CGFloat = 0.18,
                                     areaRatio: CGFloat = 0.15) -> Int {
        guard let hero = bodies.max(by: { $0.width < $1.width }), hero.width > 0 else { return 0 }
        let heroArea = hero.width * hero.height
        return bodies.filter { b in
            guard b.width >= hero.width * ratio else { return false }
            guard heroArea > 0 else { return true }
            return b.width * b.height >= heroArea * areaRatio
        }.count
    }

    /// 躯干框去重(第十一批):上半身+全身双模式并跑时,同一个人常被两种
    /// 模式各框一次——上半身框嵌套在全身框上部,高度只有全身框约一半,
    /// IoU 常 0.35-0.5 卡在 0.5 线下漏去重,单人照被数成 2 个躯干
    /// (173 实测『躯干2个[100%,81%]/同伴级2→判有同行』单人照整批误进
    /// 「与其他人」)。追加嵌套判据:交叠面积 ≥ 较小框面积 60% = 同一人
    /// 的两个框,保留先到者。并不同人的身体框(并肩/前后站)互相包含率
    /// 低,不受影响;成人框完全包住身前小孩的极端构图会被并掉(取舍:
    /// 该构图脸检通常能数出两人,不走躯干层)。
    static func mergedBodyBoxes(_ boxes: [CGRect], containment: CGFloat = 0.6) -> [CGRect] {
        var merged: [CGRect] = []
        for b in boxes {
            let duplicate = merged.contains { m -> Bool in
                let inter = m.intersection(b)
                guard !inter.isNull, inter.width > 0, inter.height > 0 else { return false }
                let interArea = inter.width * inter.height
                let areaM = m.width * m.height, areaB = b.width * b.height
                let union = areaM + areaB - interArea
                guard union > 0 else { return false }
                if interArea / union > 0.5 { return true }   // 近重合(Vision 双框原判据)
                let smaller = min(areaM, areaB)
                guard smaller > 0 else { return false }
                return interArea / smaller >= containment    // 嵌套:同人上半身框⊂全身框
            }
            if !duplicate { merged.append(b) }
        }
        return merged
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

    /// 聚合转正(J,第十批):单张证据不够线的 U 脸,若在 ≥6 次观测里稳定
    /// best 指向同一簇 → 跨照片一致性即身份铁证(嵌入噪声方向随机,真
    /// 同伴的糊脸方向稳定)。地板 0.18 排除纯噪声,领先 0.02 排除二义脸
    /// (172 实测 c1@0.20>c2@0.19 领先 0.01 不该赌)。救 Y 的鸡生蛋盲区:
    /// 中频同伴的合影从未有一张被认出 → 历史同框计数恒 0 → Y 永不触发
    /// (172 实测 c9@0.27>c6@0.20U,分数边距全过 Y 门却卡在同框 0)。
    /// 第十三批 minPair 8→6:175 实测杨如合影仍有多张 U 未转正,一致性
    /// 要求不变(稳定指向同一簇),观测次数放宽到 6。
    static func aggregatedRecovered(pairCount: Int, best: Float, second: Float,
                                    minPair: Int = 6, minBest: Float = 0.18,
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

    /// 簇内双峰拆分(第十二批):两个长相相近的人(用户实测:两个从未同框的
    /// 小孩)被质心连接并成一簇——混合质心落在两人中间,各自的脸对它都有
    /// 0.5-0.65,verify/coherence 全放行(均值类守卫对「两个相近的人」
    /// 天然免疫;小孩脸跨人相似度天然比成人高,link 空带的成人标定不适用)。
    /// 拆法:余弦 2-均值,种子=组内最不相似的一对(>320 成员时按步长采样
    /// 选种,确定性保持),迭代分配→重算质心到稳定。返回两半成员下标与
    /// 两半质心的相似度——同人按姿态/年代分的两半,质心都逼近本人平均脸,
    /// 相似度 ≥0.65+;两个相近的人各为一半,质心相似度 ~0.5。调用方以
    /// splitFloor(0.58) 决断。<4 个向量或一半吞光(无第二峰)返回 nil。
    static func bimodalHalves(_ vectors: [[Float]]) -> (a: [Int], b: [Int], centroidSim: Float)? {
        let n = vectors.count
        guard n >= 4, !vectors.isEmpty,
              vectors.allSatisfy({ $0.count == vectors[0].count }) else { return nil }
        let dim = vectors[0].count
        guard dim > 0 else { return nil }

        func unitCos(_ x: [Float], _ y: [Float]) -> Float {
            var dot: Float = 0, nx: Float = 0, ny: Float = 0
            for i in 0..<dim {
                dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i]
            }
            let denom = (nx * ny).squareRoot()
            return denom > 1e-8 ? dot / denom : 0
        }

        // 种子:最不相似的一对(大簇按步长采样选种,确定性:下标小者优先)
        let stride = max(1, n / 320)
        let sample = stride == 1 ? Array(0..<n) : Array(strideFrom0(stride: stride, n: n))
        var worst: Float = 2, seedA = 0, seedB = min(1, n - 1)
        for si in sample.indices {
            for sj in (si + 1)..<sample.count {
                let c = unitCos(vectors[sample[si]], vectors[sample[sj]])
                if c < worst { worst = c; seedA = sample[si]; seedB = sample[sj] }
            }
        }

        var centroidA = vectors[seedA]   // 均值质心,每轮归一化后比较
        var centroidB = vectors[seedB]
        var assign = [Int](repeating: -1, count: n)
        for _ in 0..<10 {
            var sumA = [Float](repeating: 0, count: dim)
            var sumB = [Float](repeating: 0, count: dim)
            var newAssign = [Int](repeating: 0, count: n)
            for i in 0..<n {
                newAssign[i] = unitCos(vectors[i], centroidA) >= unitCos(vectors[i], centroidB) ? 0 : 1
                if newAssign[i] == 0 {
                    for d in 0..<dim { sumA[d] += vectors[i][d] }
                } else {
                    for d in 0..<dim { sumB[d] += vectors[i][d] }
                }
            }
            var normA: Float = 0, normB: Float = 0
            for d in 0..<dim { normA += sumA[d] * sumA[d]; normB += sumB[d] * sumB[d] }
            normA = max(normA.squareRoot(), 1e-8); normB = max(normB.squareRoot(), 1e-8)
            centroidA = sumA.map { $0 / normA }
            centroidB = sumB.map { $0 / normB }
            if newAssign == assign { assign = newAssign; break }
            assign = newAssign
        }
        var a: [Int] = [], b: [Int] = []
        for i in 0..<n {
            if assign[i] == 0 { a.append(i) } else { b.append(i) }
        }
        guard a.count >= 2, b.count >= 2 else { return nil }   // 一半吞光=无第二峰
        return (a, b, unitCos(centroidA, centroidB))
    }

    private static func strideFrom0(stride: Int, n: Int) -> [Int] {
        var out: [Int] = []
        var i = 0
        while i < n { out.append(i); i += stride }
        return out
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
