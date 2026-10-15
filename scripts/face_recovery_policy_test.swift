import Foundation
import CoreGraphics

@main
struct FaceRecoveryPolicyTest {
    static func main() {
        let owner = 0

        let firstDifferentFace = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.50,
            strong: true,
            embedding: [0.50, 0, sqrt(Float(0.75))]
        )
        let secondDifferentFace = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.49,
            strong: true,
            embedding: [0.49, 0, -sqrt(Float(1) - Float(0.49 * 0.49))]
        )
        expect(
            vetoes([firstDifferentFace, secondDifferentFace]) == [1],
            "不同人以相近分数命中同一簇时应将较弱认领标为未解释"
        )

        let similarFaceA = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.52,
            strong: true,
            embedding: [1, 0, 0]
        )
        let similarFaceB = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.51,
            strong: true,
            embedding: [0.95, sqrt(Float(1) - Float(0.95 * 0.95)), 0]
        )
        expect(
            vetoes([similarFaceA, similarFaceB]).isEmpty,
            "脸间相似的同人双脸不应因共享人物簇而被额外 veto"
        )

        let mirrorFace = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.44,
            strong: true,
            embedding: [0.99, sqrt(Float(1) - Float(0.99 * 0.99)), 0]
        )
        expect(
            vetoes([similarFaceA, mirrorFace]).isEmpty,
            "同人双脸(镜像/海报/倒影,嵌入一致≥0.48)分数悬殊也不 veto——降级会凭空造出未识别脸把单人照推进「与其他人」(第十一批)"
        )
        let strayLowFace = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.44,
            strong: true,
            embedding: [0, 1, 0]
        )
        expect(
            vetoes([similarFaceA, strayLowFace]) == [1],
            "跨人低分认领(嵌入不一致)的绝对分 veto 照常生效(166 同伴保护)"
        )
        let strayFarFace = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.30,
            strong: false,
            embedding: [0, 0.6, 0.8]
        )
        expect(
            vetoes([similarFaceA, strayFarFace]) == [1],
            "跨人大分差认领(嵌入不一致)的分差 veto 照常生效"
        )

        let tiedBest = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.50,
            strong: true,
            embedding: [1, 0]
        )
        let belowFloor = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.50,
            strong: true,
            embedding: [0.479, sqrt(Float(1) - Float(0.479 * 0.479))]
        )
        let aboveFloor = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.50,
            strong: true,
            embedding: [0.481, sqrt(Float(1) - Float(0.481 * 0.481))]
        )
        expect(
            vetoes([tiedBest, belowFloor]) == [1],
            "低于 0.48 的脸间相似度应稳定 veto 同分的后一个认领"
        )
        expect(
            vetoes([tiedBest, aboveFloor]).isEmpty,
            "高于 0.48 的脸间相似度应保留同分的双脸认领"
        )

        expect(
            FaceRecoveryPolicy.companionSuspectedByBody(detectedFaces: 1, significantBodies: 2),
            "一主角脸+≥2 主角级躯干应判疑似同行(脸被检测层漏检)"
        )
        expect(
            !FaceRecoveryPolicy.companionSuspectedByBody(detectedFaces: 1, significantBodies: 1),
            "一主角脸+一躯干=纯单人,人体兜底不触发"
        )
        expect(
            !FaceRecoveryPolicy.companionSuspectedByBody(detectedFaces: 2, significantBodies: 2),
            "多主角脸照片不走人体兜底,由脸匹配路径裁决"
        )
        expect(
            FaceRecoveryPolicy.companionSuspectedByBody(detectedFaces: 0, significantBodies: 2),
            "复检零脸+≥2 主角级躯干(两人背影/远景)应判疑似同行"
        )
        expect(
            !FaceRecoveryPolicy.companionSuspectedByBody(detectedFaces: 0, significantBodies: 1),
            "零脸+一躯干不触发人体兜底"
        )

        // 同伴级躯干计数(170 实测『躯干3个/主角级1』= 30% 线漏掉真同行者):
        // 18% 线——同行者常比主角瘦小/站后半步,远处路人 <15%
        let heroBody = CGRect(x: 0.1, y: 0.0, width: 0.40, height: 0.90)
        expect(FaceRecoveryPolicy.significantBodyCount([]) == 0, "无躯干=0")
        expect(FaceRecoveryPolicy.significantBodyCount([heroBody]) == 1, "单躯干=1")
        expect(
            FaceRecoveryPolicy.significantBodyCount([
                heroBody, CGRect(x: 0.52, y: 0.0, width: 0.32, height: 0.85)
            ]) == 2,
            "≥30% 宽的第二躯干计为同伴"
        )
        expect(
            FaceRecoveryPolicy.significantBodyCount([
                heroBody, CGRect(x: 0.10, y: 0.45, width: 0.06, height: 0.18)
            ]) == 1,
            "<18% 宽的远处小躯干(路人)不计入"
        )
        expect(
            FaceRecoveryPolicy.significantBodyCount([
                heroBody, CGRect(x: 0.50, y: 0.0, width: 0.20, height: 0.80)
            ]) == 2,
            "18-30% 宽的贴身同行躯干计入(170 实测卡线漏径)"
        )
        expect(
            FaceRecoveryPolicy.significantBodyCount([
                heroBody, CGRect(x: 0.50, y: 0.0, width: 0.20, height: 0.80),
                CGRect(x: 0.02, y: 0.50, width: 0.04, height: 0.15)
            ]) == 2,
            "同行者计入且背景小躯干不计入(混合场景)"
        )
        expect(
            FaceRecoveryPolicy.significantBodyCount([
                heroBody, CGRect(x: 0.60, y: 0.55, width: 0.12, height: 0.30)
            ]) == 1,
            "宽度过线(30%)但整体很小(面积 10%)的远景路人不计入(第十三批面积门)"
        )
        expect(
            FaceRecoveryPolicy.significantBodyCount([
                heroBody, CGRect(x: 0.52, y: 0.0, width: 0.088, height: 0.85)
            ]) == 2,
            "171 瘦小同行者(宽 22% 高相近,面积 21%)仍计入——面积门不误伤"
        )

        // —— 躯干去重(第十一批):上半身+全身双模式同框各框一次,上半身框
        // 嵌套在全身框上部(IoU 0.35-0.5 卡 0.5 线下),单人照被数成 2 躯干 ——
        // 173 实测『躯干2个[100%,81%]/同伴级2→判有同行』单人照误进「与其他人」
        let fullBody = CGRect(x: 0.10, y: 0.0, width: 0.40, height: 0.90)
        let upperBody = CGRect(x: 0.12, y: 0.50, width: 0.32, height: 0.40)
        expect(
            FaceRecoveryPolicy.mergedBodyBoxes([fullBody, upperBody]).count == 1,
            "同人上半身框嵌套在全身框内(IoU≈0.36<0.5)应并成 1 个躯干"
        )
        expect(
            FaceRecoveryPolicy.significantBodyCount(
                FaceRecoveryPolicy.mergedBodyBoxes([fullBody, upperBody])) == 1,
            "去重后单人照同伴级躯干=1,不再误判有同行(宽度比 80%≈实测 81% 样本)"
        )
        expect(
            FaceRecoveryPolicy.mergedBodyBoxes([
                heroBody, CGRect(x: 0.52, y: 0.0, width: 0.32, height: 0.85)
            ]).count == 2,
            "并肩两人的躯干框不并"
        )
        expect(
            FaceRecoveryPolicy.mergedBodyBoxes([
                heroBody, CGRect(x: 0.42, y: 0.0, width: 0.30, height: 0.85)
            ]).count == 2,
            "部分交叠(前后站)的两人躯干框不并(交叠/较小框 ≈27% < 60%)"
        )
        expect(
            FaceRecoveryPolicy.mergedBodyBoxes([heroBody, heroBody]).count == 1,
            "近重合双框(IoU>0.5)照旧去重"
        )

        // SigLIP 计数兜底(贴脸/重叠构图连躯干检测器都只框出一人时的最后证据):
        // two ≥ 0.10 且领先 one ≥ 0.025 才判——真余弦尺度命中带才 0.10-0.14
        expect(
            FaceRecoveryPolicy.companionSuspectedBySigLIP(twoScore: 0.13, oneScore: 0.08),
            "两人解读明确胜出(two 达标且领先)应判疑似同行"
        )
        expect(
            !FaceRecoveryPolicy.companionSuspectedBySigLIP(twoScore: 0.11, oneScore: 0.10),
            "领先幅度不足 margin 的不判(单人照保护)"
        )
        expect(
            !FaceRecoveryPolicy.companionSuspectedBySigLIP(twoScore: 0.09, oneScore: 0.03),
            "two 低于 minTwo(语义不读图/怪图)不判"
        )
        expect(
            !FaceRecoveryPolicy.companionSuspectedBySigLIP(twoScore: 0.12, oneScore: 0.14),
            "一人解读反超时绝不判两人"
        )

        // —— 同框先验(Y):质心 0.22-0.27 的糊脸 + 高频同框者身份 → 认作同行者 ——
        expect(
            FaceRecoveryPolicy.priorRecoveredByCoOccurrence(best: 0.27, second: 0.18, coCount: 159),
            "171 实测样本 c1@0.27>c3@0.18 且同框 159 张应认(Y)"
        )
        expect(
            !FaceRecoveryPolicy.priorRecoveredByCoOccurrence(best: 0.22, second: 0.18, coCount: 350),
            "领先仅 0.04 太薄不认(留给 K 近邻通道裁决)"
        )
        expect(
            !FaceRecoveryPolicy.priorRecoveredByCoOccurrence(best: 0.27, second: 0.18, coCount: 19),
            "同框不足 20 张的低频同框者不认"
        )
        expect(
            !FaceRecoveryPolicy.priorRecoveredByCoOccurrence(best: 0.21, second: 0.05, coCount: 200),
            "低于 0.22 地板的噪声脸不认"
        )
        expect(
            FaceRecoveryPolicy.priorRecoveredByCoOccurrence(best: 0.30, second: -2.0, coCount: 40),
            "无第二名竞争(哨兵 -2)时领先巨大应认"
        )
        // —— 近邻复核(K):平均脸质心压分的糊脸,簇内 top-5 近邻过线 → 认 ——
        expect(
            FaceRecoveryPolicy.priorRecoveredByKNN(bestKNN: 0.34, secondKNN: 0.20),
            "质心 0.2x 但簇内近邻 0.34 且领先 0.14 应认(K)"
        )
        expect(
            !FaceRecoveryPolicy.priorRecoveredByKNN(bestKNN: 0.28, secondKNN: 0.10),
            "近邻均值不到 0.30 地板不认"
        )
        expect(
            !FaceRecoveryPolicy.priorRecoveredByKNN(bestKNN: 0.33, secondKNN: 0.30),
            "近邻领先不足 0.06(第二簇竞争)不认"
        )
        expect(
            FaceRecoveryPolicy.priorRecoveredByKNN(bestKNN: 0.31, secondKNN: nil),
            "无第二簇竞争(nil)时达标即认"
        )

        // —— 聚合转正(J):跨照片稳定指向同一簇 → 集体转正 ——
        expect(
            FaceRecoveryPolicy.aggregatedRecovered(pairCount: 8, best: 0.27, second: 0.20),
            "172 实测样本 c9@0.27>c6@0.20 配对 ≥8 次应转正(J)"
        )
        expect(
            FaceRecoveryPolicy.aggregatedRecovered(pairCount: 6, best: 0.27, second: 0.20),
            "172 实测样本 c9@0.27>c6@0.20 配对 ≥6 次应转正(J,第十三批 8→6)"
        )
        expect(
            !FaceRecoveryPolicy.aggregatedRecovered(pairCount: 5, best: 0.27, second: 0.20),
            "配对 5 次不足 6 不转正(跨照片证据不够)"
        )
        expect(
            !FaceRecoveryPolicy.aggregatedRecovered(pairCount: 30, best: 0.17, second: 0.05),
            "低于 0.18 地板的纯噪声脸不转正"
        )
        expect(
            !FaceRecoveryPolicy.aggregatedRecovered(pairCount: 30, best: 0.20, second: 0.19),
            "领先 0.01 的二义脸不赌(172 实测 c1@0.20>c2@0.19)"
        )
        // —— parseFaceSegments:裁决 detail 的结构化解析(二轮统计载体) ——
        let segs = FaceRecoveryPolicy.parseFaceSegments(
            "≥30px rect检出 2/关键点 1/并集 2,≥30% 主角 2脸; c1@0.13>c2@0.10U | c0@0.57>c7@0.14S")
        expect(
            segs.count == 2 && segs[0].ci == 1 && abs(segs[0].best - 0.13) < 0.001
                && segs[0].secondCi == 2 && segs[0].tag == "U",
            "首段(与诊断前缀以『; 』相连)解析出 U 脸字段"
        )
        expect(
            segs.count == 2 && segs[1].ci == 0 && segs[1].secondCi == 7 && segs[1].tag == "S",
            "第二段(无前缀)解析出 S 脸"
        )
        expect(
            FaceRecoveryPolicy.parseFaceSegments(
                "未检出≥30px候选脸;躯干2个[100%,40%]/同伴级2→判有同行").isEmpty,
            "零脸 detail 解析不出脸段"
        )
        expect(
            FaceRecoveryPolicy.parseFaceSegments(
                "≥30px rect检出 2,≥30%主角2脸; c1@0.20>c2@0.19U | 嵌入失败U").count == 1,
            "『嵌入失败U』段被跳过,只解析出 1 个脸段"
        )

        // —— 双峰拆分(第十二批):两个相近的人并成一簇(混合质心落在两人
        // 中间,verify/coherence 免疫),2-均值找出两半、两半质心相似度决断 ——
        // 174 实测:28 张「未命名」组里两个从未同框的小孩
        let kidA = [
            [Float(1), 0.10, 0], [Float(1), -0.10, 0], [Float(0.90), 0.15, 0.05],
            [Float(1), 0, -0.06], [Float(0.95), -0.12, 0.03]
        ]
        let kidB = [
            [Float(0.50), 0.87, 0.02], [Float(0.52), 0.85, -0.01], [Float(0.48), 0.88, 0],
            [Float(0.55), 0.83, 0.03], [Float(0.50), 0.86, -0.02]
        ]
        let twoKids = FaceRecoveryPolicy.bimodalHalves(kidA + kidB)
        expect(
            twoKids != nil && Set(twoKids!.a) == Set(0...4) && Set(twoKids!.b) == Set(5...9),
            "两个方向相近的人(质心 cos≈0.53)应精确拆成两半"
        )
        expect(
            twoKids != nil && twoKids!.centroidSim < 0.58,
            "两半质心相似度低于拆分线 0.58(调用方据此拆簇)"
        )
        let poseSpread = [
            [Float(1), 0.30, 0], [Float(1), -0.30, 0], [Float(0.90), 0, 0.40],
            [Float(1), 0, -0.35], [Float(0.95), 0.20, 0.10], [Float(1), -0.10, -0.20]
        ]
        let single = FaceRecoveryPolicy.bimodalHalves(poseSpread)
        expect(
            single != nil && single!.centroidSim >= 0.58,
            "单人按姿态分散的簇:无论 2-均值怎么分,两半质心都逼近本人平均脸,不拆"
        )
        expect(
            FaceRecoveryPolicy.bimodalHalves(Array(poseSpread.prefix(3))) == nil,
            "不足 4 个向量不判双峰"
        )
        let onePeak = Array(repeating: [Float(1), Float(0), Float(0)], count: 6)
        expect(
            FaceRecoveryPolicy.bimodalHalves(onePeak) == nil,
            "同方向紧致簇(无第二峰,一半吞光)返回 nil"
        )
        let rerun = FaceRecoveryPolicy.bimodalHalves(kidA + kidB)
        expect(
            rerun != nil && twoKids != nil && rerun!.a == twoKids!.a && rerun!.b == twoKids!.b,
            "种子与分配确定性:同输入两次结果一致(缓存簇 ID 稳定的前提)"
        )

        print("PASS: 同图双脸补认策略的 53 项确定性回归断言全部通过")
    }

    private static func vetoes(_ matches: [FaceRecoveryPolicy.Match]) -> Set<Int> {
        FaceRecoveryPolicy.vetoedIndices(in: matches, recognizedClusters: [0])
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError("FAIL: \(message)") }
    }
}
