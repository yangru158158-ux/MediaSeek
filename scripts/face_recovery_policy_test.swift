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

        let lowScoreFace = FaceRecoveryPolicy.Match(
            cluster: owner,
            score: 0.44,
            strong: true,
            embedding: [0.99, sqrt(Float(1) - Float(0.99 * 0.99)), 0]
        )
        expect(
            vetoes([similarFaceA, lowScoreFace]) == [1],
            "原有低分/大分差 veto 必须继续生效"
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

        print("PASS: 同图双脸补认策略的 20 项确定性回归断言全部通过")
    }

    private static func vetoes(_ matches: [FaceRecoveryPolicy.Match]) -> Set<Int> {
        FaceRecoveryPolicy.vetoedIndices(in: matches, recognizedClusters: [0])
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError("FAIL: \(message)") }
    }
}
