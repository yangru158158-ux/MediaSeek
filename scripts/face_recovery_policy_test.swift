import Foundation

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

        print("PASS: 同图双脸补认策略的 5 项确定性回归断言全部通过")
    }

    private static func vetoes(_ matches: [FaceRecoveryPolicy.Match]) -> Set<Int> {
        FaceRecoveryPolicy.vetoedIndices(in: matches, recognizedClusters: [0])
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError("FAIL: \(message)") }
    }
}
