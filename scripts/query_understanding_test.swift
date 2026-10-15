// 查询理解确定性回归(第十五批起):core() 的指令/否定/载体后缀清洗。
// 起因:用户实测搜「地图」被裸「图」后缀砍成「地」——OCR 按单字「地」
// 灌入 1,891 张、语义通道拿不到词典条目(地图→map)全空。护栏=词典整词
// 不动 + 裸「图」仅在词干 ≥2 字或词干本身是词典单字时才剥。
// 纯 Foundation 逻辑,与 App 共用同一源文件,CI 上直接编译运行。
import Foundation

enum QueryUnderstandingTest {
    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else { fatalError("FAIL: \(message)") }
    }

    static func main() {
        // —— 后缀剥离护栏(第十五批) ——
        expect(QueryUnderstanding.core("地图") == "地图",
               "词典整词「地图」不被裸「图」截肢(用户实测事故样本)")
        expect(QueryUnderstanding.core("拼图") == "拼图",
               "二字真词「拼图」词干仅 1 字,不剥")
        expect(QueryUnderstanding.core("企图") == "企图",
               "二字真词「企图」词干仅 1 字,不剥")
        expect(QueryUnderstanding.core("美女图") == "美女",
               "词干 ≥2 字的「美女图」照常剥成「美女」")
        expect(QueryUnderstanding.core("猫图") == "猫",
               "「猫图」词干「猫」是词典单字,照常剥")
        expect(QueryUnderstanding.core("照片") == "照片",
               "整词即后缀(查询就是「照片」)不动,空结果回退原查询")
        expect(QueryUnderstanding.core("地图的照片") == "地图",
               "「地图的照片」剥载体后缀后留下真词「地图」")
        expect(QueryUnderstanding.core("电脑屏幕的截图") == "电脑屏幕",
               "长载体后缀照常剥")

        // —— 指令/否定清洗回归(既有行为不得回退) ——
        expect(QueryUnderstanding.core("帮我找出有猫的图片、没有猫不要显示") == "猫",
               "指令前缀+否定子句截断组合")
        expect(QueryUnderstanding.core("找出电脑屏幕的照片") == "电脑屏幕",
               "指令前缀+载体后缀组合")
        expect(QueryUnderstanding.core("包含2026的截图") == "2026",
               "存在前缀剥「包含」,留数字词供 OCR 通道")

        // —— 词典英译抽查(语义通道依赖) ——
        expect(QueryUnderstanding.english(for: "地图") == "map",
               "「地图」词典命中 map(第十五批语义通道恢复的前提)")
        expect(QueryUnderstanding.english(for: "地") == nil,
               "单字「地」不是词典条目,裸「地」查不到英文(事故时的语义空根因)")

        print("PASS: 查询理解 13 项确定性回归断言全部通过")
    }
}

QueryUnderstandingTest.main()
