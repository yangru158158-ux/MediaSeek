// 查询理解确定性回归(第十五批起):core() 的指令/否定/载体后缀清洗。
// 起因:用户实测搜「地图」被裸「图」后缀砍成「地」——OCR 按单字「地」
// 灌入 1,891 张、语义通道拿不到词典条目(地图→map)全空。护栏=词典整词
// 不动 + 裸「图」仅在词干 ≥2 字或词干本身是词典单字时才剥。
// 纯 Foundation 逻辑,与 App 共用同一源文件,CI 上直接编译运行。
import Foundation

@main
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
        // 注:dictionary 与 LabelZh 反查表都命中「地图」,english 按位置拼接成
        // "map map"——既有行为,断言用包含式(语义通道拿到的仍是有效英文)
        expect(QueryUnderstanding.english(for: "地图")?.contains("map") == true,
               "「地图」词典命中 map(第十五批语义通道恢复的前提)")
        expect(QueryUnderstanding.english(for: "地") == nil,
               "单字「地」不是词典条目,裸「地」查不到英文(事故时的语义空根因)")
        expect(QueryUnderstanding.english(for: "沙滩")?.contains("beach") == true,
               "口语变体「沙滩」入册(第十九批:此前只有「海滩」,搜沙滩语义空转)")

        // —— 性别词补全(第四十四批:woman 裸搜仅 2 张事故) ——
        expect(QueryUnderstanding.english(for: "女性")?.contains("woman") == true,
               "「女性」入册(性别查询走词典→模板,不再依赖标签碰运气)")
        expect(QueryUnderstanding.english(for: "男性")?.contains("man") == true,
               "「男性」入册")
        let manTokens = QueryUnderstanding.expandedTokens(["man"])
        expect(manTokens.contains("boy") == true && manTokens.contains("guy") == true,
               "man 一跳扩展含 boy/guy(与 woman 含 girl 对称,标签覆盖不再缺半边)")
        let womanTokens = QueryUnderstanding.expandedTokens(["woman"])
        expect(womanTokens.contains("lady") == true,
               "woman 一跳扩展含 lady")
        expect(womanTokens.contains("adult") == false,
               "泛化词不做同义入口(woman 的扩展不含 adult,防灌水回潮)")

        // —— 泛化词不自剔(第四十五批:搜 machine 标签 1306 张命中归零事故) ——
        expect(QueryUnderstanding.labelTokens("machine").contains("machine"),
               "查询本身就是泛化词 machine 时保留——剔它等于删掉查询本体")
        expect(QueryUnderstanding.labelTokens("person").contains("person"),
               "查询本身就是 person 时保留(用户点名要这个标签)")
        // 注:「机器设备」这类混合直译(machine 泛化 + device 实义)旧规则会
        // 剔 machine 留实义词——但照片标签牌按同一本词典挂,剔词典直译词
        // 等于对着挂 machine 牌的 1306 张关掉标签通道。第四十八批重立规则:
        // 泛化剔除只管用户手敲的英文词,词典直译词一律免疫
        expect(QueryUnderstanding.labelTokens("机器").contains("machine"),
               "词典显示词「机器」(2字)直搜 machine 标签正常(库内 1306 张)")
        expect(QueryUnderstanding.labelTokens("屏幕截图").contains("screenshot"),
               "词典显示词「屏幕截图」直译 screenshot 免疫泛化剔除(库内 5197 张)")
        expect(QueryUnderstanding.labelTokens("机器设备").contains("machine"),
               "混合直译的泛化词也保留(机器设备→machine+device,1306 张机器标签可达)")
        expect(QueryUnderstanding.labelTokens("网页").contains("screenshot"),
               "复合直译带出的泛化词保留(网页→website screenshot)")
        expect(QueryUnderstanding.labelTokens("朋友").contains("people"),
               "词典直译 people 保留(朋友→friend people,人物标签可达)")
        expect(QueryUnderstanding.labelTokens("pdf").contains("document"),
               "词典直译 document 保留(pdf→document,文档标签可达)")
        expect(QueryUnderstanding.labelTokens("machine beach").contains("beach"),
               "machine beach 保留实义词 beach")
        expect(QueryUnderstanding.labelTokens("machine beach").contains("machine") == false,
               "有别的实义词时 machine 照常防灌水剔除(三十九批设计不回退)")
        expect(QueryUnderstanding.labelTokens("people beach photo").contains("people") == false,
               "多词长查询的泛化词剔除照常生效(防全库灌水)")
        expect(manTokens.contains("gentleman") == true,
               "man 一跳扩展含 gentleman(与 lady 对称)")

        // —— 第五十批:性别词对比过滤词表(SigLIP 绝对分不分男女,相对差分) ——
        expect(QueryUnderstanding.genderOpposite(of: "woman") == "man",
               "woman 反义 man")
        expect(QueryUnderstanding.genderOpposite(of: "man") == "woman",
               "man 反义 woman")
        expect(QueryUnderstanding.genderOpposite(of: "WOMAN") == "man",
               "大写归一小写后命中")
        expect(QueryUnderstanding.genderOpposite(of: "女人") == "man",
               "女人 反义 man")
        expect(QueryUnderstanding.genderOpposite(of: "男人") == "woman",
               "男人 反义 woman")
        expect(QueryUnderstanding.genderOpposite(of: "美女") == "man",
               "美女 反义 man")
        expect(QueryUnderstanding.genderOpposite(of: "beach") == nil,
               "非性别词无反义(不过滤)")
        expect(QueryUnderstanding.genderOpposite(of: " 女人 ") == "man",
               "空白归一后命中")

        // —— 第五十四批:词典直译同词去重(双词稀释性别信号=「女人」4217 洪水源头) ——
        expect(QueryUnderstanding.english(for: "女人") == "woman",
               "女人 直译去重为单词(此前 dictionary+englishFromZh 各拼一次成 woman woman)")
        expect(QueryUnderstanding.english(for: "屏幕截图") == "screenshot",
               "屏幕截图=全词条目最长匹配(54批曾拼重复串,60批嵌套碎片不再重复捐 screen)")

        // —— 第五十六批:短词短路只护种子,扩展不得注入泛化词 ——
        // 207 实测:搜「女人」(2字)短路分支放行同义扩展注入的 person/people,
        // 标签通道对全库人像标签命中 4,279 张灌进语义结果(且逐张取图极慢)
        let nv = QueryUnderstanding.labelTokens("女人")
        expect(nv.contains("person") == false && nv.contains("people") == false,
               "女人(2字)扩展注入的 person/people 被剔——标签通道不再灌全库人像")
        expect(nv.contains("woman") && nv.contains("girl") && nv.contains("lady"),
               "女人 种子与正常扩展保留(woman/girl/lady)")
        expect(QueryUnderstanding.labelTokens("朋友").contains("people"),
               "朋友 直译种子 people 保留(48批词典直译免疫设计不回退)")
        expect(QueryUnderstanding.labelTokens("机器").contains("machine"),
               "机器(2字)直译 machine 保留(短路护种子设计不回退)")

        // —— 第六十批:覆盖率规则(「华研机器人」事故:机器/机器人碎片把股票名
        // 偷换成 robot machine,「机器」直译的 machine 借直译免疫闯过泛化剔除,
        // 标签通道灌进上千张截图,搜出 1,183 个结果里只有 2 个是真的) ——
        expect(QueryUnderstanding.english(for: "华研机器人") == nil,
               "华研机器人(专名)碎片盖不满全词,一个不捐——整词交 SigLIP 多语言")
        expect(QueryUnderstanding.english(for: "机器人") == "robot",
               "机器人 整词=词典条目全覆盖;嵌套碎片「机器」不再重复捐 machine")
        let jsb = QueryUnderstanding.english(for: "机器设备") ?? ""
        expect(jsb.contains("machine") && jsb.contains("equipment"),
               "机器设备=机器+设备 全覆盖并列复合词照常直译(「设备」词条 60 批补;"
               + "同区间标签译词 device 一并捐,与蔬菜 vegetables/vegetable 同理)")
        expect(QueryUnderstanding.english(for: "妈妈的生日") != nil,
               "连接字「的」不计未覆盖——描述性词组照常捐译文(妈妈+的+生日)")
        let hynTokens = QueryUnderstanding.labelTokens("华研机器人")
        expect(hynTokens.contains("machine") == false && hynTokens.contains("robot") == false,
               "华研机器人 标签通道不再吃 machine/robot 碎片——灌水根除")

        // —— 第五十五批:性别对比串=文献标准形式(CVPR 2025 口径,带冠词小写) ——
        if let pair = QueryUnderstanding.genderContrast(of: "女人") {
            expect(pair.fwd == "a photo of a woman" && pair.opp == "a photo of a man",
                   "女人 → 文献式对比串 woman/man 对")
        } else { fatalError("FAIL: 女人应有性别对比串") }
        if let pair = QueryUnderstanding.genderContrast(of: "men") {
            expect(pair.fwd == "a photo of a man" && pair.opp == "a photo of a woman",
                   "men 复数归一为单数 man,反向 woman")
        } else { fatalError("FAIL: men 应有性别对比串") }
        expect(QueryUnderstanding.genderContrast(of: "beach") == nil,
               "非性别词无对比串")

        // 第五十五批补:对比串全词表穷尽(genderOpposites/genderEnglish 双表
        // 键漂移护栏——加词漏配 genderEnglish 时 CI 立红)
        let singulars: Set<String> = ["woman", "man", "girl", "boy", "lady", "gentleman"]
        for (k, _) in QueryUnderstanding.genderOpposites {
            guard let p = QueryUnderstanding.genderContrast(of: k) else {
                fatalError("FAIL: 性别词「\(k)」无对比串(genderEnglish 漏配?)")
            }
            let fw = String(p.fwd.dropFirst("a photo of a ".count))
            let ow = String(p.opp.dropFirst("a photo of a ".count))
            if !singulars.contains(fw) || !singulars.contains(ow) || fw == ow {
                fatalError("FAIL: 性别词「\(k)」对比串未归一/同侧:\(p.fwd) / \(p.opp)")
            }
        }

        // —— 全量标签词典回归(第四十七批):词典里每一个词都必须能工作 ——
        // 不是逐词救火:全部标签词(≈1566)做两个方向的穷尽断言——
        //   ①英文标签直搜不被任何逻辑剔除(含 12 个泛化词,守卫后本尊保留)
        //   ②词典中文反查能命中对应英文标签(标签精确通道)
        // 以后改 labelTokens/词典/泛化名单漏掉任何一词,CI 立即红
        var zhToEns: [String: [String]] = [:]
        for (en, cn) in LabelZh.zh { zhToEns[cn, default: []].append(en) }
        var passEn = 0
        for en in LabelZh.zh.keys {
            if QueryUnderstanding.labelTokens(en).contains(en) { passEn += 1 }
            else { fatalError("FAIL: 英文标签直搜被剔除:\(en)") }
        }
        expect(passEn == LabelZh.zh.count,
               "全量 \(LabelZh.zh.count) 个英文标签直搜全部保留")
        var passCn = 0
        for (cn, ens) in zhToEns {
            let tokens = QueryUnderstanding.labelTokens(cn)
            if ens.contains(where: { tokens.contains($0) }) { passCn += 1 }
            else { fatalError("FAIL: 词典中文「\(cn)」搜不到对应标签 \(ens)") }
        }
        expect(passCn == zhToEns.count,
               "全量 \(zhToEns.count) 个中文词反查全部命中标签通道")

        // 断言数口径:全文 expect 调用 55 项(另含 1 行函数定义)
        print("PASS: 查询理解 55 项确定性回归断言全部通过(含全量标签词典 \(LabelZh.zh.count) 词穷尽)")
    }
}
