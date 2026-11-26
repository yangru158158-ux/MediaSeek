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
        expect(jsb.contains("machine") && jsb.contains("equipment") && jsb.contains("device"),
               "机器设备=机器+设备 全覆盖并列复合词照常直译(「设备」词条 60 批补;"
               + "同区间标签译词 device 一并捐,与蔬菜 vegetables/vegetable 同理)")
        expect(QueryUnderstanding.english(for: "妈妈的生日") != nil,
               "连接字「的」不计未覆盖——描述性词组照常捐译文(妈妈+的+生日)")
        let hynTokens = QueryUnderstanding.labelTokens("华研机器人")
        expect(hynTokens.contains("machine") == false && hynTokens.contains("robot") == false,
               "华研机器人 标签通道不再吃 machine/robot 碎片——灌水根除")

        // —— 第六十二批:专名判定(覆盖率规则的通道延伸) ——
        // 211 实测:华研机器人裸搜仍出 96 张「语义」命中——碎片不捐了,但
        // 整词原文兜底把词交给 SigLIP 多语言,行业截图噪声带分数全数过第三档
        // 地板。连续生字符≥2 判专名,整词只走文字通道;单生字符(修饰词)
        // 与零命中(词典没收)照旧原文兜底,不误杀
        expect(QueryUnderstanding.isProperNoun("华研机器人"),
               "华研机器人=华研(2 连续生字符)+机器人 → 专名,视觉通道整词跳过")
        expect(QueryUnderstanding.isProperNoun("机器设备") == false,
               "机器设备 全覆盖=复合词,不是专名(前置过滤器照常捐译)")
        expect(QueryUnderstanding.isProperNoun("妈妈的生日") == false,
               "连接字不算生字符,描述性词组不是专名")
        expect(QueryUnderstanding.isProperNoun("屏幕和截图") == false,
               "「和」是连接字,复合词不是专名")
        expect(QueryUnderstanding.isProperNoun("机器人") == false,
               "整词即词典条目,不是专名")
        expect(QueryUnderstanding.isProperNoun("woman") == false,
               "ASCII 零命中不是专名(裸英文照走模板)")
        expect(QueryUnderstanding.isProperNoun("财报") == false,
               "中文零命中不是专名(词典没收的普通词,整词原文交 SigLIP 多语言)")
        expect(QueryUnderstanding.english(for: "红裙子") == nil,
               "红裙子 盖不满不捐碎片(60 批规则不变)")
        expect(QueryUnderstanding.isProperNoun("红裙子") == false,
               "单生字符「红」=修饰词形态,不判专名(保持原文兜底)")
        // 全词典不变量:任何词条自身必然全覆盖 → 不判专名(键漂移护栏,
        // 以后改覆盖评估/词典让词条自判专名时 CI 立红)
        for (zh, _) in QueryUnderstanding.dictionary {
            if QueryUnderstanding.isProperNoun(zh) {
                fatalError("FAIL: 词典条目「\(zh)」自身被判专名(覆盖评估回归)")
            }
        }
        for (zh, _) in LabelZh.englishFromZh {
            if QueryUnderstanding.isProperNoun(zh) {
                fatalError("FAIL: 标签反查条目「\(zh)」自身被判专名(覆盖评估回归)")
            }
        }

        // —— 第六十四批:人名闸的零命中判据(english==nil 比它宽,验收拦下) ——
        expect(QueryUnderstanding.isDictionaryUnknown("管雯"),
               "管雯=真零命中(人名,词典/标签表无任何碎片)——人名闸目标形态")
        expect(QueryUnderstanding.isDictionaryUnknown("财报"),
               "财报=零命中(词典没收的普通词,无标签时照走语义)")
        expect(QueryUnderstanding.isDictionaryUnknown("woman"),
               "裸英文=零命中(性别词有人名闸豁免,深潜不废)")
        expect(QueryUnderstanding.isDictionaryUnknown("红裙子") == false,
               "红裙子=盖不满的修饰词,不是零命中——批 62 原文兜底承诺不因人名闸破")
        expect(QueryUnderstanding.isDictionaryUnknown("机器人") == false,
               "机器人=全覆盖复合词,不是零命中")
        expect(QueryUnderstanding.isDictionaryUnknown("华研机器人") == false,
               "华研机器人=专名(盖不满),不是零命中(批 62 专名闸已管)")

        // —— 第六十九批:英文→中文证件桥(纯加法) ——
        expect(QueryUnderstanding.chineseCounterpart(for: "id") == "身份证"
            && QueryUnderstanding.chineseCounterpart(for: "ID") == "身份证"
            && QueryUnderstanding.chineseCounterpart(for: "iD") == "身份证",
               "裸词 id(不分大小写)→身份证:外国人缩写词,218 用户点名")
        expect(QueryUnderstanding.chineseCounterpart(for: "identity") == "身份证"
            && QueryUnderstanding.chineseCounterpart(for: "Identity Card") == "身份证"
            && QueryUnderstanding.chineseCounterpart(for: "identity document") == "身份证"
            && QueryUnderstanding.chineseCounterpart(for: "id card") == "身份证",
               "identity 族四个变体→身份证(整短语键,拆词后单词只有 identity/id 命中)")
        expect(QueryUnderstanding.chineseCounterpart(for: "passport") == "护照",
               "passport→护照(文字变体追加 passport/签发国,原 %passport% 匹配不动)")
        expect(QueryUnderstanding.chineseCounterpart(for: "Driver License") == "驾照"
            && QueryUnderstanding.chineseCounterpart(for: "driving license") == "驾照"
            && QueryUnderstanding.chineseCounterpart(for: "driver's license") == "驾照",
               "驾驶照三拼写→驾照")
        expect(QueryUnderstanding.chineseCounterpart(for: "licence") == "驾照"
            && QueryUnderstanding.chineseCounterpart(for: "Licence") == "驾照",
               "licence→驾照(第七十一批翻案:218「语义全对」是 so400m 分布,标准版证件照够不着第二档且底网被文字命中挡住,入桥=模型无关修法)")
        expect(QueryUnderstanding.chineseCounterpart(for: "license") == "驾照"
            && QueryUnderstanding.chineseCounterpart(for: "LICENSE") == "驾照",
               "license(美式)→驾照(第七十四批用户批准:与 licence 同词异拼,对称面原则)")
        expect(QueryUnderstanding.chineseCounterpart(for: "driving") == nil
            && QueryUnderstanding.chineseCounterpart(for: "card") == nil,
               "driving/card 泛词不入表(桥了会把各种证件窄化成驾照)")
        let idv = QueryUnderstanding.bridgedVariants(for: "identity")
        expect(idv == ["身份证"],
               "bridgedVariants(identity)=[身份证](批128收缩:只桥中文对应词,不再追加 ocrAliases)")
        expect(QueryUnderstanding.bridgedVariants(for: "beach") == []
            && QueryUnderstanding.bridgedVariants(for: "id card") == ["身份证"],
               "表外词返回空(原匹配零改动);表内短语同样只桥对应词")
        expect(QueryUnderstanding.bridgedVariants(for: "licence") == ["驾照"],
               "licence 桥接变体=驾照(批128收缩:去准驾车型/机动车驾驶证——字段别名出现在所有提及驾照的文档,284 装机 licence OCR 后半部中文杂音)")

        // —— 第九十四批:原生语义: 模板构造(唯一规则,零转换) ——
        expect(QueryUnderstanding.nativeSiglipQuery("license") == "This is a photo of license.",
               "原生语义:license 不经英文桥,裸词直进模板壳(对照:语义:license 桥成 driver license card)")
        expect(QueryUnderstanding.nativeSiglipQuery("licence") == "This is a photo of licence.",
               "原生语义:licence 与 license 是两根不同向量——拼写即身份,不再合并")
        expect(QueryUnderstanding.nativeSiglipQuery("駕駛證") == "This is a photo of 駕駛證.",
               "原生语义:繁体不归一、中文原文直进(SigLIP2 中文原生能力的首个受控实验入口)")
        expect(QueryUnderstanding.nativeSiglipQuery("a man riding a horse on the beach")
                    == "This is a photo of a man riding a horse on the beach.",
               "原生语义:语句不拆词,整句直进(批93 平均已在批107 退役,语义:多词改逐词交集)")
        expect(QueryUnderstanding.nativeSiglipQuery("2026") == "This is a photo of 2026.",
               "原生语义:时间词也是内容词,不解析不过滤(用户拍板:语句可能带时间)")

        // —— 第九十八批:专名闸「的」豁免(美丽的天空误杀案) ——
        expect(QueryUnderstanding.isProperNoun("美丽的天空") == false,
               "含「的」=描述短语不判专名(249 装机实锤:美丽2生字符+天空在册→旧闸判专名→视觉踢除→裸搜0)")
        expect(QueryUnderstanding.isProperNoun("华研机器人") == true,
               "无「的」的专名闸门本体不变(批62 原始案例)")
        expect(QueryUnderstanding.isProperNoun("红裙子") == false,
               "1 生字符修饰词形态维持不判专名(批62 既有边界)")

        // —— 第七十二批:繁→简归一 + 证件与 licence 同权 ——
        expect(QueryUnderstanding.simplified("駕駛證") == "驾驶证"
            && QueryUnderstanding.simplified("證件") == "证件"
            && QueryUnderstanding.simplified("護照") == "护照"
            && QueryUnderstanding.simplified("美團") == "美团",
               "繁→简字表:证件家族+股票名(駕駛證/證件/護照/美團)")
        expect(QueryUnderstanding.simplified("动物") == "动物"
            && QueryUnderstanding.simplified("美團 存款") == "美团 存款"
            && QueryUnderstanding.simplified(QueryUnderstanding.simplified("駕駛證 護照"))
                   == QueryUnderstanding.simplified("駕駛證 護照"),
               "简体/同形字透传,空格保留,幂等(单遍无链式)")
        expect(QueryUnderstanding.scriptVariants(of: "駕駛證") == ["駕駛證", "驾驶证"]
            && QueryUnderstanding.scriptVariants(of: "动物") == ["动物"],
               "scriptVariants:繁体词=原词+转换词(原词在前保排序),简体词单元素")
        expect((QueryUnderstanding.ocrAliases["证件"] ?? []) == ["身份证", "护照", "驾驶证"],
               "证件入别名表:文字追加+底网豁免与 licence 同权(221 实测搜「证件」文字 98/语义 0)")
        // 第八十三批:document→card(232 装机搜「驾驶证」语义 4,023 杂音,
        // "document" 泛词对文档密集图库打穿 tier-2);断言意图不变——
        // 驾驶证必须有词典对,繁体归一后不退回裸多语言嵌入
        expect(QueryUnderstanding.english(for: "驾驶证") == "driver license card"
            && QueryUnderstanding.english(for: "驾照") == "driver license card",
               "驾驶证词典对补齐+批83收窄:繁体归一到驾驶证后不再退回裸多语言嵌入,驾照同款")
        expect(QueryUnderstanding.english(for: "护照") == "passport",
               "护照词典对补齐:中文查询语义模板从裸多语言升级为 passport(第六十九批)")
        expect((QueryUnderstanding.ocrAliases["驾照"] ?? []) == (QueryUnderstanding.ocrAliases["驾驶证"] ?? []),
               "驾照与驾驶证别名同表:此前只有驾驶证有,输驾照漏别名(兄弟缺口)")

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

        // —— 批134:查询端下划线处理护栏(批130验收余账) ——
        // 批130 加了拆分逻辑但两侧测试都没断言它——误删时 CI 全绿,286 装机
        // 「play_card 标签=0 全灭」的事故会静默回潮;Python 镜像同集断言。
        // 批143(用户定「两个都精确」)护栏改写:词表内的键打进来自身要能命中
        // 自己的牌(批142 起靠整键),但不再拆片段——拆出的片段会顺带命中同名
        // 的其它牌(3 (2) 装机:balloon_hotair 标签=56 对 热气球 标签=7,多的
        // 是 balloon 牌);词表外的下划线串照旧拆
        let pcTokens = QueryUnderstanding.labelTokens("play_card")
        expect(pcTokens.contains("play_card") && !pcTokens.contains("play") && !pcTokens.contains("card"),
               "play_card(词表内的键)整键进词集、不拆片段")
        expect(QueryUnderstanding.labelTokensMatchCount(pcTokens, "play_card") == 1
               && QueryUnderstanding.labelTokensMatchCount(pcTokens, "card") == 0,
               "play_card 命中自己的牌,不命中 card 牌")
        let idcTokens = QueryUnderstanding.labelTokens("id_card")
        expect(idcTokens.contains("id") && idcTokens.contains("card"),
               "词表外的下划线串照旧拆片段(id_card→id+card)")
        // 批143:全量复合键——①打进来都命中自己的牌;②片段不由拆分注入
        // (同义表引入的除外,如 blue_sky→sky);③命中词函数与计数函数恒一致
        var splitLeak: [String] = [], ownMiss: [String] = [], termMismatch: [String] = []
        for (en, cn) in LabelZh.zh {
            let typedTokens = QueryUnderstanding.labelTokens(en)
            if en.contains("_") {
                if QueryUnderstanding.labelTokensMatchCount(typedTokens, en) == 0 { ownMiss.append(en) }
                let viaSynonyms = QueryUnderstanding.expandedTokens([en])
                for frag in en.components(separatedBy: "_") where !frag.isEmpty {
                    if typedTokens.contains(frag) && !viaSynonyms.contains(frag) {
                        splitLeak.append("\(en)→\(frag)")
                    }
                }
            }
            let zhTokens = QueryUnderstanding.labelTokens(cn)
            for (toks, plate) in [(typedTokens, en), (zhTokens, en)] {
                if QueryUnderstanding.labelTokensMatchedTerms(toks, plate).count
                    != QueryUnderstanding.labelTokensMatchCount(toks, plate) {
                    termMismatch.append(plate)
                }
            }
        }
        expect(ownMiss.isEmpty, "复合键打进来都命中自己的牌(整键): \(ownMiss.prefix(3))")
        expect(splitLeak.isEmpty, "复合键的片段不由拆分注入(同义表引入的除外): \(splitLeak.prefix(3))")
        expect(termMismatch.isEmpty,
               "命中词函数与计数函数对全量标签×两个方向恒一致: \(termMismatch.prefix(3))")
        let bhTokens = QueryUnderstanding.labelTokens("balloon_hotair")
        expect(QueryUnderstanding.labelTokensMatchedTerms(bhTokens, "outdoor, balloon_hotair, balloon") == ["balloon_hotair"],
               "balloon_hotair 只命中自己的牌,同牌上的 balloon 不算(3 (2) 装机同款)")

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

        // —— 批142:标签通道【匹配级】穷尽 ——
        // 上面两条穷举只断言「词集里有这个词」(①en ∈ tokens ②∃ 同名键 ∈ tokens),
        // 而牌能不能匹配上是另一段逻辑(labelTokensMatchCount,批126 起复合词
        // 要求全部片段在词集)。本机探针(同一份源码编译)逐个标签判
        // 「labelTokens(cn) 能否匹配上牌 en」,实测:
        //   · 208 个下划线复合键里 203 个中文搜不上自己的牌(词集里是整键
        //     balloon_hotair,没有 balloon/hotair 两个片段;能搜上的 5 个是
        //     碰巧在短语词典里有带空格的写法);
        //   · adult_cat / decorative_plant / drone_machine / iron_clothing /
        //     laundry_machine 连英文直搜都搜不上(拆出的片段是泛化词,被剔);
        //   · 25 个中文名对应多个英文键(广告牌→billboard/billboards),单值
        //     反查表先到先得、先到哪个随字典遍历顺序变——同一段代码两次
        //     启动,「无人机」「音箱」一次能搜到牌、一次两个键都搜不到。
        // 穷举升级:每个标签两个方向都必须真能匹配上自己的牌
        var matchZh = 0, matchEn = 0
        for (en, cn) in LabelZh.zh {
            let zh = QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens(cn), en)
            guard zh > 0 else { fatalError("FAIL: 词典中文「\(cn)」搜不到自己的标签牌 \(en)") }
            matchZh += 1
            let typed = QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens(en), en)
            guard typed > 0 else { fatalError("FAIL: 英文标签直搜匹配不上自己的牌 \(en)") }
            matchEn += 1
        }
        expect(matchZh == LabelZh.zh.count && matchEn == LabelZh.zh.count,
               "全量 \(LabelZh.zh.count) 个标签:中文反查与英文直搜都能匹配上自己的牌")

        // 批142 单元面:整键命中、不注入片段、批126 守卫不破
        let adultCat = QueryUnderstanding.labelTokens("成年猫")
        expect(adultCat.contains("adult_cat") && !adultCat.contains("adult"),
               "「成年猫」词集=整键 adult_cat,不注入片段 adult")
        expect(QueryUnderstanding.labelTokensMatchCount(adultCat, "adult") == 0,
               "「成年猫」不命中简单词牌 adult(全部人像照不灌入)")
        expect(QueryUnderstanding.labelTokensMatchCount(adultCat, "person, adult, adult_cat") == 1,
               "「成年猫」命中自己那张牌上的 adult_cat,同牌的 person/adult 不计数")
        expect(QueryUnderstanding.labelTokensMatchCount(
            QueryUnderstanding.labelTokens("热气球"), "outdoor, balloon_hotair") == 1,
               "「热气球」命中 balloon_hotair 牌(批142 前 203 个复合键的常态:词集只有整键)")
        expect(QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens("身份证"), "play_card") == 0,
               "277 事故守卫(批126)不破:「身份证」词集含 card,仍不命中 play_card")
        expect(QueryUnderstanding.labelTokensMatchCount(["card"], "play_card, credit_card") == 0,
               "词集只有片段 card 时任何 *_card 牌都不命中")
        expect(QueryUnderstanding.labelTokensMatchCount(["blue"], "blue_sky") == 0,
               "词集只有片段 blue 时不命中 blue_sky(整键不在词集,仍走片段全中规则)")
        expect(QueryUnderstanding.labelTokensMatchCount(
            QueryUnderstanding.labelTokens("laundry_machine"), "laundry_machine") == 1,
               "laundry_machine 英文直搜命中自己的牌(machine 被当泛化词剔掉后靠整键)")
        expect(QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens("天空"), "blue_sky") == 1
               && QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens("文档"), "printed_page") == 1,
               "复合键桥接生效:天空→blue_sky、文档→printed_page(10-13 入表;文档自批125、天空自批126 起没生效)")
        expect(QueryUnderstanding.labelTokensMatchCount(
            QueryUnderstanding.labelTokens("电子产品"), "consumer_electronics") == 1
               && QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens("木头"), "wood_processed") == 1,
               "复合键桥接生效:电子产品→consumer_electronics、木头→wood_processed")
        // 批142 同名多键:原名下全部键都进词集,与字典遍历顺序无关
        let billboard = QueryUnderstanding.labelTokens("广告牌")
        expect(billboard.contains("billboard") && billboard.contains("billboards"),
               "「广告牌」两个键都在词集(此前先到先得只留一个)")
        let phone = QueryUnderstanding.labelTokens("手机")
        expect(phone.contains("cell_phone") && phone.contains("mobile_phone"),
               "「手机」两个键都在词集")
        expect(QueryUnderstanding.labelTokensMatchCount(QueryUnderstanding.labelTokens("无人机"), "drone") == 1
               && QueryUnderstanding.labelTokensMatchCount(
                    QueryUnderstanding.labelTokens("无人机"), "drone_machine") == 1,
               "「无人机」两张牌(drone / drone_machine)都命中")
        expect(LabelZh.englishKeysFromZh["手机"] == ["cell_phone", "mobile_phone"]
               && LabelZh.englishKeysFromZh["猫"] == ["cat"],
               "反查全集:手机=两键且有序;单键名只有一个键")
        // 批142 二修(验收 medium):多键名在短语里也要补全——首版只认整词原子,
        // 「无人机表演」「广告牌设计」仍随启动只捐一个键
        let droneShow = QueryUnderstanding.labelTokens("无人机表演")
        expect(droneShow.contains("drone") && droneShow.contains("drone_machine"),
               "短语里的多键名也补全:「无人机表演」两个键都在词集")
        let billboardDesign = QueryUnderstanding.labelTokens("广告牌设计")
        expect(billboardDesign.contains("billboard") && billboardDesign.contains("billboards"),
               "短语里的多键名也补全:「广告牌设计」两个键都在词集")
        let hybridBillboard = QueryUnderstanding.labelTokens("华研广告牌")
        expect(!hybridBillboard.contains("billboard") && !hybridBillboard.contains("billboards"),
               "盖不满的原子词不捐:「华研广告牌」(专名+多键名)两个键都不进词集")
        let birthdayCake = QueryUnderstanding.labelTokens("生日蛋糕")
        expect(!birthdayCake.contains("cake_regular"),
               "嵌套只留最长:「生日蛋糕」里的「蛋糕」不单独捐,不带入 cake_regular")

        // 断言数口径:全文 expect 调用 121 项(另含 1 行函数定义,批94 增 5 批98 增 3
        // 批134 增 1 批142 增 15:匹配级穷举 1 + 整键/守卫/桥接/同名多键单元面 14;
        // 批142 二修增 4:短语里的多键名 2 + 盖不满/嵌套守卫 2;
        // 批143 净增 6:批134 护栏那条改写为 3 条(词表内的键不拆片段),另增
        // 全量复合键 3 条(自身命中/片段不注入/命中词与计数一致)+ balloon_hotair 1 条)
        print("PASS: 查询理解 121 项确定性回归断言全部通过(含全量标签词典 \(LabelZh.zh.count) 词的词集穷尽与匹配级穷尽)")
    }
}
