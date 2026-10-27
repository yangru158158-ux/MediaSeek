import Foundation

/// 口语化查询理解:清洗指令词/否定子句,并为视觉通道提供中→英翻译。
/// 嵌入模型不理解否定(「没有猫不要显示」会被当成查询的一部分),
/// 也基本不懂中文→英文图文对齐(SigLIP2 以英文训练),所以查询必须先过这里。
enum QueryUnderstanding {

    /// 「找出有猫的图片、没有猫不要显示」→「猫」;「电脑屏幕的照片」→「电脑屏幕」
    static func core(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return trimmed }
        var t = trimmed

        // 1) 否定/排除子句一律截断——嵌入模型不理解否定,留着只会污染向量
        for stop in ["没有", "不要", "不想", "不用", "排除", "不显示", "别显示", "过滤", "去掉", "除了"] {
            if let r = t.range(of: stop) { t = String(t[..<r.lowerBound]) }
        }
        t = t.trimmingCharacters(in: CharacterSet(charactersIn: " 、,。;,.;;!?!?·…"))

        // 2) 去指令前缀(长词优先,只应用第一个命中的)
        let prefixes = ["帮我找出", "帮我找到", "帮我搜", "帮我查找", "帮我看看", "请找出", "请找到",
                        "我想找", "我要找", "我想看", "帮忙找", "找一下", "我想", "我要",
                        "找出", "找到", "查找", "搜寻", "搜一下", "搜索", "看看",
                        "显示", "展示", "给我看", "搜", "查", "找"]
        for p in prefixes where t.hasPrefix(p) {
            t = String(t.dropFirst(p.count)); break
        }

        // 3) 去载体后缀。两重护栏(第十五批:用户实测搜「地图」被砍成「地」,
        //    OCR 按单字「地」灌入 1,891 张、语义通道拿不到词典条目全空):
        //    ① 整词恰是词典条目(「地图」→map)——真概念词不做后缀手术;
        //    ② 裸「图」是高危后缀(地图/拼图/企图/意图都以它结尾),只在剩余
        //       词干 ≥2 字(美女图→美女)或词干本身是词典单字(猫图→猫)时才剥
        let suffixes = ["的图片", "的照片", "的相片", "的截图", "的影片", "的图",
                        "图片", "照片", "相片", "截图", "影片", "图"]
        let exactEntry = dictionary.contains { $0.0 == t }
        if !exactEntry {
            for s in suffixes where t.hasSuffix(s) {
                let stemLen = t.count - s.count
                if stemLen == 0 { continue }   // 整词即后缀(查询就是「照片」)不动
                if s == "图", stemLen == 1,
                   !dictionary.contains(where: { $0.0 == String(t.dropLast()) }) { continue }
                t = String(t.dropLast(s.count)); break
            }
        }

        // 4) 「有猫」「包含猫」这类存在前缀
        for p in ["包含", "含有", "带有", "有"] where t.hasPrefix(p) && t.count > p.count {
            t = String(t.dropFirst(p.count)); break
        }

        let out = t.trimmingCharacters(in: .whitespacesAndNewlines)
        return out.isEmpty ? trimmed : out
    }

    /// 覆盖率计算容忍的无实义连接字:「妈妈的生日」=妈妈+的+生日,
    /// 「的」不算未覆盖;「华研机器人」的「华」「研」不是连接字,照常判
    /// 部分覆盖(专名)
    private static let connectives: Set<Character> = ["的", "了", "与", "和", "及", "或", "在", "是"]

    /// 核心词命中的词典条目按出现位置拼成英文查询(视觉通道用),无命中返回 nil。
    /// 单字键只在整词完全相等时生效:「中华人民共和国」含「人」字,
    /// 但搜它不是在搜人像——子串匹配会让全库 people/adult 标签灌进来。
    /// 除本词典外还查 LabelZh 反查表:标签词典里展示的中文(如「消费电子」)
    /// 直接翻回英文标签(consumer_electronics),走标签精确通道。
    ///
    /// 第六十批覆盖率规则(「华研机器人」事故:机器/机器人两个碎片把股票
    /// 名偷换成 robot machine,「机器」直译的 machine 又借直译免疫闯过泛化
    /// 剔除,标签通道灌进上千张截图):
    /// ① 按空白拆原子词,逐词独立判覆盖;
    /// ② 一个词只有被词典碎片区间并集【盖满】(连接字除外)才捐译文——
    ///    盖满=复合描述词(屏幕截图/机器设备/妈妈的生日);盖不满=碎片
    ///    一个不捐,通道归属见 isProperNoun(第六十二批:连续生字符≥2 判
    ///    专名跳过视觉通道;单生字符/零命中照旧整词原文给 SigLIP 多语言);
    /// ③ 碎片区间嵌套时只留最长(机器人→robot,「机器」不再重复捐词,
    ///    与第五十四批「woman woman」去重同理);不嵌套的照常都留
    ///    (屏幕+截图)。
    /// 单个原子词的覆盖率评估(第六十批规则内核,english(for:) 与
    /// isProperNoun(_:) 共用):
    /// - noMatches:词内没有任何词典/标签碎片命中(词典没收的普通词,如财报)
    /// - complete:盖满(连接字可裸露);spans 已做嵌套剔除(只剔严格内含)
    ///   与位置排序,直接可捐
    /// - partial:盖不满;携带最长「连续生字符」数(未覆盖且非连接字的
    ///   连续段,连接字/已覆盖字符都会断开计数)——供 isProperNoun 区分
    ///   专名(华研=2 连续生字符)与单字修饰词(红裙子的红=1)
    private enum TermCoverage {
        case noMatches
        case complete(spans: [(start: Int, end: Int, en: String)])
        case partial(longestUnknownRun: Int)
    }

    private static func termCoverage(_ term: String) -> TermCoverage {
        // 该原子词内的全部碎片命中(区间用字符偏移)
        var matches: [(start: Int, end: Int, en: String)] = []
        let tables: [[String: String]] = [Dictionary(dictionary, uniquingKeysWith: { a, _ in a }),
                                          LabelZh.englishFromZh]
        for table in tables {
            for (zh, en) in table {
                if zh.count == 1 {
                    if term == zh { matches.append((0, term.count, en)) }
                    continue
                }
                if let r = term.range(of: zh) {
                    let s = term.distance(from: term.startIndex, to: r.lowerBound)
                    matches.append((s, s + zh.count, en))
                }
            }
        }
        guard !matches.isEmpty else { return .noMatches }
        // 嵌套只留最长:长度降序贪心选。只剔【严格内含】的(机器⊂机器人);
        // 同区间不同译词不剔(蔬菜=词典 vegetables+标签表 vegetable,单复数
        // 都要捐,标签通道才对得上)——CI 穷尽断言拦下的回归
        var spans: [(start: Int, end: Int, en: String)] = []
        for m in matches.sorted(by: {
            ($0.end - $0.start) != ($1.end - $1.start)
                ? ($0.end - $0.start) > ($1.end - $1.start)
                : $0.start < $1.start
        }) {
            if spans.contains(where: {
                $0.start <= m.start && m.end <= $0.end
                    && ($0.end - $0.start) > (m.end - m.start)
            }) { continue }
            spans.append(m)
        }
        // 覆盖率:选中区间并集 + 未覆盖字符里的连接字,须盖满全词
        var covered = Set<Int>()
        for s in spans { for i in s.start..<s.end { covered.insert(i) } }
        // 最长连续生字符段(连接字与已覆盖字符断开计数)
        var longestRun = 0, run = 0
        for (i, c) in term.enumerated() {
            if !covered.contains(i) && !connectives.contains(c) {
                run += 1
                longestRun = max(longestRun, run)
            } else {
                run = 0
            }
        }
        if longestRun > 0 {
            return .partial(longestUnknownRun: longestRun)   // 盖不满:不捐碎片
        }
        return .complete(spans: spans.sorted(by: {
            $0.start != $1.start ? $0.start < $1.start : $0.end < $1.end
        }))
    }

    static func english(for core: String) -> String? {
        var out: [String] = []
        // 同词去重是全 core 全局口径(第五十四批,验收发现曾降级为词内局部:
        // 「沙滩 沙滩」会拼出 beach beach)
        var seen = Set<String>()
        for term in core.components(separatedBy: .whitespacesAndNewlines) where !term.isEmpty {
            guard case .complete(let spans) = termCoverage(term) else { continue }
            // 捐出(spans 已按位置排序;seen 全 core 作用域,同词只捐一次)
            for m in spans where seen.insert(m.en).inserted {
                out.append(m.en)
            }
        }
        return out.isEmpty ? nil : out.joined(separator: " ")
    }

    /// 专名判定(第六十二批,211 用户实测:华研机器人裸搜仍出 96 张「语义」
    /// 命中——第六十批只管住了碎片捐赠,整词原文兜底还是把词交给了 SigLIP
    /// 多语言,行业相关截图的噪声带分数(≥第三档地板)整批捞进结果)。
    /// 判据=覆盖率规则的通道延伸:词内有词典碎片命中、但盖不住的连续生字符
    /// ≥2(华研机器人里的「华研」)→ 专名,整词只走文字通道(OCR/用户标签),
    /// 视觉通道整词跳过——SigLIP 只能把专名嵌成品类语义(机器人/科技/金融),
    /// 区分不了「关于这家公司」和「关于这个行业」,真命中全来自 OCR。
    /// 只 1 个生字符(红裙子的「红」)是普通修饰词形态,保持批 60 现状
    /// (整词原文给 SigLIP);零命中(财报)不是专名,词典没收的普通词,
    /// 照走多语言原文。
    static func isProperNoun(_ term: String) -> Bool {
        if case .partial(let run) = termCoverage(term) { return run >= 2 }
        return false
    }

    /// 词典「零命中」判定(第六十四批人名闸配套):词内没有任何词典/标签
    /// 碎片命中(管雯/财报/woman)。注意 english(for:)==nil 比它宽——还涵盖
    /// 盖不满的修饰词(红裙子:命中裙子但红裸露);人名闸只对真零命中生效,
    /// 批 62「单生字符修饰词照旧整词原文给 SigLIP」的承诺在任何条件下不破
    /// (验收拦下:闸用 english==nil 时红裙子类词在命中标签条件下会被误伤)
    static func isDictionaryUnknown(_ term: String) -> Bool {
        if case .noMatches = termCoverage(term) { return true }
        return false
    }

    /// 泛化标签词:长查询(≥3 字)不带这些词去匹配标签——
    /// 它们什么照片都能沾上,是「一个词灌出全库」的元凶
    static let genericTokens: Set<String> = [
        "people", "person", "adult", "human", "document", "screenshot",
        "text", "structure", "outdoor", "machine", "clothing", "plant",
    ]

    /// 标签通道比对词集(第四十五批自 SearchEngine 迁入,纳入 CI 断言):
    /// 原文分词 + 词典英译分词 → 一跳同义扩展 → 泛化词剔除。
    /// 剔除规则(第四十八批,验收拦下「屏幕截图」事故后重立):
    /// 照片标签牌按同一本词典挂,词典直译词就是显示词的本体,
    /// 泛化剔除只针对用户手敲的英文词,直译词一律免疫——
    /// 「屏幕截图」直译 screen+screenshot,旧规则(混合直译剔泛化)把
    /// screenshot 剔掉,对着 5197 张挂 screenshot 标签牌的照片把标签通道
    /// 关掉;「机器设备」同理。手敲词的守卫不变:全是泛化词时保留
    /// (machine 本尊直搜,剔它等于删掉查询本体),带实义词时照常剔防灌水
    static func labelTokens(_ query: String) -> Set<String> {
        // 第七十二批:繁→简归一——标签词典全是简体键,繁体词原样查零命中;
        // 幂等,简体查询零变化
        let query = simplified(query)
        let raw = Set(query.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty })
        var tokens = raw
        var dictTokens = Set<String>()
        if let en = english(for: query) {
            for t in en.lowercased().split(separator: " ") {
                let s = String(t)
                tokens.insert(s)
                dictTokens.insert(s)
            }
        }
        if query.count >= 3 {
            let typed = raw.filter { $0.allSatisfy(\.isASCII) }
            let typedAllGeneric = !typed.isEmpty && typed.subtracting(genericTokens).isEmpty
            tokens = typedAllGeneric
                ? expandedTokens(tokens)
                : expandedTokens(tokens).subtracting(genericTokens.subtracting(dictTokens))
        } else {
            // 第五十六批:短词(2字)短路保护只护「种子词」(原词+词典直译)——
            // 同义扩展不得注入泛化词。「女人」(2字)走本分支,woman 的同义扩展
            // 注入 person/people,标签通道对全库人像标签命中 4,279 张灌进语义
            // 结果(207 用户实测,搜得还特别慢:4千张逐张取图)。剔除只针对
            // 扩展注入的泛化词,种子里的保留:「朋友」直译 friend people 的
            // people 是种子,不受影响(48 批设计);「机器」直译 machine 同理
            let seed = tokens
            tokens = expandedTokens(tokens).subtracting(genericTokens.subtracting(seed))
        }
        return tokens
    }

    /// 性别词对照(第五十批引入,第五十五批重标定):SigLIP 绝对分数区分
    /// 男女弱(0.08 地板男女同放行),相对差才可靠。对比串=文献标准形式
    /// 「a photo of a woman/man」,真实照片实测边距:女 +0.030~+0.049 /
    /// 男 -0.056~-0.023 / 中性 -0.002~+0.004,过滤线 0.010
    /// (见 genderEnglish 注释与 SearchEngine 对比过滤)
    static let genderOpposites: [String: String] = [
        "woman": "man", "man": "woman",
        "women": "men", "men": "women",
        "girl": "boy", "boy": "girl",
        "female": "male", "male": "female",
        "lady": "gentleman", "gentleman": "lady",
        "女人": "man", "男人": "woman",
        "女生": "boy", "男生": "girl",
        "女士": "gentleman", "男士": "lady",
        "女孩": "boy", "男孩": "girl",
        "美女": "man", "帅哥": "woman",
        "母亲": "man", "父亲": "woman",
    ]

    /// 查询词是性别词时返回反义词(供对比过滤),非性别词返回 nil
    static func genderOpposite(of query: String) -> String? {
        let key = query.lowercased().trimmingCharacters(in: .whitespaces)
        return genderOpposites[key]
    }

    /// 性别词→规范英文单数(第五十五批)。对比串按文献标准形式(CVPR 2025
    /// 零样本性别分类等:"a photo of a man/woman" 带冠词小写 + argmax),
    /// 真实照片实测边距:女 +0.030~+0.049 / 男 -0.056~-0.023 / 中性 ±0.004
    /// (无冠词模板与画作样本都显著弱于真人照片+标准提示词,勿再混用)
    static let genderEnglish: [String: String] = [
        "woman": "woman", "women": "woman", "man": "man", "men": "man",
        "girl": "girl", "boy": "boy", "female": "woman", "male": "man",
        "lady": "lady", "gentleman": "gentleman",
        "女人": "woman", "男人": "man", "女生": "girl", "男生": "boy",
        "女士": "lady", "男士": "gentleman", "女孩": "girl", "男孩": "boy",
        "美女": "woman", "帅哥": "man", "母亲": "woman", "父亲": "man",
    ]

    /// 性别词的对比串对(文献标准形式)。非性别词返回 nil。
    /// 女性词 → (a photo of a woman, a photo of a man);男性词反向。
    /// 两侧都过 genderEnglish 归一(复数→单数:men 的反义 women → woman)
    static func genderContrast(of query: String) -> (fwd: String, opp: String)? {
        let key = query.lowercased().trimmingCharacters(in: .whitespaces)
        guard let oppRaw = genderOpposites[key] else { return nil }
        let fwd = genderEnglish[key]
            ?? (key.isEmpty || !key.allSatisfy(\.isASCII) ? nil : key)
        guard let fwd else { return nil }
        let opp = genderEnglish[oppRaw] ?? oppRaw
        return ("a photo of a \(fwd)", "a photo of a \(opp)")
    }

    /// 批82:底网豁免白名单——只有「卡面不印这个词」的证件词才享
    /// floorTierPermitted 豁免。判据是卡的物理属性,不是模型:驾驶证正面
    /// 大字印着「机动车驾驶证」→ 文字通道(含别名)在标准版和 so400m 都
    /// 直接命中实体卡,豁免只剩杂音(231 装机实测搜「驾驶证」结果过多)。
    /// 身份证正面只有「公民身份号码」、证件是泛词、护照卡面印英文——
    /// 这三个的实体卡只有底网能捞(批 43 先例 0.05-0.08)。已知取舍:
    /// 糊到 OCR 认不出的驾驶证照片两模型都会漏(0.18/0.157 门槛都够不着)
    static let floorExemptWords: Set<String> = ["身份证", "护照", "港澳通行证", "户口", "证件"]

    /// 证件类 OCR 别名:人像面不印证件名(身份证正面只有「公民身份号码」「姓名」等字段),
    /// 只搜证件名会漏掉全部正面照——补字段级别名供文字通道并集保召回。
    /// 「身份證」繁体变体:香港身份证印的是繁体+英文(HONG KONG IDENTITY CARD),
    /// 简体「身份证」子串对不上,不补则中文查询漏掉全部港证(第六十九批)
    static let ocrAliases: [String: [String]] = [
        "身份证": ["公民身份", "居民身份证", "签发机关", "身份證"],        "护照": ["passport", "签发国"],
        "驾驶证": ["准驾车型", "机动车驾驶证"],
        "驾照": ["准驾车型", "机动车驾驶证"],
        "港澳通行证": ["往来港澳"],
        "户口": ["户号", "户主"],
        // 第七十二批:证件= licences 族的上位中文词,与英文桥同权(221 装机实测
        // 搜「证件」文字 98/语义 0——证件照 0.05-0.12 过不了上两档,第三档被
        // 「文字已有命中」挡;入别名表同时接上文字追加与 floorTierPermitted 豁免)
        "证件": ["身份证", "护照", "驾驶证"],
    ]

    /// 第六十九批:英文→中文证件桥(纯加法,不做查询级改写)。外国用户输英文
    /// 找中国大陆证件时两路皆断:文字通道缺别名(大陆证件只印中文,OCR 里没有
    /// 英文),语义缺证件类第三档豁免(证件照 SigLIP 只有 0.10-0.12,过不了
    /// 上两档)。桥到中文对应词后,ocrAliases 并集与 floorTierPermitted 豁免
    /// 都接上。⚠️ 不改写查询的原因:香港身份证靠 OCR 文本里的英文 "identity"
    /// 命中(218 用户实测 identity card/identity/passport 文字通道全对),换成
    /// 简体中文词会把这些已验收命中改丢——所以只在通道层「追加减式」。
    /// licence 入表是第七十一批翻案:六十九批排除它的依据是 218 实测(so400m)
    /// 「licence 语义全对」——但那是 so400m 的分数分布,标准版下证件照够不着
    /// 第二档、第三档又被「文字已有命中」规则挡住,licence 语义恒 0(220 用户
    /// 实测)。门槛不可跨模型迁移,排除的前提随模型消失;入桥=模型无关修法。
    /// license(美式拼写)第七十四批入表(用户批准):与 licence 同词异拼,
    /// 对称面原则——只桥英式不桥美式,美式用户撞同一个洞。
    /// driving/card 泛词不入(会窄化成驾照)
    static let chineseCounterparts: [String: String] = [
        "id": "身份证", "identity": "身份证",
        "id card": "身份证", "identity card": "身份证", "identity document": "身份证",
        "passport": "护照",
        "driver license": "驾照", "driving license": "驾照", "driver's license": "驾照",
        "licence": "驾照", "license": "驾照",
    ]

    /// 英文词的中文对应证件词(大小写/首尾空白不敏感);非表内词返回 nil 走原路
    static func chineseCounterpart(for term: String) -> String? {
        chineseCounterparts[term.lowercased().trimmingCharacters(in: .whitespaces)]
    }

    /// 桥接追加的文字变体 = [中文对应词] + 其 OCR 别名;表外词返回空(原匹配零改动)
    static func bridgedVariants(for term: String) -> [String] {
        guard let zh = chineseCounterpart(for: term) else { return [] }
        return [zh] + (ocrAliases[zh] ?? [])
    }

    /// 第七十二批:繁→简字符映射(单遍逐字替换)。自建表不用系统的
    /// applyingTransform——ICU 在 swift-corelibs(Windows 本地测试/ubuntu CI)
    /// 与 iOS 真机行为不一致,三端一致的确定性只能靠自己的表。收录范围:
    /// 证件家族 + 词典词条的繁体形 + 日常/金融/数码高频字;简繁同形字
    /// 天然透传无需收录。源字符互不相同且均非另一源的目标(单遍无链式),
    /// 镜像测试守这两条不变量。收录判据:标准一一对一映射,多向歧义字
    /// (乾/發複合语境类)不收——漏一个字只是该词不转换(回到原行为),
    /// 转错一个字才是事故
    static let t2sChars: [Character: Character] = [
        // 证件/政务
        "證": "证", "駕": "驾", "駛": "驶", "護": "护", "戶": "户", "傳": "传",
        "辦": "办", "廳": "厅", "處": "处", "臺": "台", "灣": "湾", "廣": "广",
        "東": "东", "區": "区", "縣": "县", "鎮": "镇", "鄉": "乡", "檯": "台",
        // 金融/财报
        "團": "团", "騰": "腾", "訊": "讯", "億": "亿", "匯": "汇", "豐": "丰",
        "寧": "宁", "龍": "龙", "馬": "马", "鳳": "凤", "華": "华", "榮": "荣",
        "達": "达", "興": "兴", "業": "业", "銀": "银", "錢": "钱", "財": "财",
        "報": "报", "紙": "纸", "帳": "账", "貸": "贷", "權": "权", "潤": "润",
        "資": "资", "產": "产", "銷": "销", "買": "买", "賣": "卖", "貴": "贵",
        "費": "费", "貨": "货", "賺": "赚", "虧": "亏", "總": "总", "統": "统",
        "計": "计", "會": "会", "記": "记", "錄": "录", "單": "单", "據": "据",
        "發": "发", "債": "债", "償": "偿", "質": "质", "稅": "税",
        // 数码/生活
        "電": "电", "話": "话", "機": "机", "腦": "脑", "網": "网", "絡": "络",
        "視": "视", "頻": "频", "遊": "游", "戲": "戏", "軟": "软", "體": "体",
        "運": "运", "動": "动", "車": "车", "庫": "库", "飛": "飞", "場": "场",
        "檔": "档", "飯": "饭", "麵": "面", "飲": "饮", "藥": "药", "醫": "医",
        "學": "学", "書": "书", "圖": "图", "館": "馆", "雜": "杂", "誌": "志",
        "門": "门", "燈": "灯", "風": "风", "雲": "云", "氣": "气", "陽": "阳",
        "陰": "阴", "樹": "树", "鳥": "鸟", "魚": "鱼", "貓": "猫", "蟲": "虫",
        "驢": "驴", "駱": "骆", "駝": "驼", "傘": "伞", "雞": "鸡", "鴨": "鸭",
        "鵝": "鹅", "蝦": "虾", "貝": "贝", "豬": "猪", "燒": "烧", "滷": "卤",
        "醬": "酱", "醃": "腌", "檸": "柠", "湯": "汤", "鍋": "锅", "盤": "盘",
        // 人物/动作/时间
        "們": "们", "誰": "谁", "個": "个", "沒": "没", "說": "说", "請": "请",
        "謝": "谢", "對": "对", "錯": "错", "認": "认", "識": "识", "讓": "让",
        "給": "给", "來": "来", "雙": "双", "兩": "两", "見": "见", "聽": "听",
        "寫": "写", "讀": "读", "習": "习", "績": "绩",
        "點": "点", "鐘": "钟", "歲": "岁", "時": "时", "間": "间", "週": "周",
        // 颜色/形状
        "紅": "红", "黃": "黄", "綠": "绿", "藍": "蓝", "圓": "圆", "長": "长",
        "寬": "宽",
        // 常用虚词/副词
        "愛": "爱", "歡": "欢", "樂": "乐", "開": "开", "關": "关", "閉": "闭",
        "滿": "满", "過": "过", "進": "进", "內": "内", "裡": "里", "裏": "里",
        "邊": "边", "隨": "随", "選": "选", "擇": "择", "確": "确", "為": "为",
        "從": "从", "於": "于", "與": "与", "這": "这", "麼": "么", "幾": "几",
        "隻": "只", "後": "后", "髮": "发", "鬆": "松",
        // 器物/金属
        "轎": "轿", "軌": "轨", "輪": "轮", "鏈": "链", "鎖": "锁",
        "鈕": "钮", "鈴": "铃", "銅": "铜", "鐵": "铁", "鋼": "钢", "錫": "锡",
        "釘": "钉", "針": "针", "釣": "钓", "褲": "裤", "襪": "袜", "圍": "围",
        // 工作文牍
        "藝": "艺", "術": "术", "攝": "摄", "標": "标", "籤": "签", "題": "题",
        "頁": "页", "碼": "码", "數": "数", "郵": "邮", "遞": "递", "輸": "输",
        "輛": "辆", "療": "疗", "診": "诊", "經": "经", "級": "级", "試": "试",
        "勝": "胜", "敗": "败", "軍": "军", "亞": "亚", "賽": "赛", "隊": "队",
        "員": "员", "練": "练", "營": "营", "養": "养", "補": "补", "維": "维",
        "廚": "厨", "衛": "卫", "廁": "厕", "寢": "寝", "節": "节", "禮": "礼",
        "賓": "宾", "結": "结", "聯": "联", "職": "职", "顧": "顾", "諮": "咨",
        "詢": "询", "談": "谈", "議": "议", "決": "决", "執": "执", "專": "专",
        "項": "项", "務": "务", "導": "导", "師": "师", "範": "范", "訓": "训",
        "檢": "检", "驗": "验", "測": "测", "設": "设", "備": "备", "裝": "装",
        "遷": "迁", "樓": "楼", "麗": "丽", "勳": "勋", "塵": "尘", "夢": "梦",
        "幹": "干", "劃": "划", "畫": "画", "沖": "冲", "衝": "冲",
    ]

    /// 繁→简归一(逐字单遍替换,简体/同形字原样透传;幂等)
    static func simplified(_ s: String) -> String {
        String(s.map { t2sChars[$0] ?? $0 })
    }

    /// 第七十二批:文字/标签类通道的查询变体 = 原词 + 繁→简转换词(不同才追加)。
    /// 双向无损:原词照常匹配(「美團」继续命中繁体月结单),转换词补简体侧
    /// (「駕駛證」同时命中大陆简体「驾驶证」OCR)。语义通道不走这里——它在
    /// 自己的入口直接归一(词典/模板/豁免整体接简体键)
    static func scriptVariants(of term: String) -> [String] {
        let s = simplified(term)
        return s == term ? [term] : [term, s]
    }

    /// 英文同义扩展:标签是 woman、查询是 girl 时精确匹配会漏,必须互相打通。
    /// 下划线标签(blue_sky/printed_page 等)是独立 token,与近义标签互相桥接
    static let synonyms: [String: [String]] = [
        "person": ["people", "human", "adult", "man", "woman"],
        "people": ["person", "adult", "crowd"],
        "man": ["male", "boy", "guy", "gentleman", "person", "people"], "men": ["man", "people"],
        "woman": ["girl", "female", "lady", "person", "people"], "women": ["woman", "people"],
        "girl": ["woman", "female", "person", "people"], "lady": ["woman", "female"],
        "guy": ["man", "male", "boy"], "gentleman": ["man", "male"],
        "female": ["woman", "girl", "person", "people"],
        "boy": ["man", "male", "person", "people"], "male": ["man", "boy", "guy"],
        "baby": ["infant"], "child": ["kid", "baby"], "kid": ["child"],
        "cat": ["kitten"], "kitten": ["cat"],
        "dog": ["puppy"], "puppy": ["dog"],
        "car": ["vehicle", "automobile"],
        "screen": ["monitor", "display"], "monitor": ["screen", "display"],
        "food": ["meal", "dish"], "flower": ["blossom"],
        "sea": ["ocean"], "ocean": ["sea"], "beach": ["sea", "shore", "coast"],
        "document": ["paperwork", "text", "printed_page"], "text": ["document"],
        "printed_page": ["document", "paper"],
        "portrait": ["face", "person"], "face": ["portrait"],
        "beautiful": ["pretty", "attractive"],
        // Vision 标签体系内部桥接:同一画面常打近义词的不同变体
        "sky": ["blue_sky", "cloudy"], "blue_sky": ["sky"],
        "cloud": ["cloudy"], "cloudy": ["cloud"],
        "building": ["structure", "architecture"], "structure": ["building"],
        "electronics": ["consumer_electronics", "device"],
        "consumer_electronics": ["electronics"],
        "wood": ["wood_processed"], "wood_processed": ["wood"],
        "plant": ["tree", "flower"], "tree": ["plant"],
        "clothes": ["clothing", "apparel"], "clothing": ["clothes"],
        "animal": ["pet"], "pet": ["animal"],
        "vehicle": ["car"], "sunset": ["sunrise"], "sunrise": ["sunset"],
        "meal": ["food", "dish"], "dish": ["food", "meal"],
        "fruit": ["food"], "dessert": ["cake", "pastry"], "cake": ["dessert"],
        "grass": ["lawn", "meadow"], "lawn": ["grass"],
        "mountain": ["hill"], "hill": ["mountain"],
        "river": ["lake"], "lake": ["river"],
    ]

    /// 词集做一跳同义扩展
    static func expandedTokens(_ tokens: Set<String>) -> Set<String> {
        var out = tokens
        for t in tokens {
            if let syn = synonyms[t] { out.formUnion(syn) }
        }
        return out
    }

    /// 常用拍摄主体/场景中→英(SigLIP2 通道);命中方式=子串,长词更优先靠排序位置自然处理
    static let dictionary: [(String, String)] = [
        // 动物
        ("小猫", "cat"), ("猫咪", "cat"), ("猫", "cat"), ("小狗", "dog"), ("狗狗", "dog"),
        ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"), ("兔子", "rabbit"),
        ("老鼠", "mouse"), ("乌龟", "turtle"), ("蝴蝶", "butterfly"), ("昆虫", "insect"),
        ("狮子", "lion"), ("老虎", "tiger"), ("大象", "elephant"), ("猴子", "monkey"),
        ("熊猫", "panda"), ("海鸥", "seagull"), ("鸭子", "duck"), ("鸡", "chicken"),
        // 植物/自然
        ("花", "flower"), ("玫瑰", "rose"), ("树", "tree"), ("森林", "forest"), ("草", "grass"),
        ("叶子", "leaf"), ("山", "mountain"), ("雪", "snow"), ("海", "sea"), ("海滩", "beach"),
        ("湖", "lake"), ("河", "river"), ("瀑布", "waterfall"), ("天空", "sky"),
        ("沙滩", "beach"), ("海边", "beach seaside"),   // 口语变体:只有「海滩」在册,搜「沙滩」语义通道空转(第十九批)
        ("日出", "sunrise"), ("日落", "sunset"), ("月亮", "moon"), ("星星", "stars"),
        ("云", "clouds"), ("彩虹", "rainbow"), ("田野", "field"), ("草地", "lawn"),
        // 城市/场所
        ("建筑", "building"), ("桥", "bridge"), ("塔", "tower"), ("寺庙", "temple"),
        ("教堂", "church"), ("城堡", "castle"), ("城市", "city"), ("乡村", "countryside"),
        ("马路", "road"), ("街道", "street"), ("公路", "highway"), ("公园", "park"),
        ("游乐园", "amusement park"), ("动物园", "zoo"), ("植物园", "botanical garden"),
        ("博物馆", "museum"), ("图书馆", "library"), ("学校", "school"), ("医院", "hospital"),
        ("酒店", "hotel"), ("机场", "airport"), ("车站", "station"), ("商场", "shopping mall"),
        ("超市", "supermarket"), ("餐厅", "restaurant"), ("咖啡厅", "cafe"), ("银行", "bank"),
        // 交通
        ("汽车", "car"), ("跑车", "sports car"), ("火车", "train"), ("地铁", "subway"),
        ("公交车", "bus"), ("自行车", "bicycle"), ("摩托车", "motorcycle"),
        ("飞机", "airplane"), ("船", "boat"), ("轮船", "ship"), ("卡车", "truck"),
        // 食物
        ("食物", "food"), ("火锅", "hot pot"), ("面条", "noodles"), ("米饭", "rice"),
        ("蛋糕", "cake"), ("面包", "bread"), ("水果", "fruit"), ("苹果", "apple"),
        ("香蕉", "banana"), ("橙子", "orange"), ("葡萄", "grapes"), ("西瓜", "watermelon"),
        ("草莓", "strawberry"), ("咖啡", "coffee"), ("奶茶", "milk tea"), ("茶", "tea"),
        ("蔬菜", "vegetables"), ("沙拉", "salad"), ("寿司", "sushi"), ("披萨", "pizza"),
        ("冰淇淋", "ice cream"), ("烧烤", "barbecue"),
        // 人物/事件
        ("人", "person"), ("人们", "people"), ("人物", "person"), ("人像", "portrait person"),
        ("女生", "girl"), ("男生", "boy"), ("女孩", "girl"), ("男孩", "boy"),
        ("男人", "man"), ("女人", "woman"),
        ("男性", "man male"), ("女性", "woman female"),   // 第四十四批:性别词补全
        ("男士", "man gentleman"), ("女士", "lady woman"),
        ("小伙", "young man"), ("姑娘", "girl young woman"),
        ("孩子", "child"), ("婴儿", "baby"),
        ("老人", "elderly person"), ("自拍", "selfie"), ("合影", "group photo"),
        ("婚礼", "wedding"), ("生日", "birthday"), ("派对", "party"), ("毕业", "graduation"),
        ("舞蹈", "dancing"), ("唱歌", "singing"), ("运动", "sports"), ("跑步", "running"),
        ("游泳", "swimming"), ("足球", "soccer"), ("篮球", "basketball"),
        ("羽毛球", "badminton"), ("网球", "tennis"), ("爬山", "hiking"), ("瑜伽", "yoga"),
        ("动物", "animal"), ("风景", "landscape scenery"), ("植物", "plant"),
        ("美女", "beautiful woman"), ("帅哥", "handsome man"),
        ("合照", "group photo"), ("同学", "student people"), ("老师", "teacher"),
        ("家人", "family"), ("爸爸", "father man"), ("妈妈", "mother woman"),
        ("朋友", "friend people"), ("宝宝", "baby"),
        ("吃的", "food"), ("好吃的", "food"), ("饮品", "drink"),
        // 证件/文书(同时带上 document 词,便于命中文档类标签)
        ("证件", "id card"), ("证件照", "id portrait"),
        ("身份证", "id card"), ("驾照", "driver license card"),
        // 第七十二批:驾驶证此前无词典对(只有驾照有)——繁体「駕駛證」归一到
        // 驾驶证后若查不到英文,模板还是会退回裸多语言嵌入,补齐与驾照同款
        // 第八十三批:document→card。232 装机实测搜「驾驶证」语义 4,023——
        // "document" 泛词对文档密集图库(银行月结单/收据/票据全是打印物)
        // 在 tier-2(0.18) 大面积过线,tier-2 命中≥30 不再降档,底网豁免
        // 修复管不到这一层。收窄成卡片形态词只匹配实体证卡;文字通道
        // 不受影响(ocrAliases 已有 机动车驾驶证/准驾车型 字段别名)
        ("驾驶证", "driver license card"),
        // 第六十九批:护照此前无词典对,中文查询走裸多语言嵌入——补 passport
        // 模板(与 ocrAliases 的护照条目、英文桥三处同族)
        ("护照", "passport"),
        ("文件", "document"), ("文字", "text document"), ("pdf", "document"),
        ("网页", "website screenshot"), ("聊天记录", "chat messages screenshot"),
        ("聊天", "chat conversation messages"), ("对话", "chat conversation messages"),
        ("消息", "chat messages"), ("微信", "wechat chat app"), ("qq", "qq chat app"),
        ("桌面", "computer desktop screen"),
        // 物品/电子
        ("手机", "smartphone"), ("电脑", "computer"), ("笔记本", "laptop"),
        ("屏幕", "screen"), ("显示器", "monitor"), ("电视", "television"),
        ("键盘", "keyboard"), ("鼠标", "computer mouse"), ("耳机", "headphones"),
        ("相机", "camera"), ("手表", "wristwatch"), ("眼镜", "glasses"),
        ("沙发", "sofa"), ("床", "bed"), ("椅子", "chair"), ("桌子", "table"),
        // 第六十批:补「设备」词条,让「机器设备」这类并列复合词达成全覆盖
        // (机器+设备),覆盖率规则下照常直译;不补则「设备」两字悬空判专名
        ("设备", "equipment"),
        ("办公室", "office"), ("厨房", "kitchen"), ("卧室", "bedroom"),
        ("浴室", "bathroom"), ("客厅", "living room"), ("电梯", "elevator"),
        ("楼梯", "stairs"), ("窗户", "window"), ("灯", "lamp"), ("垃圾桶", "trash can"),
        ("行李箱", "luggage"), ("雨伞", "umbrella"), ("书", "book"), ("信封", "envelope"),
        // 文档/文字类
        ("截图", "screenshot"), ("文档", "document"), ("合同", "contract"),
        ("发票", "invoice"), ("收据", "receipt"), ("名片", "business card"),
        ("手写", "handwriting"), ("图表", "chart"), ("表格", "spreadsheet"),
        ("地图", "map"), ("菜单", "menu"), ("海报", "poster"), ("漫画", "cartoon"),
        ("油画", "painting"), ("涂鸦", "graffiti"), ("二维码", "qr code"),
        ("证件", "id card"), ("护照", "passport"), ("门票", "ticket"),
        ("快递单", "shipping label"), ("说明书", "manual"), ("笔记", "notes"),
        // 旅行/目的地(英文描述短语供 SigLIP 检索;套官方 "This is a photo of X." 模板)
        ("马尔代夫", "maldives tropical island beach"), ("三亚", "sanya tropical beach"),
        ("巴厘岛", "bali tropical beach"), ("普吉岛", "phuket tropical beach"),
        ("泰国", "thailand"), ("日本", "japan"), ("韩国", "korea"),
        ("夏威夷", "hawaii beach"), ("新加坡", "singapore"), ("欧洲", "europe travel"),
        ("杭州", "hangzhou west lake city scenery"), ("北京", "beijing city landmarks"),
        ("上海", "shanghai city skyline"), ("广州", "guangzhou city"),
        ("深圳", "shenzhen city skyline"), ("香港", "hong kong city skyline"),
        ("澳门", "macau city"), ("南京", "nanjing city"), ("苏州", "suzhou garden"),
        ("厦门", "xiamen coastal city"), ("青岛", "qingdao coastal city"),
        ("大连", "dalian coastal city"), ("成都", "chengdu city"), ("重庆", "chongqing city night"),
        ("西安", "xian ancient city"), ("武汉", "wuhan city"), ("长沙", "changsha city"),
        ("天津", "tianjin city"), ("昆明", "kunming city"), ("大理", "dali yunnan"),
        ("丽江", "lijiang old town"), ("桂林", "guilin karst landscape"),
        ("黄山", "huangshan mountain"), ("哈尔滨", "harbin winter snow"),
        ("西藏", "tibet plateau mountains"), ("巴黎", "paris"), ("东京", "tokyo city"),
        ("首尔", "seoul city"), ("纽约", "new york city"), ("伦敦", "london"), ("曼谷", "bangkok"),
        ("台北", "taipei 101 night market"), ("高雄", "kaohsiung harbor city"),
        ("福州", "fuzhou city river"), ("台湾", "taiwan city coastline"),
        // 颜色(视觉词典兜底;主通道是颜色筛选器)
        ("蓝色", "blue"), ("红色", "red"), ("绿色", "green"), ("黄色", "yellow"),
        ("黑色", "black"), ("白色", "white"), ("粉色", "pink"), ("紫色", "purple"),
        ("灰色", "gray"),
    ]

    /// 裸地名集合:整个查询恰为地名时走 GPS 地点浏览。
    /// 城市名没有视觉特征,SigLIP 不可能从画面认出「南京」——相册的拍摄地点
    /// 元数据才是确定解(词典英译只对有视觉锚点的词有效,如「杭州→west lake」)。
    static let placeNames: Set<String> = [
        // 直辖市/港澳台
        "北京", "上海", "天津", "重庆", "香港", "澳门", "台湾", "台北", "新北", "高雄",
        "台中", "台南", "桃园", "基隆",
        // 华东
        "杭州", "南京", "苏州", "无锡", "常州", "南通", "嘉兴", "湖州", "绍兴", "宁波",
        "温州", "金华", "台州", "合肥", "福州", "厦门", "泉州", "漳州", "南昌", "青岛",
        "济南", "烟台", "潍坊", "临沂",
        // 华南/华中/华北
        "广州", "深圳", "珠海", "佛山", "东莞", "中山", "惠州", "汕头", "湛江", "武汉",
        "长沙", "郑州", "洛阳", "开封", "太原", "石家庄", "呼和浩特",
        // 西南/西北/东北
        "成都", "西安", "昆明", "大理", "丽江", "贵阳", "遵义", "南宁", "桂林", "海口",
        "三亚", "兰州", "西宁", "银川", "乌鲁木齐", "拉萨", "哈尔滨", "长春", "沈阳", "大连",
        // 自然/旅行目的地
        "黄山", "西藏", "桂林阳朔", "敦煌", "张掖", "稻城", "香格里拉", "西双版纳",
        "青海湖", "呼伦贝尔", "长白山", "张家界", "九寨沟", "峨眉山", "泰山", "华山",
        // 国际
        "马尔代夫", "巴厘岛", "普吉岛", "清迈", "芭堤雅", "曼谷", "新加坡", "吉隆坡",
        "槟城", "日本", "东京", "大阪", "京都", "冲绳", "名古屋", "首尔", "釜山",
        "夏威夷", "塞班", "长滩", "宿务", "迪拜", "土耳其", "埃及", "希腊", "圣托里尼",
        "冰岛", "瑞士", "法国", "巴黎", "普罗旺斯", "意大利", "罗马", "威尼斯", "西班牙",
        "葡萄牙", "英国", "伦敦", "德国", "挪威", "芬兰", "瑞典", "俄罗斯", "美国",
        "纽约", "洛杉矶", "旧金山", "拉斯维加斯", "夏威夷", "加拿大", "多伦多", "温哥华",
        "澳大利亚", "悉尼", "墨尔本", "新西兰", "斐济", "塞舌尔", "毛里求斯", "大溪地",
        "欧洲", "东南亚",
    ]

    /// 整个查询恰为一个地名时返回它(供 GPS 地点浏览);否则 nil
    static func barePlace(_ query: String) -> String? {
        let q = query.trimmingCharacters(in: .whitespaces)
        return placeNames.contains(q) ? q : nil
    }
}
