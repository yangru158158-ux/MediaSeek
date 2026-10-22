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

    /// 核心词命中的词典条目按出现位置拼成英文查询(视觉通道用),无命中返回 nil。
    /// 单字键只在整词完全相等时生效:「中华人民共和国」含「人」字,
    /// 但搜它不是在搜人像——子串匹配会让全库 people/adult 标签灌进来。
    /// 除本词典外还查 LabelZh 反查表:标签词典里展示的中文(如「消费电子」)
    /// 直接翻回英文标签(consumer_electronics),走标签精确通道。
    static func english(for core: String) -> String? {
        var matches: [(pos: Int, en: String)] = []
        for (zh, en) in dictionary {
            if zh.count == 1 {
                if core == zh { matches.append((0, en)) }
                continue
            }
            if let r = core.range(of: zh) {
                matches.append((core.distance(from: core.startIndex, to: r.lowerBound), en))
            }
        }
        for (cn, en) in LabelZh.englishFromZh {
            if cn.count == 1 {
                if core == cn { matches.append((0, en)) }
                continue
            }
            if let r = core.range(of: cn) {
                matches.append((core.distance(from: core.startIndex, to: r.lowerBound), en))
            }
        }
        guard !matches.isEmpty else { return nil }
        // 第五十四批:同词去重——「女人」同时命中 dictionary 与 englishFromZh
        // 各拼一次得到 "woman woman",重复稀释性别/主体信号(语义通道泛化灌水
        // 元凶之一);标签通道不受影响(token 化成 Set 本就去重)
        var seen = Set<String>()
        var out: [String] = []
        for m in matches.sorted(by: { $0.pos < $1.pos }) where seen.insert(m.en).inserted {
            out.append(m.en)
        }
        return out.joined(separator: " ")
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
            tokens = expandedTokens(tokens)
        }
        return tokens
    }

    /// 性别词对照(第五十批):SigLIP 绝对分数区分男女弱(男照 0.06-0.10 与
    /// 女照 0.10-0.11 同带,0.08 地板一起放行——用户实测搜 woman 杨如程玉婷
    /// 混列),相对差才是可靠信号(实测:女照 woman分-man分=+0.021 / 男照
    /// -0.027~-0.038 / 中性 -0.004)。查询是性别词时视觉通道同时跑反义词,
    /// 只留「正向分-反向分 ≥ 0.012」的照片(见 SearchEngine 对比过滤)
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
    /// 女性词 → (a photo of a woman, a photo of a man);男性词反向
    static func genderContrast(of query: String) -> (fwd: String, opp: String)? {
        let key = query.lowercased().trimmingCharacters(in: .whitespaces)
        guard let opp = genderOpposites[key] else { return nil }
        let fwd = genderEnglish[key]
            ?? (key.isEmpty || !key.allSatisfy(\.isASCII) ? nil : key)
        guard let fwd else { return nil }
        return ("a photo of a \(fwd)", "a photo of a \(opp)")
    }

    /// 证件类 OCR 别名:人像面不印证件名(身份证正面只有「公民身份号码」「姓名」等字段),
    /// 只搜证件名会漏掉全部正面照——补字段级别名供文字通道并集保召回
    static let ocrAliases: [String: [String]] = [
        "身份证": ["公民身份", "居民身份证", "签发机关"],
        "护照": ["passport", "签发国"],
        "驾驶证": ["准驾车型", "机动车驾驶证"],
        "港澳通行证": ["往来港澳"],
        "户口": ["户号", "户主"],
    ]

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
        ("身份证", "id card"), ("驾照", "driver license document"),
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
