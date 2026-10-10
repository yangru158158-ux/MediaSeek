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

        // 3) 去载体后缀
        let suffixes = ["的图片", "的照片", "的相片", "的截图", "的影片", "的图",
                        "图片", "照片", "相片", "截图", "影片", "图"]
        for s in suffixes where t.hasSuffix(s) {
            t = String(t.dropLast(s.count)); break
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
        guard !matches.isEmpty else { return nil }
        return matches.sorted { $0.pos < $1.pos }.map { $0.en }.joined(separator: " ")
    }

    /// 泛化标签词:长查询(≥3 字)不带这些词去匹配标签——
    /// 它们什么照片都能沾上,是「一个词灌出全库」的元凶
    static let genericTokens: Set<String> = [
        "people", "person", "adult", "human", "document", "screenshot",
        "text", "structure", "outdoor", "machine", "clothing", "plant",
    ]

    /// 英文同义扩展:标签是 woman、查询是 girl 时精确匹配会漏,必须互相打通
    static let synonyms: [String: [String]] = [
        "person": ["people", "human", "adult", "man", "woman"],
        "people": ["person", "adult", "crowd"],
        "man": ["male", "person", "people"], "men": ["man", "people"],
        "woman": ["girl", "female", "person", "people"], "women": ["woman", "people"],
        "girl": ["woman", "female", "person", "people"], "lady": ["woman", "female"],
        "female": ["woman", "girl", "person", "people"],
        "boy": ["man", "male", "person", "people"], "male": ["man", "boy", "person"],
        "baby": ["infant"], "child": ["kid", "baby"], "kid": ["child"],
        "cat": ["kitten"], "kitten": ["cat"],
        "dog": ["puppy"], "puppy": ["dog"],
        "car": ["vehicle", "automobile"],
        "screen": ["monitor", "display"], "monitor": ["screen", "display"],
        "food": ["meal", "dish"], "flower": ["blossom"],
        "sea": ["ocean"], "ocean": ["sea"], "beach": ["sea", "shore"],
        "document": ["paperwork", "text"], "text": ["document"],
        "portrait": ["face", "person"], "face": ["portrait"],
        "beautiful": ["pretty", "attractive"],
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
        ("网页", "website screenshot"), ("聊天记录", "chat screenshot"),
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
        // 颜色(视觉词典兜底;主通道是颜色筛选器)
        ("蓝色", "blue"), ("红色", "red"), ("绿色", "green"), ("黄色", "yellow"),
        ("黑色", "black"), ("白色", "white"), ("粉色", "pink"), ("紫色", "purple"),
        ("灰色", "gray"),
    ]
}
