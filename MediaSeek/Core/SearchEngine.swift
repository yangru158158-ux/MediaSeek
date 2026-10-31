import Foundation
import Photos
import CoreLocation

enum SearchScope: String, CaseIterable, Identifiable {
    case all = "全部"
    case photo = "照片"
    case video = "视频"
    case file = "文件"
    var id: String { rawValue }
}

enum SearchTarget {
    case asset(PHAsset)        // 按 mediaType 区分照片/视频
    case file(ImportedFile)
}

struct DisplayHit: Identifiable {
    let id: String
    let refKey: String
    let target: SearchTarget
    let kind: ItemKind
    let title: String
    let score: Float
    let date: Date?
    let color: String?
}

/// 检索:查询同时嵌入 SigLIP 空间(查图/查视频)与 EmbeddingGemma 空间
/// (查文件、文本块、照片标签),合并去重后按相似度排序。
/// 一次搜索的完整产出:全部命中 + 文字命中集合(供 全部/文字/语义 三维度切换)
/// + 已应用的时间/地点标签(展示用)
struct SearchOutcome {
    let hits: [DisplayHit]
    let textIDs: Set<String>
    let timeLabel: String?
    let placeLabel: String?
    /// 批96:AND 契约显式化——零命中词被 combine 静默丢弃时的点名提示
    /// (「『XX』全网无命中,已从同时匹配中忽略」);nil=无。带默认值,
    /// 既有构造点零改动
    var note: String? = nil
    static let empty = SearchOutcome(hits: [], textIDs: [], timeLabel: nil, placeLabel: nil)
}

final class SearchEngine {
    private let store: VectorStore
    private let models: ModelManager
    private let photo: PhotoLibraryService
    private let imports: ImportLibrary

    /// 空结果诊断:本次查询在全库的最高原始语义分(百分数展示用);
    /// 有结果或语义通道未参与时为 nil
    private(set) var lastSemanticTopScore: Float?

    init(store: VectorStore, models: ModelManager, photo: PhotoLibraryService, imports: ImportLibrary) {
        self.store = store
        self.models = models
        self.photo = photo
        self.imports = imports
    }

    /// 查询规格:连接词(and/or/无)× 通道(文字/语义/自动)可自由组合
    /// 例:「or 文字:发票 收据」「and 语义:猫 狗」「文字:身份证」「and:2026 屏幕」
    struct QuerySpec {
        enum Connector { case and, or, none }
        enum Channel { case any, text, semantic, native }
        let connector: Connector
        let channel: Channel
        let body: String
        var isExplicit: Bool { channel != .any || connector != .none }
        /// 批94:.native(原生语义:)与 .semantic 同享排他待遇:关 OCR 免占座、
        /// 一档地板 0.25、textCovered 按「无」计——A/B 实验唯一变量=向量
        var semanticOnly: Bool { (channel == .semantic || channel == .native) && connector == .none }
    }

    static func parseQuery(_ raw: String) -> QuerySpec {
        // 全角符号归一(第十八批):中文输入法打 and：/文字：/全角空格是常态。
        // 此前 and:/or: 只认半角冒号——全角版本整个前缀失效,「and：」三个
        // 字被当进查询词去做语义检索,结果自然是垃圾;全角空格也不分词
        var s = raw.replacingOccurrences(of: "：", with: ":")
            .replacingOccurrences(of: "\u{3000}", with: " ")
        var connector = QuerySpec.Connector.none
        let lower = s.lowercased()
        // and:/or:(冒号,40 版写法)与 and /or:(空格)都支持
        if lower.hasPrefix("and:") || lower.hasPrefix("and ") {
            connector = .and
            s = String(s.dropFirst(4))
        } else if lower.hasPrefix("or:") || lower.hasPrefix("or ") {
            connector = .or
            s = String(s.dropFirst(3))
        }
        s = s.trimmingCharacters(in: CharacterSet(charactersIn: " :"))

        var channel = QuerySpec.Channel.any
        if s.hasPrefix("文字:") || s.hasPrefix("文字:") {
            channel = .text
            s = String(s.dropFirst(3)).trimmingCharacters(in: CharacterSet(charactersIn: " :"))
        } else if s.hasPrefix("原生语义:") {
            // 第九十四批:原生语义:=SigLIP 原生能力受控实验前缀。通道待遇
            // 与 语义: 完全同构(semanticOnly 含 .native),唯一区别=向量:
            // 官方模板壳+原文零转换(不词典/不桥/不平均/不繁简/不 core 清洗/
            // 不时间解析),构造见 QueryUnderstanding.nativeSiglipQuery。
            // 判在 语义: 前:两前缀互不为对方前缀,先后无行为差,仅为可读
            channel = .native
            s = String(s.dropFirst(5)).trimmingCharacters(in: CharacterSet(charactersIn: " :"))
        } else if s.hasPrefix("语义:") || s.hasPrefix("语义:") {
            channel = .semantic
            s = String(s.dropFirst(3)).trimmingCharacters(in: CharacterSet(charactersIn: " :"))
        }
        // 「文字:」未写连接词时默认 and(与旧版行为一致)
        if channel == .text && connector == .none { connector = .and }
        return QuerySpec(connector: connector, channel: channel, body: s)
    }

    /// 「不相关」黑名单:该查询词下被用户标记的结果不再出现(term 归一化为小写去空白)
    func blockRef(term: String, refKey: String) throws {
        try store.blockRef(term: term.lowercased().trimmingCharacters(in: .whitespaces), refKey: refKey)
    }
    func unblockRef(term: String, refKey: String) throws {
        try store.unblockRef(term: term.lowercased().trimmingCharacters(in: .whitespaces), refKey: refKey)
    }
    func blockedRefs(term: String) -> Set<String> {
        (try? store.blockedRefs(term: term.lowercased().trimmingCharacters(in: .whitespaces))) ?? []
    }
    /// 库内标签词典(标签,出现次数降序)——标签通道可精确命中的全部英文词
    func labelVocabCounts() -> [(label: String, count: Int)] {
        (try? store.labelVocabCounts()) ?? []
    }

    /// OCR 命中片段(标题展示用):截取命中词前后各 12 字
    static func ocrSnippet(_ text: String, _ term: String) -> String {
        guard let r = text.range(of: term) else { return String(text.prefix(30)) }
        let s = text.index(r.lowerBound, offsetBy: -12, limitedBy: text.startIndex) ?? text.startIndex
        let e = text.index(r.upperBound, offsetBy: 12, limitedBy: text.endIndex) ?? text.endIndex
        return String(text[s..<e]).replacingOccurrences(of: "\n", with: " ")
    }

    static func containsDigit(_ t: String) -> Bool {
        t.unicodeScalars.contains { $0.value >= 48 && $0.value <= 57 }
    }

    static func splitTerms(_ query: String) -> [String] {
        query.components(separatedBy: CharacterSet(charactersIn: " ,,、,，；/"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

// topK 4000 ≈ 全库:SQLite 向量检索本就全表扫描,加大 LIMIT 几乎零成本
    func search(_ rawQuery: String, scope: SearchScope, topK: Int = 4000,
                within: Set<String>? = nil) async throws -> SearchOutcome {
        let raw = rawQuery.trimmingCharacters(in: .whitespaces)
        guard !raw.isEmpty else { return .empty }
        let spec = Self.parseQuery(raw)
        // 批94:原生语义: 零转换从源头守起——core() 的指令/否定/载体清洗
        // 也是变换(「没有猫的照片」会被截成空),原生前缀直取 spec.body
        var query = spec.channel == .native ? spec.body : QueryUnderstanding.core(spec.body)
        guard !query.isEmpty else { return .empty }

        let siglip = await MainActor.run { models.siglip }
        lastSemanticTopScore = nil

        // 自然语言时间/地点:普通查询与结果内都参与(第七十四批起 语义: 前缀
        // 也参与时间解析;地点仍仅普通查询——语义: 若挂 GPS 过滤,无定位的
        // 语义命中会被整批滤掉,裸搜侧靠「地点浏览并集」规避而该分支只对
        // .any 开放)。时间词命中即剥离——结果内同样生效:从当前基数集里按
        // 拍摄/创建时间筛(与第一次搜索同构);地点「在XX」失败原样保留。
        var dateRange: QueryDateRange?
        var placeRegion: CLCircularRegion?
        var placeLabel: String?
        // 第七十四批:时间解析从「仅裸搜」扩到「裸搜 + 语义: 前缀」——
        // 语义: 排他下数字/年份词没有任何通道能消费(数字不走视觉,OCR 被
        // 排他),「语义:2026 杨如」实测与直接搜杨如无异(498 全标签照片)。
        // 剥离成时间过滤器后剩余词继续语义排他管线,resolve 按拍摄时间过滤。
        // 文字:/and:/or: 维持不解析:文字前缀的本义就是逐词精确匹配,
        // and: 的契约是逐词命中交集——年份在这两处是合法的内容词
        if spec.channel == .any || spec.channel == .semantic, spec.connector == .none {
            let t = TimeQueryParser.parse(query)
            if let r = t.range {
                dateRange = r
                query = t.remaining.trimmingCharacters(in: .whitespaces)
            }
            if spec.channel == .any, within == nil {
                if !query.isEmpty, let cand = PlaceQueryParser.extract(query) {
                    if let region = await PlaceGeocoder.region(for: cand.name) {
                        placeRegion = region
                        placeLabel = cand.name
                        query = cand.remaining.trimmingCharacters(in: .whitespaces)
                    }
                }
                // 裸地名(整个查询就是「南京」「马尔代夫」):城市名无视觉特征,
                // 语义通道认不出;挂 GPS 地点过滤,后面与地点浏览(全量拍摄记录)取并集。
                // 地理编码失败则原地降级为普通内容搜索,不误伤
                if placeRegion == nil, !query.isEmpty,
                   let p = QueryUnderstanding.barePlace(query),
                   let region = await PlaceGeocoder.region(for: p, radius: 50_000) {
                    placeRegion = region
                    placeLabel = p
                }
                if query.isEmpty, dateRange == nil, placeRegion == nil {
                    query = QueryUnderstanding.core(spec.body)   // 时间词没能构成区间,回退原查询
                }
            } else if query.isEmpty, dateRange == nil {
                query = QueryUnderstanding.core(spec.body)   // 结果内:没有成区间的时间词,回退原查询
            }
        }

        // 批103:抽取路径(在X+内容词)改交集——用户教义「地点+内容=在地拍摄∩
        // 内容」(批97/99 确立)取代旧加法并集。254 装机实锤「在杭州 猫」=248:
        // 旧加法把全网猫(语义220)灌进结果,杭州拍摄反而被淹没。旧设计注释
        // (挂 GPS 过滤会杀掉无定位截图)由交集基数只含「GPS 在域 ∪ 用户
        // 命名该地」化解:无定位又无命名的照片本就不属于「在杭州拍的」。
        // 实现:先收基数集合,内容搜索全局跑完按基数过滤(规避 SQLite IN
        // 参数上限,排序继承内容通道全局名次)
        if let placeRegion, let pl = placeLabel, !query.isEmpty,
           spec.channel == .any, spec.connector == .none, within == nil {
            var base = Set<String>()
            photo.fetchAllAssets().enumerateObjects { asset, _, _ in
                if scope == .file { return }   // 文件无 GPS
                if scope == .photo, asset.mediaType != .image { return }
                if scope == .video, asset.mediaType != .video { return }
                guard Self.assetPasses(asset, dateRange: dateRange, place: placeRegion) else { return }
                base.insert(asset.localIdentifier)
            }
            if scope == .all || scope == .photo {
                for vt in QueryUnderstanding.scriptVariants(of: pl) {
                    for r in ((try? store.searchUserTags(query: vt)) ?? []) { base.insert(r) }
                }
            }
            guard !base.isEmpty else {
                return SearchOutcome(hits: [], textIDs: [],
                                     timeLabel: dateRange?.label, placeLabel: pl)   // 该地无拍摄:诚实空
            }
            let main = try await runBareSearch(query: query, spec: spec, scope: scope, topK: topK,
                                               siglip: siglip, dateRange: dateRange,
                                               placeRegion: nil, placeLabel: nil, among: nil)
            let filtered = main.hits.filter { base.contains($0.refKey) }
            let textIDs = Set(filtered
                .map { "\($0.kind.rawValue)-\($0.refKey)" }
                .filter { main.textIDs.contains($0) })
            return SearchOutcome(hits: filtered, textIDs: textIDs,
                                 timeLabel: dateRange?.label, placeLabel: pl)
        }

        // 纯时间/地点浏览:无内容词,按拍摄时间倒序列出该时段的照片/视频/文件
        if query.isEmpty, dateRange != nil || placeRegion != nil {
            var hits: [SearchHit] = []
            photo.fetchAllAssets().enumerateObjects { asset, _, _ in
                guard hits.count < topK else { return }
                guard Self.assetPasses(asset, dateRange: dateRange, place: placeRegion) else { return }
                hits.append(SearchHit(kind: asset.mediaType == .video ? .videoFrame : .photo,
                                      refKey: asset.localIdentifier, frameIndex: 0,
                                      space: "browse", title: nil, date: asset.creationDate,
                                      score: 1.0, color: nil))
            }
            // 批99全局化:纯地点浏览同样并入用户命名标签(用户证据全局适用);
            // 批102:补 scope 守卫(验收 medium:userTag 是照片侧数据,视频
            // 页签不得混入);本分支末尾 resolve 的 dateRange 同批补透传
            if let pl = placeLabel, scope == .all || scope == .photo {
                var seenB = Set(hits.map(\.refKey))
                for h in placeUserTagHits(pl) where !seenB.contains(h.refKey) {
                    seenB.insert(h.refKey)
                    hits.append(h)
                }
            }
            if let allFiles = try? imports.allFiles() {
                for f in allFiles where hits.count < topK {
                    if fileMatchesFilters(f, dateRange: dateRange) {
                        hits.append(SearchHit(kind: .file, refKey: f.id, frameIndex: 0,
                                              space: "browse", title: f.name,
                                              date: f.addedAt, score: 1.0, color: nil))
                    }
                }
            }
            // 批102:纯浏览 resolve 补 dateRange(GPS 侧已 assetPasses 预滤,
            // 此处为 userTag 侧的照片真实日期裁决补闸)
            return SearchOutcome(hits: resolve(hits, dateRange: dateRange), textIDs: [],
                                 timeLabel: dateRange?.label, placeLabel: placeLabel)
        }

        // 第四十一批:此分支只接管裸搜——前缀查询(and:/or:/文字:)在结果内
        // 也要落到下方 searchScoped 保持 AND/排他语义,与全库搜索一致;此前
        // 被拦进主管线,文字: 在结果内多词退化成并集。第七十一批:文件范围
        // 的整批接管撤销(searchScoped 守卫已删,文件页签结果内与全库同规)
        if let within, !within.isEmpty,
           spec.channel == .any && spec.connector == .none {
            // 结果内纯时间词(如「2025年」):基数集合本身按拍摄/创建时间筛+倒序,
            // 与第一次搜索的纯时间浏览同构,只是范围缩进当前结果集
            if query.isEmpty, let dateRange {
                var hits: [SearchHit] = []
                if scope != .file {
                    let assets = PHAsset.fetchAssets(withLocalIdentifiers: Array(within), options: nil)
                    assets.enumerateObjects { asset, _, _ in
                        if scope == .photo, asset.mediaType != .image { return }
                        if scope == .video, asset.mediaType != .video { return }
                        guard Self.assetPasses(asset, dateRange: dateRange, place: nil) else { return }
                        hits.append(SearchHit(kind: asset.mediaType == .video ? .videoFrame : .photo,
                                              refKey: asset.localIdentifier, frameIndex: 0,
                                              space: "browse", title: nil, date: asset.creationDate,
                                              score: 1.0, color: nil))
                    }
                }
                if scope == .all || scope == .file, let allFiles = try? imports.allFiles() {
                    for f in allFiles where within.contains(f.id) {
                        if fileMatchesFilters(f, dateRange: dateRange) {
                            hits.append(SearchHit(kind: .file, refKey: f.id, frameIndex: 0,
                                                  space: "browse", title: f.name,
                                                  date: f.addedAt, score: 1.0, color: nil))
                        }
                    }
                }
                hits.sort { ($0.date ?? .distantPast) > ($1.date ?? .distantPast) }
                return SearchOutcome(hits: resolve(hits), textIDs: [],
                                     timeLabel: dateRange.label, placeLabel: nil)
            }
            // 结果内 = 同一条主管线 + 基数集合约束:三维度/排名/标题与第一次搜索
            // 完全一致(旧 searchWithin 是特设核验流程,通道与主流程不同步,
            // 反复出病:门槛悬空、漏标签通道,已删);时间词由 runMainSearch 内部
            // 按 dateRange 过滤(时间+内容词组合,如「2025年 海滩」)
            return try await runBareSearch(query: query, spec: spec, scope: scope, topK: topK,
                                           siglip: siglip, dateRange: dateRange, placeRegion: nil,
                                           placeLabel: nil, among: within)
        }
        if spec.channel == .text || spec.connector != .none {
            let scoped = try await searchScoped(spec: spec, scope: scope,
                                                topK: topK, siglip: siglip, within: within)
            return SearchOutcome(hits: scoped.hits, textIDs: scoped.textIDs,
                                 timeLabel: nil, placeLabel: nil,
                                 note: Self.andNote(scoped.ignored))
        }
        return try await runBareSearch(query: query, spec: spec, scope: scope, topK: topK,
                                       siglip: siglip, dateRange: dateRange,
                                       placeRegion: placeRegion, placeLabel: placeLabel,
                                       among: within)
    }

    /// 第四十一批:裸搜执行器——多词(有分隔符)=每词同时命中(AND)。
    /// 此前裸搜多词逐词取并集(第二十一批「或」语义),「宝宝 老公」190 个
    /// 结果几乎全是只含「宝宝」的;用户定规(2026-10-21):有分隔符=同时
    /// 完全匹配。AND 复用 searchScoped 的 and: 路径(每词 OCR∪语义∪标签
    /// 命中再交集,已验收实现,不另造一套防漂移);单词保持原主管线;or: 前缀
    /// 仍可显式取并集。第七十一批:文件范围的多词豁免随 searchScoped 守卫
    /// 删除而撤销——文件页签多词同样走逐词 AND(文字命中交集),单词/文件
    /// 范围照旧主管线(OCR 通道已放行)。
    /// 已知收窄(有意):全部范围多词改道后不跑 Gemma 文件语义通道——旧裸搜
    /// 对文件是整句模糊匹配,与逐词 AND 精确语义不兼容,文件改靠 OCR 原文
    /// (PDF/CSV/XLSX/TXT 内容均已入库 ocr_text)双词精确命中。
    /// 空态分数提示(lastSemanticTopScore)仅 语义: 查询展示,而 语义: 已被
    /// 下方 channel 校验挡回主管线(主管线自己写该字段),本函数无需写
    /// 批99全局化(用户定则):用户产生的证据适用全局——地址搜索任何路径,
    /// 用户命名符合地址的标签与 GPS 拍摄合并入结果。简繁并查、refKey 去重;
    /// 照片侧数据;title 标注证据来源便于核对
    private func placeUserTagHits(_ place: String) -> [SearchHit] {
        var out: [SearchHit] = []
        var seen = Set<String>()
        for vt in QueryUnderstanding.scriptVariants(of: place) {
            for r in ((try? store.searchUserTags(query: vt)) ?? []) where !seen.contains(r) {
                seen.insert(r)
                out.append(SearchHit(kind: .photo, refKey: r, frameIndex: 0,
                                     space: "userTag", title: "用户命名「\(place)」",
                                     date: nil, score: 0.9, color: nil))
            }
        }
        return out
    }

    /// 批96:AND 零命中提示文案——按契约返回空(用户拍板改②):零命中词使
    /// 「同时包含」无法成立,诚实出 0 并点名;出路(去词重搜/结果内缩小)
    /// 由空态页补充。「无命中」不带范围限定词:全局=全网零命中,结果内=
    /// 基数集内零命中,两态皆准
    private static func andNote(_ ignored: [String]) -> String? {
        ignored.isEmpty ? nil : "『\(ignored.joined(separator: "』『"))』无命中——同时匹配无法成立"
    }

    /// 批96:纯 ASCII 多词判定——英文的空格是构词符不是意图分隔符,
    /// "a man riding a horse" 被逐词 AND 肢解剥夺自然语言表达(用户点名:
    /// 交集能力结果内搜已完美承载,裸搜该说人话);中文/中英混合带分隔符
    /// =用户显式意图,照旧 AND(用户拍板:「猫 cat」保持 AND)
    static func isAsciiPhrase(_ query: String) -> Bool {
        let ts = splitTerms(query)
        return ts.count >= 2 && ts.allSatisfy { $0.unicodeScalars.allSatisfy(\.isASCII) }
    }

    private func runBareSearch(query: String, spec: QuerySpec, scope: SearchScope,
                               topK: Int, siglip: SigLIPEmbedder?, dateRange: QueryDateRange?,
                               placeRegion: CLCircularRegion?, placeLabel: String?,
                               among: Set<String>? = nil) async throws -> SearchOutcome {
        // 只对裸搜(channel == .any)生效:语义: 前缀必须保持排他(只要语义
        // 命中),不得被改道 and: 并入 OCR——验收 medium 实锤(此前只看
        // scope 和词数,语义:多词被误改道且未在提交说明声明)
        // 批96:纯 ASCII 多词(英文短语/描述句)不改道——整串概念走主管线
        if spec.channel == .any, Self.splitTerms(query).count >= 2,
           !Self.isAsciiPhrase(query) {
            let andSpec = QuerySpec(connector: .and, channel: .any, body: query)
            let scoped = try await searchScoped(spec: andSpec, scope: scope, topK: topK,
                                                siglip: siglip, within: among,
                                                dateRange: dateRange)
            return SearchOutcome(hits: scoped.hits, textIDs: scoped.textIDs,
                                 timeLabel: dateRange?.label, placeLabel: placeLabel,
                                 note: Self.andNote(scoped.ignored))
        }
        return try await runMainSearch(query: query, spec: spec, semanticOnly: spec.semanticOnly,
                                       scope: scope, topK: topK, siglip: siglip,
                                       dateRange: dateRange, placeRegion: placeRegion,
                                       placeLabel: placeLabel, among: among)
    }

    /// 门槛阶梯的降档判据(第六十二批):「捞到 0 条才降档」→「捞不足 30 条
    /// 就降档」。61 批按模型缩放把各档地板整体下移后,分数分布上沿恰好越过
    /// 新一档地板的查询,会在高一档捞到顶部一小撮(非空即停),把更宽的低档
    /// 彻底堵死——so400m 搜 woman 46→8(211 用户实测)。低档结果天然是高档
    /// 的超集(同查询同 limit 只降地板),替换即合并;放宽进来的命中仍受
    /// 性别对比过滤与低置信尾线收口,不会灌屏。30=结果页首屏量级
    private static let tierMinResults = 30

    /// 第六十八批:第三档底网(全局 0.08×gate)从「薄了就换网」收紧为反空兜底。
    /// 旧逻辑命中不足 tierMinResults 就整网换到噪声带底网,而 so400m 分数压缩后
    /// 真命中与噪声挤在 15% 带宽内、尾线切不开——「智搜」187 张/「id」15 张/
    /// 「动物」尾部文档全是底网杂音(2026-10-23 用户实测三案)。允许降档,满足其一:
    ///   ① 结果内(among 非空)或证件类词(ocrAliases 在册):小集合/人像面的
    ///      弱分真命中仍靠这档浮顶(证件照对「身份证」只有 0.05-0.08,批 43 先例)
    ///   ② 纯反空:上两档零命中,且该词文字通道也零命中(财报类生词先例保留);
    ///      文字已有命中时底网只剩噪声,不启用
    /// 语义: 前缀(semanticOnly)排他跳过文字通道,文字覆盖按「无」计
    static func floorTierPermitted(among: Set<String>?, semanticOnly: Bool,
                                   upperTierEmpty: Bool, aliasExempt: Bool,
                                   textCovered: Bool) -> Bool {
        if among != nil { return true }
        if aliasExempt { return true }
        if semanticOnly { return upperTierEmpty }
        return upperTierEmpty && !textCovered
    }

    private func runMainSearch(query: String, spec: QuerySpec, semanticOnly: Bool,
                               scope: SearchScope, topK: Int, siglip: SigLIPEmbedder?,
                               dateRange: QueryDateRange?, placeRegion: CLCircularRegion?,
                               placeLabel: String?, among: Set<String>? = nil) async throws -> SearchOutcome {
        // among = 结果内基数(第一次搜索的结果集合):非 nil 时所有通道只在此集合内
        // 取命中,管线其余部分(通道权重/RRF 排名/三维度切分)与全局搜索完全一致
        // 多词查询:「2026 电脑屏幕」→ 每个词独立走自己的通道(数字词天然
        // 只会命中文字;中文词 视觉/标签/OCR 三路都试),「或」语义:照片
        // 满足任一条件(像/含字)即显示(第二十一批恢复逐词,见 OCR 层注释)
        let terms = Self.splitTerms(query)
        // 第七十二批:语义/标签侧全按简体键运转(词典英文模板/专名闸/证件豁免/
        // 标签 token),繁体词原样进会绕过整套校准——「駕駛證」裸嵌入捞 254 个
        // 文档噪声(221 装机实测)。视觉词与 concept 统一繁→简归一;文字通道
        // 保留原词,转换词以追加变体并入(scriptVariants,原词+转换词双向无损)
        let simTerms = terms.map { QueryUnderstanding.simplified($0) }
        let visualTerms = simTerms.filter { !Self.containsDigit($0) }
        let concept = visualTerms.isEmpty ? query : visualTerms.joined(separator: " ")
        // 第六十二批:专名词(华研机器人——连续生字符≥2,判据见
        // QueryUnderstanding.isProperNoun)整词只走文字通道。第六十批管住了
        // 碎片捐赠,但整词原文兜底仍把词交给 SigLIP 多语言,行业相关截图的
        // 噪声带分数(≥第三档地板)整批捞成「语义」命中(211 实测 96 张)。
        // 多词「或」语义:剩下的词照常走视觉;全专名时整条视觉通道跳过
        // (纯数字词同此:SigLIP 嵌数字串只有噪声,211 前靠门槛挡、如今直接不跑)
        let semanticTerms = visualTerms.filter { term in
            if QueryUnderstanding.isProperNoun(term) { return false }
            // 第六十四批人名闸(与 semanticHits 同规):零命中词命中用户标签
            // (归组命名的名字)→ 人名。SigLIP 分不开同段人名(「管雯」「程玉婷」
            // 的嵌入都是「中文女性人名」互相同义),语义对人名只有噪声(213 用户
            // 实测:结果内搜管雯混入大量程玉婷);名字→脸的绑定只在标签与 OCR
            // 文字里,这两条通道照常。判据与标签通道同一调用(searchUserTags
            // 子串匹配)——标签通道能命中的词,语义对它只会添乱。
            // 判据边界(验收拦下后收窄):①只对真零命中(isDictionaryUnknown)
            // 生效——english==nil 还涵盖盖不满的修饰词(红裙子),批 62 的
            // 原文兜底承诺不破;②性别词豁免(genderOpposite 命中即非人名)——
            // 裸英文性别词(woman/man)也是零命中,不豁免会在标签含其子串的
            // 边角废掉性别深潜
            if QueryUnderstanding.isDictionaryUnknown(term),
               QueryUnderstanding.genderOpposite(of: term) == nil,
               !((try? store.searchUserTags(query: term)) ?? []).isEmpty {
                return false
            }
            return true
        }
        // 第六十三批性别词深潜:查询含性别词(woman/女人/美女…)时照片档直接
        // 下探第三档噪声地板并豁免尾线。一档收网对品类词是召回天花板(212 实测
        // 搜 woman=42 全卡一档,~700 张真命中躺在低分段没人捞);「是不是该性别」
        // 交给对比边距(≥0.010)裁决。低分段可行性已双模型实测(run 39898534426
        // dist:远景女照 margin +0.019~+0.030/男照 -0.006~-0.030/合影 +0.005~+0.006
        // 被 0.010 挡住;中性物体历史标定 ±0.004 同样过不了线)
        let genderDive = simTerms.contains { QueryUnderstanding.genderOpposite(of: $0) != nil }

        let imageKinds: [ItemKind] = scope == .all ? [.photo, .videoFrame]
            : (scope == .photo ? [.photo] : (scope == .video ? [.videoFrame] : []))
        // Gemma 语义只负责文件名/文件内容——照片标签的余弦虚高且区分度低
        // (「猫」vs「people, adult」和 vs「cat」差不了几分),会顶掉真命中
        let fileKinds: [ItemKind] = scope == .all || scope == .file ? [.file, .fileChunk] : []

        var channels: [(weight: Double, hits: [SearchHit])] = []
        let gemma = await MainActor.run { models.gemma }

        // 标签精确通道(优先):词典英译+同义词与 Vision 英文标签 token 级比对,零幻觉。
        // 文字: 前缀 = 排他文字通道,标签/视觉/用户标签一概不参与(用户 2026-10-11:
        // 「结果内搜指定了文字,视频还出现就不对」——111 重构后主管线漏了这层排他)
        var labelHits: [SearchHit] = []
        let tokens = QueryUnderstanding.labelTokens(concept)
        // 有标签实据的视频 refKey(查询词∩帧标签非空)——视觉通道放行视频帧的唯一证据
        var videoLabelRefs: Set<String> = []
        if !tokens.isEmpty, spec.channel != .text {
            // 照片标签行(视频范围不得参与,否则照片漏进视频结果)
            if imageKinds.contains(.photo), let rows = try? store.allPhotoLabels() {
                var scored: [(refKey: String, matches: Int, title: String)] = []
                for row in rows where among?.contains(row.refKey) ?? true {
                    let labelTokens = Set(row.title.lowercased()
                        .components(separatedBy: CharacterSet(charactersIn: ", ")))
                    let m = tokens.intersection(labelTokens).count
                    if m > 0 { scored.append((row.refKey, m, row.title)) }
                }
                labelHits = Array(scored.sorted { $0.matches > $1.matches }.prefix(topK).map {
                    SearchHit(kind: .photo, refKey: $0.refKey, frameIndex: 0,
                              space: "label", title: $0.title, date: nil,
                              score: Float($0.matches), color: nil)
                })
            }
            // 视频帧标签:视频侧最可靠的召回(帧画面打 Vision 标签,与照片同款)。
            // SigLIP 图文分数区分度不足(马尔代夫旅行视频 vs 抖音下载视频分不开),
            // 标签零幻觉;结果卡带标签文本可解释
            if scope == .all || scope == .video, let rows = try? store.allVideoFrameLabels() {
                var scored: [(refKey: String, matches: Int, title: String)] = []
                for row in rows where among?.contains(row.refKey) ?? true {
                    let labelTokens = Set(row.title.lowercased()
                        .components(separatedBy: CharacterSet(charactersIn: ", ")))
                    let m = tokens.intersection(labelTokens).count
                    if m > 0 {
                        scored.append((row.refKey, m, row.title))
                        videoLabelRefs.insert(row.refKey)
                    }
                }
                labelHits += scored.sorted { $0.matches > $1.matches }.prefix(topK).map {
                    SearchHit(kind: .videoFrameLabel, refKey: $0.refKey, frameIndex: 0,
                              space: "label", title: $0.title, date: nil,
                              score: Float($0.matches), color: nil)
                }
            }
        }
        if !labelHits.isEmpty { channels.append((1.6, labelHits)) }

        // 视觉通道:SigLIP2 原生多语言(分词实验已证),中文原文直接查;
        // 不再用语义路由——泛化标签(document/screenshot)会被几乎任何词
        // 以 ≥0.60 命中,造成整库截图照灌进结果(「医学出生证明」事故)
        // 显式「语义:」查询门槛放宽到 0.14(用户点名要语义召回)
        // 文字: 前缀排他:视觉通道(照片/视频帧)不参与
        // 批94:原生语义: 不看 semanticTerms(其逐词过滤闸对整句向量无意义,
        // 专名/纯数字句也会空表直接跳过通道)——判 .native 即跑
        if !imageKinds.isEmpty, spec.channel != .text,
           (!semanticTerms.isEmpty || spec.channel == .native), let siglip {
            // so400m 校准(第六十一批):SigLIP 绝对门槛按激活模型缩放
            // (so400m 命中带 = base × 0.87,CI 双模型分布实测,详见
            // ModelManager.siglipGateScale 注释)
            let gate = await MainActor.run { models.siglipGateScale }
            // 批95:so400m 下裸搜/语义: 的中文词走原生模板壳(矩阵三族 3:0
            // 实证);标准版 false=词典英译原配。批107 起单词径与 语义:多词
            // 交集径同用此原料(批106 先扩到平均径,批107 平均退役改交集)
            let nativeZh = await MainActor.run { models.preferNativeZhQuery }
            // 批93:语义: 多词=组合概念——各词模板向量「归一化→平均→再归一化」
            // (CLIP 论文 80 模板 ensemble 同款配方)。整串拼嵌的落点由文本塔
            // 注意力决定、不可控(词袋效应:license certificate document 拼串
            // 落 document 带 992 灌屏,换组词可能落荒地);平均向量确定落在
            // 各概念向量的中间地带。裸搜多词不经过此处(批41 改道逐词 AND),
            // 单词两路等价——行为变化仅 语义: 多词,正是要消灭的洪水
            // 批107:平均向量退役——「中间地带」恰是错位所在:用户意图是
            // AND 交集,熔合概念在异质词对上落荒地(259 装机 猫沙发 594 张
            // 房产广告灌屏),改逐词宽召回交集(见下方 isectWords 分支)
            // 批105①:摘要行路径标签与嵌入分支同源赋值——批104 三态标签把
            // 批96 英文整串与批93 平均分支错标成「模板/条件原生」,标准版英文
            // 短语日志失真;标签只能来自实际执行的分支,平均失败降级也自动正确
            let q: [Float]
            var embedPath: String
            // 批109:非 nil=语义:多词交集路径(阶梯/视频帧/空态三处按它改道);
            // sieved=高精筛子生效(免尾线——收口已由筛子完成)
            var isectWords: [String]? = nil
            var isectSieved = false
            if spec.channel == .native {
                // 批94:原生向量——唯一规则=官方模板壳+spec.body 原文(未过
                // core 清洗/时间剥离,也不经词典/桥/平均三道变换);多词语句
                // 整句直进,SigLIP 原生判断首次可受控测量
                q = try siglip.embedQuery(QueryUnderstanding.nativeSiglipQuery(spec.body))
                embedPath = "原生前缀"
            } else if spec.channel == .any, Self.isAsciiPhrase(query) {
                // 批96:英文整串概念——构词空格不当意图分隔,整句(经 core/时间
                // 剥离后的 query)套原生模板壳,与 原生语义: 同款构造
                q = try siglip.embedQuery(QueryUnderstanding.nativeSiglipQuery(query))
                embedPath = "英文整串"
            } else if semanticOnly, semanticTerms.count >= 2 {
                // 批107:语义:多词 = AND 意图(「猫 沙发」=有猫且有沙发)。
                // 批93/批106 的平均向量是概念熔合:共现词对(阳光 海滩)碰巧
                // 熔出风景照,异质词对(猫 沙发)熔出「毛绒室内质感」——259
                // 装机实测原生「沙发」邻域的房产广告截图 best 抬到 0.151、
                // 尾线 0.128 切完仍 594 张灌屏(258 模板系 84 张)。改逐词宽
                // 召回取交集(与裸搜 AND 线批103 同规),综合分=min(各词分)。
                // q 置空:本路径不消费它,下游三处消费点均按 isectWords 跳过
                isectWords = semanticTerms
                q = []
                embedPath = "交集"   // 批110:是否有牌约束在下方交集块内细化(标签∩语义)
            } else {
                let joined = semanticTerms.joined(separator: " ")
                q = try siglip.embedQuery(Self.siglipQuery(joined, nativeZh: nativeZh))
                // 条件原生的生效线与 siglipQuery 内部一致:仅非 ASCII 串吃原生壳
                embedPath = (nativeZh && !joined.unicodeScalars.allSatisfy(\.isASCII))
                    ? "条件原生" : "模板"
            }
            // 照片与视频帧分开走各自的门槛阶梯:分数分布不同,且照片行为已验收冻结
            var hits: [SearchHit] = []
            if imageKinds.contains(.photo) {
                if let words = isectWords {
                    // 批111(用户设计定稿):高精并集筛子 × SigLIP 交集。
                    // 目的与手段(用户教义):目的是「符合现实的结果」,高精
                    // 筛子与尾线都只是手段——筛子有效则免尾线(收口的活已被
                    // 筛子干了),无筛子才保留尾线。批109 逐词分层的死因:
                    // OCR 把无映射词抬成「高精词」后与他集求交(文字出现≠
                    // 画面内容);批110 乘性约束的死因:每词都要牌命中,
                    // 「美女与卢沟桥合影」只有 OCR 一侧证据会被乘性归零。
                    // 定稿公式:筛子 S=∪(每词的 照片标签∪帧标签∪用户命名,
                    // 跨词并集,至少一词有画面事实证据即过筛);基底 B=∩(每词
                    // SigLIP 宽召回 0.070);结果=B∩S(S 非空,免尾线)或
                    // B+尾线(S 空,批108 行为)。语义相关性由 B 保证(每词都
                    // 过 SigLIP 宽召回),事实证据由 S 保证(至少一词),双高
                    // 原巧合(无任何证据)被筛掉。
                    // 批112:OCR 剔出筛子(263 装机实锤)——OCR 命中的截图类
                    // 与 SigLIP 高原交集天然重叠(写着「沙发」的广告截图恰是
                    // SigLIP 高原分照片),文字证据≠画面证据,筛子只收画面
                    // 事实(标签牌/帧标签/用户命名);免尾线维持(筛后量级
                    // 个位数-十几,远不到尾线治的灌屏量级)
                    var sieve: Set<String> = []
                    for t in words {
                        let lt = QueryUnderstanding.labelTokens(t)
                        if !lt.isEmpty {
                            if imageKinds.contains(.photo), let rows = try? store.allPhotoLabels() {
                                for row in rows where among?.contains(row.refKey) ?? true {
                                    let toks = Set(row.title.lowercased()
                                        .components(separatedBy: CharacterSet(charactersIn: ", ")))
                                    if !lt.intersection(toks).isEmpty { sieve.insert(row.refKey) }
                                }
                            }
                            if scope == .all || scope == .video, let rows = try? store.allVideoFrameLabels() {
                                for row in rows where among?.contains(row.refKey) ?? true {
                                    let toks = Set(row.title.lowercased()
                                        .components(separatedBy: CharacterSet(charactersIn: ", ")))
                                    if !lt.intersection(toks).isEmpty { sieve.insert(row.refKey) }
                                }
                            }
                        }
                        for vt in QueryUnderstanding.scriptVariants(of: t) {
                            for r in ((try? store.searchUserTags(query: vt)) ?? [])
                                where among?.contains(r) ?? true { sieve.insert(r) }
                        }
                    }
                    // 基底:全部词 SigLIP 宽召回交集(批107 机制)
                    var bKeys: Set<String>? = nil
                    var firstWordTop: Float = 0
                    for t in words {
                        guard let qw = try? siglip.embedQuery(
                            Self.siglipQuery(t, nativeZh: nativeZh)) else { continue }
                        let exempt = QueryUnderstanding.floorExemptWords.contains(t)
                            || QueryUnderstanding.chineseCounterpart(for: t) != nil
                        let wh = Self.dedupByRef(try store.search(
                            space: siglip.space, kinds: [.photo], query: qw,
                            limit: topK, minScore: exempt ? 0.04 : 0.08 * gate, among: among))
                        if bKeys == nil { firstWordTop = wh.first?.score ?? 0 }
                        let keys = Set(wh.map(\.refKey))
                        bKeys = bKeys.map { $0.intersection(keys) } ?? keys
                    }
                    var result = bKeys ?? []
                    if !sieve.isEmpty {
                        result.formIntersection(sieve)
                        isectSieved = true
                    }
                    // SigLIP min 分:结果集上逐词打分(排序+摘要读数;视频
                    // 多帧同 refKey 保最高,与性别对比同规)
                    var sigMin: [String: Float] = [:]
                    if !result.isEmpty {
                        for t in words {
                            guard let qw = try? siglip.embedQuery(
                                Self.siglipQuery(t, nativeZh: nativeZh)) else { continue }
                            let scored = Self.dedupByRef((try? store.search(
                                space: siglip.space, kinds: [.photo, .videoFrame], query: qw,
                                limit: max(topK, result.count * 8), minScore: -1, among: result)) ?? [])
                            for s in scored {
                                sigMin[s.refKey] = min(sigMin[s.refKey] ?? .greatestFiniteMagnitude, s.score)
                            }
                        }
                    }
                    hits = result.map { ref -> SearchHit in
                        SearchHit(kind: .photo, refKey: ref, frameIndex: 0,
                                  space: "isect", title: nil, date: nil,
                                  score: sigMin[ref] ?? 0, color: nil)
                    }.sorted { $0.score > $1.score }
                    // 路径标签:有筛子=高精筛∩语义(免尾线);无=纯语义交集(尾线照切)
                    embedPath = isectSieved ? "高精筛∩语义" : "交集"
                    // 交集空:空态提示用首词最高分(下方空态守卫段按 isectWords
                    // 跳过;不引行号——本段插入会推移下游行号,批107 验收教训)
                    if hits.isEmpty { lastSemanticTopScore = firstWordTop }
                } else if genderDive {
                    // 深潜:第三档地板直取(阶梯与薄判都不适用——品类词要的是
                    // 全量召回,质量裁决交给下方性别对比线)
                    let floor: Float = (among != nil ? 0.03 : 0.08) * gate
                    hits = Self.dedupByRef(try store.search(
                        space: siglip.space, kinds: [.photo], query: q, limit: topK,
                        minScore: floor, among: among))
                } else {
                let floor: Float = (semanticOnly ? 0.25 : 0.30) * gate
                hits = Self.dedupByRef(try store.search(
                    space: siglip.space, kinds: [.photo], query: q, limit: topK,
                    minScore: floor, among: among))
                // 弱信号兜底:品牌词/抽象词(「微信」「聊天」)对截图的视觉相似度
                // 天然低于常规门槛,一刀切会整词清零(2026-10-10 用户实测三个词全空);
                // 降档条件=薄判(不足 tierMinResults,第六十二批)
                if hits.count < Self.tierMinResults {
                    hits = Self.dedupByRef(try store.search(
                        space: siglip.space, kinds: [.photo], query: q, limit: topK,
                        minScore: 0.18 * gate, among: among))
                }
                if hits.count < Self.tierMinResults {
                    // 第三档分两种:全局地板 0.08(旧注释称「高于无关带上沿」,
                    // so400m 实测噪声带已越过此线——智搜 187/id 15/动物尾部三案,
                    // 故不再常规启用);结果内 0.03——用户点名的小集合里相对排名
                    // 优先,弱分真匹配(证件照对「身份证」只有 0.05-0.08)也要
                    // 浮顶,噪声由低置信尾线 0.95 收。
                    // 第六十八批:全局模式改反空兜底——文字通道已有命中或上两档
                    // 已有命中时不再换底网(判据与三案实测见 floorTierPermitted)
                    // 第七十二批:探测与文字通道同口径(原词+繁→简转换词)——
                    // 文字通道命中过任一变体即算 covered,否则繁体词探测简体库恒空
                    let probeTerms = (terms.isEmpty ? [query] : terms)
                        .flatMap { QueryUnderstanding.scriptVariants(of: $0) }
                    // 批82:底网豁免白名单(floorExemptWords)+英文证件桥;
                    // 批90:驾驶证/驾照入白名单
                    let exempt = probeTerms.contains {
                        QueryUnderstanding.floorExemptWords.contains($0)
                            || QueryUnderstanding.chineseCounterpart(for: $0) != nil
                    }
                    // 批91:证件类豁免词的底网固定、不随模型缩放——so400m
                    // 分布整体下移(批61),0.08×0.87=0.070 捞不到真卡(244 装机:
                    // licence/驾驶证 语义全空,用户确认在 so400m);批43 先例本是
                    // 0.05-0.08,白名单窄+模板已收窄,固定值不放杂音。
                    // 批92:0.05→0.04——ANE 查询嵌入非确定(±0.005 抖动,见
                    // SigLIPEmbedder.queryVecCache),卡分数贴线会话间横跳,
                    // 0.04 留出抖动余量;反空兜底维持 0.08×gate
                    let floor: Float
                    if among != nil {
                        floor = 0.03
                    } else if exempt {
                        floor = 0.04
                    } else {
                        floor = 0.08 * gate
                    }
                    if Self.floorTierPermitted(
                        among: among, semanticOnly: semanticOnly, upperTierEmpty: hits.isEmpty,
                        aliasExempt: exempt,
                        textCovered: !semanticOnly && probeTerms.contains {
                            !((try? store.searchOCR(query: $0)) ?? []).isEmpty }) {
                        hits = Self.dedupByRef(try store.search(
                            space: siglip.space, kinds: [.photo], query: q, limit: topK,
                            minScore: floor, among: among))
                    }
                }
                }
            }
            if imageKinds.contains(.videoFrame), isectWords == nil {
                // 批107:交集路径不含视频帧——AND 即精度,多词交集下单帧分无
                // 稳定语义;打了对应标签的视频仍由标签通道(零幻觉)独立承载
                // 真尺度校准:纯 SigLIP 的视频帧区分度不足——古庙绿崖 0.11 vs 真海滩 0.13-0.14,
                // 相对标定尾线必然放进噪声(2026-10-12 马尔代夫实测)。按 113 确立的通道优先级
                // (标签零幻觉 > OCR > SigLIP 兜底),视频帧入场必须有实据:
                // 帧标签与查询词有交集,或分数 ≥0.15 的强信号
                var vHits = Self.dedupByRef(try store.search(
                    space: siglip.space, kinds: [.videoFrame], query: q, limit: topK,
                    minScore: 0.08 * gate, among: among))
                vHits = vHits.filter { videoLabelRefs.contains($0.refKey) || $0.score >= 0.15 * gate }
                hits.append(contentsOf: vHits)
                hits = Self.dedupByRef(hits)
            }
            // 兜底后仍空 = 全库最高分都不到 0.08,记下最高原始分供空态提示
            // (批107:交集路径的空态分已在交集计算内置,此处须跳过——q 为空数组)
            if hits.isEmpty, isectWords == nil {
                lastSemanticTopScore = (try? store.search(
                    space: siglip.space, kinds: imageKinds, query: q, limit: 1,
                    minScore: -1, among: among))?.first?.score
            }
            // 第五十批:性别词对比过滤——SigLIP 绝对分区分男女弱(0.08 地板
            // 男女同放行,用户实测搜 woman 杨如程玉婷混列),相对差才可靠
            // (第五十五批真实照片+文献式提示词实测:女照 +0.030~+0.049 /
            // 男照 -0.056~-0.023 / 中性 -0.002~+0.004)。对比串用文献标准
            // 「a photo of a woman/man」(CVPR 2025 口径),正反向对称打分,
            // 只留正向分领先 ≥0.010 的命中(女性最低值3倍余量,中性最高值
            // 2倍间隔);合影两性
            // 分数相抵(差≈0)也被滤掉——性别词搜索要的是单一性别主导的照片。
            // 第五十一/五十二/五十三批收紧:①逐词探测(多词整串查单词表不命中);
            // ②反查 among=本组命中集 + limit×8 余量(limit 截的是行数不是键数,
            // 视频多帧行数可超 limit;×8 覆盖后仍截断者按 0 放行=与旧行为同,不劣化);
            // ③正反向都用性别单词嵌入打分——hits.score 是整串 concept 的分,
            // 多词查询下与单词反向分做差不符 0.012 的单词-单词标定口径,边距一律对称
            if let oppTerm = terms.first(where: { QueryUnderstanding.genderOpposite(of: $0) != nil }),
               let pair = QueryUnderstanding.genderContrast(of: oppTerm),
               let qFwd = try? siglip.embedQuery(pair.fwd),
               let qOpp = try? siglip.embedQuery(pair.opp) {
                let cands = Set(hits.map(\.refKey))
                func scores(_ query: [Float]) -> [String: Float] {
                    Dictionary(((try? store.search(space: siglip.space, kinds: imageKinds, query: query,
                                  limit: max(topK, cands.count * 8), minScore: 0, among: cands)) ?? [])
                        .map { ($0.refKey, $0.score) },
                    uniquingKeysWith: { max($0, $1) })   // 视频多帧同 refKey,保最高分
                }
                let fwdScores = scores(qFwd)
                let oppScores = scores(qOpp)
                hits = Self.dedupByRef(hits.filter {
                    ((fwdScores[$0.refKey] ?? $0.score) - (oppScores[$0.refKey] ?? 0)) >= 0.010
                })
            }
            // 相对尾部截断:远弱于头部的长尾(如「人」里混入的屏幕翻拍照)直接砍掉。
            // 低置信收紧(真尺度 2026-10-12):整组最佳 <0.13 = 没有真匹配,列表全是
            // 人像/风景基线噪声(同一张海景人像 马尔代夫0.09/护照0.12/身份证0.12),
            // 此时尾线收到 best×0.95 只留贴近最佳的一小撮,不灌整屏
            let best = hits.first?.score ?? 0
            // 性别词深潜豁免尾线(第六十三批):尾线以最佳分为基准,最佳 0.35 时
            // 85% 线=0.298,下探捞回的低分段真命中会被一刀全砍,深潜白做;
            // 性别词的噪声闸=对比边距(0.010),尾线职责由它接管
            // 批111:筛子有效免尾线(目的与手段:收口的活已由筛子干了);
            // 无筛子才尾线;性别词豁免与全局同规(深潜的裁决在对比边距)
            let tailFactor: Float = genderDive || isectSieved ? 0 : (best >= 0.13 * gate ? 0.85 : 0.95)
            // 批104⑤:语义查询摘要行——分数带从「推断」变「读数」:边界项抖动
            // (255 装机 原生语义:阳光 9→10 的 ±0.005 疑云)、档位取舍、尾线
            // 切割,此后全部凭 diag.log 原始分定案,不再靠猜
            // 批108:增记「尾线后」计数——命中=尾线切割前(诊断阶梯),尾线后=
            // 切割后幸存(≈界面语义通道基数,259 装机曾把命中数误当界面数,
            // 两口径并列后不再靠推算)
            let survivors = best > 0 && tailFactor > 0
                ? hits.filter { $0.score >= best * tailFactor } : hits
            IndexingCoordinator.DiagLog.shared.event(
                String(format: "语义[%@] 路径=%@ 命中=%d best=%.3f min=%.3f 尾线后=%d",
                       query,
                       embedPath,
                       hits.count, best, hits.map(\.score).min() ?? 0, survivors.count))
            channels.append((1.0, survivors))
        }

        // 用户标签通道:中文子串精确匹配(文字: 排他,不参与)
        if spec.channel != .text, imageKinds.contains(.photo) {
            // 第二十一批:多词按「或」逐词命中标签——整串「杨如 男」永远匹配
            // 不到标签「杨如」,他本人的照片在裸搜多词下一张出不来
            var tagRefs = Set<String>()
            // 第七十二批:标签按繁→简变体并查(归组命名是用户手敲简体)
            for t in (terms.isEmpty ? [concept] : terms)
                .flatMap({ QueryUnderstanding.scriptVariants(of: $0) }) {
                for r in ((try? store.searchUserTags(query: t)) ?? [])
                where among?.contains(r) ?? true { tagRefs.insert(r) }
            }
            if !tagRefs.isEmpty {
                channels.append((1.6, tagRefs.map {
                    SearchHit(kind: .photo, refKey: $0, frameIndex: 0,
                              space: "userTag", title: nil, date: nil, score: 1.0, color: nil)
                }))
            }
        }

        // 文件语义通道(文字: 排他,不参与):Gemma 文本嵌入查文件名/内容。
        // 绝对门槛不可靠:嵌入基线高且随查询/语言漂移(「马尔代夫」能把
        // 汇丰结单和 rar 灌进来,0.30/0.50 都拦不住)。改自校准离群检测:
        // 低门槛取候选后,只有「最高分 ≥0.40 且显著高于中位数(+0.10)」
        // 才认为查询真的指向文件,否则整通道静默。
        // 文件标签页(范围=文件)不做离群门槛,保留完整语义浏览。
        if !fileKinds.isEmpty, spec.channel != .text, let gemma {
            let q = try gemma.embedQuery(concept)
            var fileHits = Self.dedupByRef(try store.search(
                space: gemma.space, kinds: fileKinds, query: q, limit: topK,
                minScore: scope == .file ? 0.25 : 0.20, among: among))
            // 结果内不做离群门槛(基数是用户点名的小集合),只留相对截断
            if among == nil {
                // 全局:自校准离群检测(基线高且随查询漂移,绝对门槛不可靠)
                if fileHits.count > 1 {
                    let scores = fileHits.map(\.score).sorted()
                    let median = scores[scores.count / 2]
                    let best = scores.last ?? 0
                    if best < 0.40 || best - median < 0.10 { fileHits = [] }
                }
            } else if let best = fileHits.first?.score, best < 0.40 {
                // 结果内(2026-10-12 食物实测):此前跳过质量线,CSV/结单灌进食物结果。
                // Gemma 文本-文本基线高,弱相关全挤在 0.3 档;结果内同样要过 0.40 强线
                fileHits = []
            }
            let topScore = fileHits.first?.score ?? 0
            if topScore > 0 { fileHits = fileHits.filter { $0.score >= topScore * 0.75 } }
            if fileHits.count > 60 { fileHits = Array(fileHits.prefix(60)) }
            channels.append((1.0, fileHits))
        }

        // Reciprocal Rank Fusion:贡献 = 权重/(60+名次);OCR 通道权重 8,必压语义通道
        let K = 60.0
        var fused: [String: (hit: SearchHit, score: Double)] = [:]
        func add(_ hit: SearchHit, weight: Double, rank: Int) {
            let s = weight / (K + Double(rank + 1))
            if let cur = fused[hit.refKey] {
                // 第六十二批文字先行:已被 OCR 文字命中的照片归文字通道,
                // 语义/标签/用户标签通道不再叠加它(so400m 门槛下移后,含
                // 查询词文字的截图——搜索结果截图之类——同时被视觉通道捞到,
                // 旧规则把双命中条目的归属翻成语义、融合分又带着 OCR 权重 8
                // 的底子,语义栏前排全是文字截图,真照片被压到尾部;211 用户
                // 实测搜 woman 语义栏前两名是 100%/97% 的文字截图)。
                // OCR 块先于通道循环 add,cur 只可能是先落座的 OCR 条目
                if cur.hit.space == "ocr" { return }
                let winner = hit.score > cur.hit.score ? hit : cur.hit
                fused[hit.refKey] = (winner, cur.score + s)
            } else {
                fused[hit.refKey] = (hit, s)
            }
        }

        // OCR 文字精确通道:「或」语义——每个词独立命中"含该文字"的照片,
        // 与视觉通道取并集;标题附命中片段,方便核对为什么命中(小字/页脚也会算)
        // 语义: 前缀模式下跳过(只要语义命中)
        // 第四十三批:视频范围放行 + 命中按范围过滤(与 searchScoped 的
        // d04d122 同规)。此前通道只在 全部/照片 范围跑:视频范围裸搜单词
        // 恒空——转写文字在 ocr_text 里,文字: 前缀与多词 AND 走 searchScoped
        // 都能命中,唯独裸搜单词不通(2026-10-21 用户实测 老公:视频范围 0,
        // 老公 宝宝/文字:老公 均正常);且照片范围不过滤命中身份,视频的
        // OCR 命中会泄漏进照片结果
        // 第七十一批:文件范围放行(与批 43 视频范围同病漏修)。文件内容
        // (PDF/CSV/XLSX/TXT)早已入库 ocr_text,全部范围下本通道一直能精确
        // 命中文件,唯独文件页签被排除在外——精确文字命中被 Gemma 模糊语义
        // 单通道顶替,两字词(美團)埋在长文里相似度过不了离群闸 → 恒 0
        // (220 用户实测:全部范围搜得到月结单,文件范围 0)。放行后
        // scopeAllowedKinds(.file)=[.file] 天然只出文件,照片/视频不会泄
        if !semanticOnly {
            let allowedKinds = Self.scopeAllowedKinds(scope)
            let classifier = OCRRefClassifier(store: store)
            var seen = Set<String>()
            var rank = 0
            // 第二十一批:恢复「每词独立命中」(build 32 的「或」语义在 CI 88
            // 重构时窄化成了「只有含数字的词走 OCR」)——纯中文多词(杨如 男)
            // 整串带空格当子串查,健康码印的是「杨如，男」永远 miss,用户实测
            // 裸搜搜不出、and: 反而搜得出。逐词并集,与 and:/or: 通道同构
            // 批96:ASCII 整串概念按整串查——逐词并集会让 "a"/"the" 单字母
            // 子串灌整库文字命中(批41 前的老病从这扇门复活)
            let ocrTerms = (spec.channel == .any && Self.isAsciiPhrase(query))
                ? [query] : (terms.isEmpty ? [query] : terms)
            for t in ocrTerms where !t.isEmpty {
                // 证件类 OCR 别名:人像面不印证件名(身份证正面只有「公民身份号码」等字段),
                // 只搜证件名会漏掉全部正面照——补字段级别名并集保召回
                // 第六十九批英文证件桥(纯加法):表内英文词追加中文对应词+别名,
                // 原词匹配不动——大陆证件只印中文,不追加则英文查询漏全部大陆证件
                // 第七十二批:追加繁→简转换词及其别名(原词优先,排序不变)——
                // 「駕駛證」同时命中大陆简体驾驶证 OCR,「美團」原词照常命中繁体月结单
                let conv = QueryUnderstanding.simplified(t)
                // 拆步构造(不用三目嵌链式 +,类型推断会爆炸——CI 实测 688 行
                // unable to type-check in reasonable time)
                var variants: [String] = [t] + (QueryUnderstanding.ocrAliases[t] ?? [])
                    + QueryUnderstanding.bridgedVariants(for: t)
                if conv != t {
                    variants.append(conv)
                    variants.append(contentsOf: QueryUnderstanding.ocrAliases[conv] ?? [])
                }
                for term in variants {
                let rows = (try? store.searchOCR(query: term)) ?? []
                let kinds = classifier.classify(rows.map(\.refKey))
                for ref in rows {
                    guard let k = kinds[ref.refKey], allowedKinds.contains(k),
                          !seen.contains(ref.refKey),
                          among?.contains(ref.refKey) ?? true else { continue }
                    seen.insert(ref.refKey)
                    add(SearchHit(kind: k, refKey: ref.refKey, frameIndex: 0,
                                  space: "ocr", title: "含「\(term)」:\(Self.ocrSnippet(ref.text, term))",
                                  date: nil, score: 1.0, color: nil),
                        weight: 8.0, rank: rank)
                    rank += 1
                }
                }

                // 二字滑窗已废除(第三十五批,用户点名「无分隔符请使用完整匹配,
                // 否则无法做到精确搜索」):滑窗把碎片≥2 的照片全放进结果
                // (实测搜台胞证全名灌入 81 张,绝大多数是碎片巧合),精确性
                // 崩坏。现在无分隔长词只做整词完整子串匹配(上方 searchOCR);
                // 语序记反/说法不同的模糊召回改用 语义: 或拆多词空格
            }
        }

        for ch in channels {
            for (rank, hit) in ch.hits.enumerated() { add(hit, weight: ch.weight, rank: rank) }
        }
        guard !fused.isEmpty else { return .empty }

        // 显示分 = 相对融合分(第一名 100%)
        let maxScore = fused.values.map { $0.score }.max() ?? 1.0
        let merged: [SearchHit] = fused.values.sorted { $0.score > $1.score }.map {
            SearchHit(kind: $0.hit.kind, refKey: $0.hit.refKey, frameIndex: $0.hit.frameIndex,
                      space: $0.hit.space, title: $0.hit.title, date: $0.hit.date,
                      score: Float($0.score / maxScore), color: $0.hit.color)
        }
        return SearchOutcome(
            hits: resolve(merged, dateRange: dateRange, place: placeRegion),
            textIDs: Set(merged.filter { $0.space == "ocr" }.map { "\($0.kind.rawValue)-\($0.refKey)" }),
            timeLabel: dateRange?.label, placeLabel: placeLabel)
    }

    /// 显式语法流程:通道(文字/语义/自动)× 连接词(and/or)自由组合。
    /// 返回 hits + textIDs(文字维度命中集,第十五批:此前调用方写死空集,
    /// 「文字:」前缀的全部命中被三维度切换栏归进语义栏——用户实测
    /// 文字:地图 189 个全挂语义、文字 0)
    private func searchScoped(spec: QuerySpec, scope: SearchScope,
                              topK: Int, siglip: SigLIPEmbedder?, within: Set<String>? = nil,
                              dateRange: QueryDateRange? = nil)
        async throws -> (hits: [DisplayHit], textIDs: Set<String>, ignored: [String]) {
        // 第七十一批:删除「文件范围整体返回空」守卫(第三十九批为拦照片/视频
        // 而设,把文字通道一并拦死——文件页签 文字:/and: 恒空,与裸搜同病)。
        // 放行后各通道自筛:文字通道 scopeAllowedKinds 过滤只出文件;视觉通道
        // 在 semanticHits 内对文件范围早退(文件无 SigLIP 向量)
        let terms = Self.splitTerms(spec.body)
        guard !terms.isEmpty else { return ([], [], []) }
        // so400m 校准(第六十一批):SigLIP 门槛按激活模型缩放,透传给 semanticHits
        let gate = await MainActor.run { models.siglipGateScale }
        // 批102(C5):AND/or: 逐词线接原生化——253 装机实锤「马尔代夫 阳光/
        // 骑车」恒空:落空词(词典未收)裸嵌入弱带全灭。so400m 下逐词也走
        // 原生模板壳(与单词径批95 同源);标准版 false=原行为
        let nativeZh = await MainActor.run { models.preferNativeZhQuery }
        // 第五十批:归类器一次搜索共用一个实例——逐词各建一套会把 files/
        // videoFrame 全表查询与 PHAsset 批查放大数倍(多词×别名×常见词数千命中)
        let classifier = OCRRefClassifier(store: store)

        // 文字通道:每个词独立查 OCR,连接词决定 AND/OR。
        // and 模式标题列出全部命中词(交集里每张都含全部词,不能只显示第一个)
        // 结果内模式:只在这些照片的文字索引里查
        if spec.channel == .text {
            var perTerm: [[SearchHit]] = []
            for t in terms {
                // 第三十九批修:按范围过滤 OCR 命中——验收 medium 发现,视频
                // 范围下照片 OCR 泄漏进结果(ocrKind 只归类不过滤)
                // 第四十批:改用 scopeAllowedKinds 统一映射,消除并行维护的
                // 内联 switch(其 .file 分支在批 39-70 间因顶部 file 守卫永不可达;
                // 第七十一批守卫删除后 .file 分支可达,此处为路径史实记录)
                let rows = ocrUnionHits(t, scope: scope, within: within, topK: topK, classifier: classifier)
                    .filter { Self.scopeAllowedKinds(scope).contains($0.kind) }
                let connectorIsAnd = (spec.connector == .and)
                perTerm.append(rows.enumerated().map { rank, ref in
                    let base = within != nil
                        ? "结果内文字命中「\(t)」"
                        : (connectorIsAnd
                           ? "文字全含:\(terms.joined(separator: " ")) —「\(t)」\(Self.ocrSnippet(ref.text, t))"
                           : "文字命中「\(t)」:\(Self.ocrSnippet(ref.text, t))")
                    return SearchHit(kind: ref.kind, refKey: ref.refKey, frameIndex: 0, space: "ocr",
                              title: base,
                              date: nil, score: Float(1.0 / Double(rank + 1)), color: nil)
                })
            }
            let combined = combine(perTerm, connector: spec.connector)
            let textIDs = Set(combined.filter { $0.space == "ocr" }
                .map { "\($0.kind.rawValue)-\($0.refKey)" })
            // 批96二改(用户拍板按契约返回空):有词零命中(全局=全网,结果内=基数集内)=任何照片不可能
            // 同时包含全部词,诚实出 0——与结果内「基数空绝不静默退化全局」
            // 同一哲学;原因与出路交空态页呈现,不再返回剩余词的交集
            let ignored = spec.connector == .and
                ? zip(terms, perTerm).filter { $0.1.isEmpty }.map(\.0) : []
            if !ignored.isEmpty { return ([], [], ignored) }
            return (resolve(combined, dateRange: dateRange), textIDs, ignored)
        }

        // 语义/自动通道:数字词→文字,其余词→视觉
        var perTerm: [[SearchHit]] = []
        for t in terms {
            // 第三十九批:自动通道与文字通道同规按范围过滤(验收 medium 发现:
            // and:/or: 前缀在视频范围下仍混入照片命中)
            let allowed = Self.scopeAllowedKinds(scope)
            if Self.containsDigit(t) {
                // 数字词也走 OCR 并集(带 kind 归类)再按范围过滤——此前写死
                // .photo,视频范围下数字词命中全是照片泄漏
                perTerm.append(ocrUnionHits(t, scope: scope, within: within, topK: topK, classifier: classifier)
                    .filter { allowed.contains($0.kind) }
                    .enumerated().map { rank, r in
                        SearchHit(kind: r.kind, refKey: r.refKey, frameIndex: 0, space: "ocr",
                                  title: "含「\(t)」文字", date: nil,
                                  score: Float(1.0 / Double(rank + 1)), color: nil)
                    })
            } else {
                // 批97:地点词=GPS 拍摄通道,真相优先(用户拍板方案B)。城市名
                // 无视觉特征——词典城市模板的邻居=全世界的城市感照片(249 装机
                // 实测 or:上海 杭州 语义栏 110 个多为异地城市照、相对分贴 14%
                // 弱带;而杭州 GPS 拍摄 318 张大量漏在结果外——精度与召回双病)。
                // 根因:地点解析圈死在 connector==.none 的闸内,前缀查询结构性
                // 绕过 GPS。文字: 前缀不注入(纯文字语义不变);and: 地名+
                // 内容词由此获得「在地拍摄∩内容」能力(and:上海 猫=在上海
                // 拍的猫)。裸搜整串单地名走既有地点浏览并集(路径不变)
                var hits: [SearchHit] = []
                if scope != .file, let p = QueryUnderstanding.barePlace(t),
                   let region = await PlaceGeocoder.region(for: p, radius: 50_000) {
                    photo.fetchAllAssets().enumerateObjects { asset, _, _ in
                        guard hits.count < topK else { return }
                        if scope == .photo, asset.mediaType != .image { return }
                        if scope == .video, asset.mediaType != .video { return }
                        guard Self.assetPasses(asset, dateRange: dateRange, place: region) else { return }
                        guard within?.contains(asset.localIdentifier) ?? true else { return }
                        hits.append(SearchHit(
                            kind: asset.mediaType == .video ? .videoFrame : .photo,
                            refKey: asset.localIdentifier, frameIndex: 0,
                            space: "browse", title: "\(p)拍摄",
                            date: asset.creationDate, score: 1.0, color: nil))
                    }
                    // 批99:地点词四通道定稿(用户三轮纠正):GPS 拍摄(语义侧)+
                    // OCR 文字命中(文字侧,杭州上海既是地址也是文字)+用户标签
                    // (用户亲手命名某组/某照「上海」=精度极高,绝不能出局);
                    // 出局者仅机器侧的语义视觉填充与 Vision 帧标签——前者是
                    // 「城市名无视觉特征」教义下的定义性噪声(250 装机误报源
                    // =福州/黄山城市感图片混入语义栏),后者是机器物体标签非
                    // 用户命名。区域零照片→GPS 侧空名单(OR=他词兜着,AND=
                    // 批96 契约空);地理编码失败→不进此分支照旧全填充
                    var seenPlace = Set(hits.map(\.refKey))
                    for (rank, r) in ocrUnionHits(t, scope: scope, within: within,
                                                  topK: topK, classifier: classifier)
                        .filter({ allowed.contains($0.kind) })
                        .enumerated() where !seenPlace.contains(r.refKey) {
                        seenPlace.insert(r.refKey)
                        hits.append(SearchHit(kind: r.kind, refKey: r.refKey, frameIndex: 0,
                                              space: "ocr", title: "含「\(t)」文字", date: nil,
                                              score: Float(1.0 / Double(rank + 1)), color: nil))
                    }
                    if allowed.contains(.photo) {
                        // 用户标签(批99三改;helper 同款构造,within 基数照守)
                        for h in placeUserTagHits(p)
                            where !seenPlace.contains(h.refKey)
                                && (within?.contains(h.refKey) ?? true) {
                            seenPlace.insert(h.refKey)
                            hits.append(h)
                        }
                    }
                    perTerm.append(hits)
                    continue
                }
                // 非地点词(能走到这里 hits 必为空):语义主座直接落座
                hits.append(contentsOf: semanticHits(t, siglip: siglip, topK: topK,
                                                      within: within, scope: scope, gateScale: gate,
                                                      nativeZh: nativeZh,
                                                      andRecall: spec.connector == .and))
                if let within { hits = hits.filter { within.contains($0.refKey) } }
                var seen = Set(hits.map(\.refKey))
                // 「像或含字」同构(第十六批):裸搜模式里每个词 = OCR∪视觉任一
                // 命中即算,and:/or: 自动通道此前只送语义——证件/文件类词语义
                // 天生弱(0.05-0.08 过不了地板),and:身份证 正面 恒空。每词并入
                // OCR(含证件别名)命中再按连接词交并;同照片双命中时语义条目
                // 优先(带真实分),OCR 只补缺口。(批99:地点词已在上方 continue
                // 出局,以下填充仅服务非地点词)
                // 第六十五批:视频帧标签通道(与主管线 runMainSearch 同构)——
                // 帧标签与查询词有交集的视频直接命中,零幻觉、不依赖分数。
                // 此前结果内路径没有这条通道,视频只剩 semanticHits 一跳:打了
                // 对应标签但帧分低于各档门槛的形态(结果内 0.20/0.10/0.03×gate、
                // 全局 and:/or: 0.25/0.15/0.08×gate)出不来。「无对应标签」形态
                // 本通道同样无解(命中前提就是标签∩),根治属挂起的相对证据
                // 重设计。标签条目只补缺口(语义条目带真实分排前,与 OCR 补
                // 缺口同规);照片范围不并入(视频侧数据,与下方人名标签同规)
                if allowed.contains(.videoFrame), let rows = try? store.allVideoFrameLabels() {
                    let tokens = QueryUnderstanding.labelTokens(t)
                    if !tokens.isEmpty {
                        // 与主管线同通道完全同构(验收拦下初版无界返回):命中数
                        // 降序 + topK 截断——全局 and:/or: 路径 within=nil 时
                        // 命中全库标签行也必须有界;分数=命中数(与主管线一致)
                        var scored: [(refKey: String, title: String, matches: Int)] = []
                        for row in rows where (within?.contains(row.refKey) ?? true) {
                            let rowTokens = Set(row.title.lowercased()
                                .components(separatedBy: CharacterSet(charactersIn: ", ")))
                            let m = tokens.intersection(rowTokens).count
                            if m > 0 { scored.append((row.refKey, row.title, m)) }
                        }
                        for row in scored.sorted(by: { $0.matches > $1.matches }).prefix(topK)
                        where !seen.contains(row.refKey) {
                            seen.insert(row.refKey)
                            hits.append(SearchHit(kind: .videoFrameLabel, refKey: row.refKey,
                                                  frameIndex: 0, space: "label", title: row.title,
                                                  date: nil, score: Float(row.matches), color: nil))
                        }
                    }
                }
                for (rank, r) in ocrUnionHits(t, scope: scope, within: within, topK: topK, classifier: classifier)
                    .filter({ allowed.contains($0.kind) })
                    .enumerated() where !seen.contains(r.refKey) {
                    seen.insert(r.refKey)
                    hits.append(SearchHit(kind: r.kind, refKey: r.refKey, frameIndex: 0,
                                          space: "ocr", title: "含「\(t)」文字", date: nil,
                                          score: Float(1.0 / Double(rank + 1)), color: nil))
                }
                // 人名标签并集(第十七批):裸搜「杨如」靠用户标签(人物归组命名)
                // 命中他的照片,and:/or: 前缀此前丢掉这条通道——中文人名不在
                // 词典、语义通道对人名无效、核酸截图之外的照片又不含这两个字,
                // and:杨如 男 只能出文字截图,他本人的照片一张进不了交集
                // 第三十九批:人名标签是照片侧数据,视频范围下不再并入
                if allowed.contains(.photo) {
                    // 第七十二批:标签繁→简变体并查(与主管线用户标签通道同规)
                    for vt in QueryUnderstanding.scriptVariants(of: t) {
                        for r in ((try? store.searchUserTags(query: vt)) ?? [])
                            .filter({ within?.contains($0) ?? true }) where !seen.contains(r) {
                            seen.insert(r)
                            hits.append(SearchHit(kind: .photo, refKey: r, frameIndex: 0,
                                                  space: "userTag", title: nil, date: nil,
                                                  score: 0.9, color: nil))
                        }
                    }
                }
                perTerm.append(hits)
            }
        }
        var combined = combine(perTerm, connector: spec.connector)
        // 批96二改(用户拍板按契约返回空,与上方 .text 分支同规)
        let ignored = spec.connector == .and
            ? zip(terms, perTerm).filter { $0.1.isEmpty }.map(\.0) : []
        if !ignored.isEmpty { return ([], [], ignored) }
        // AND 模式逐照片合成标题(第十七批):combine 去重只保留第一个词的命中
        // 条目,and:杨如 男 的卡片全部显示「含「杨如」文字」——第二个词的
        // 命中只字不提,用户看不出交集凭什么成立(照片明明同时含「男」)。
        // 逐照片列出每词的命中方式(文字/标签/语义)
        if spec.connector == .and, terms.count > 1 {
            var info: [String: [String]] = [:]
            for (t, list) in zip(terms, perTerm) {
                for h in list {
                    // 第六十五批:space=="label"(视频帧标签通道)标「帧标签」,
                    // 与 userTag 的「标签」区分——命中方式来源不可错标(验收拦下)
                    // 批97:space=="browse"(地点 GPS 通道)标「拍摄地」同规
                    let way = h.space == "ocr" ? "文字"
                        : (h.space == "browse" ? "拍摄地"
                           : (h.space == "userTag" ? "标签"
                              : (h.space == "label" ? "帧标签" : "语义")))
                    info[h.refKey, default: []].append("\(t)·\(way)")
                }
            }
            combined = combined.map { h in
                guard let parts = info[h.refKey], parts.count > 1 else { return h }
                return SearchHit(kind: h.kind, refKey: h.refKey, frameIndex: h.frameIndex,
                                 space: h.space,
                                 title: "同时命中:" + parts.joined(separator: " "),
                                 date: h.date, score: h.score, color: h.color)
            }
        }
        let textIDs = Set(combined.filter { $0.space == "ocr" }
            .map { "\($0.kind.rawValue)-\($0.refKey)" })
        return (resolve(combined, dateRange: dateRange), textIDs, ignored)
    }

    /// 范围允许的命中 kind(第三十九批):文字/自动通道统一按范围过滤,
    /// 防跨范围泄漏(视频范围混入照片命中)
    private static func scopeAllowedKinds(_ scope: SearchScope) -> Set<ItemKind> {
        switch scope {
        case .all: return [.photo, .videoFrame, .file]
        case .photo: return [.photo]
        case .video: return [.videoFrame]
        case .file: return [.file]
        }
    }

    /// OCR 命中并集(词本身+证件类字段别名,第十六批):与主流程 OCR 层同口径。
    /// 此前只有主流程做别名扩展,前缀路径(文字:身份证 / and:身份证)漏掉
    /// 正面照(正面不印「身份证」,只有「公民身份号码」等字段)。OCR 命中的
    /// ref 按 files/videoFrame 表归类身份;结果内模式只在基数集合里查
    private func ocrUnionHits(_ t: String, scope: SearchScope,
                              within: Set<String>?, topK: Int,
                              classifier: OCRRefClassifier)
        -> [(kind: ItemKind, refKey: String, text: String)] {
        var out: [(kind: ItemKind, refKey: String, text: String)] = []
        var seen = Set<String>()
        // 第六十九批:英文证件桥同规(主管线 OCR 通道注释)——纯加法
        // 第七十二批:繁→简转换词同规追加(与主管线一致,文字:/and:/or: 共用此处);
        // 拆步构造防类型推断爆炸(同主管线注释)
        let conv = QueryUnderstanding.simplified(t)
        var variantTerms: [String] = [t] + (QueryUnderstanding.ocrAliases[t] ?? [])
            + QueryUnderstanding.bridgedVariants(for: t)
        if conv != t {
            variantTerms.append(conv)
            variantTerms.append(contentsOf: QueryUnderstanding.ocrAliases[conv] ?? [])
        }
        for term in variantTerms {
            let rows: [(String, String)]
            if let among = within {
                rows = ((try? store.searchOCR(query: term, among: Array(among))) ?? [])
                    .map { ($0, "") }
            } else {
                rows = ((try? store.searchOCR(query: term)) ?? [])
                    .prefix(topK).map { ($0.refKey, $0.text) }
            }
            let kinds = classifier.classify(rows.map(\.0))
            for r in rows where !seen.contains(r.0) {
                seen.insert(r.0)
                out.append((kinds[r.0] ?? .photo, r.0, r.1))
            }
        }
        return out
    }

    /// OCR 命中身份归类器(第四十三批):文件表→.file;有帧向量行→.videoFrame;
    /// 含糊身份(既非文件也无帧行,如帧全被 isTooDark 跳过的纯语音视频)问
    /// PHAsset 媒体类型兜底。**一次搜索共用一个实例**:files/videoFrame 全表
    /// 查询与 PHAsset 批查都只按需做一次并逐 ref 缓存——验收 low 实锤逐词
    /// 变体各建一套会把扫描 IO 放大数倍(多词×别名×常见词数千命中)
    final class OCRRefClassifier {
        private let fileIds: Set<String>
        private let videoRefs: Set<String>
        private var cache: [String: ItemKind] = [:]
        init(store: VectorStore) {
            fileIds = (try? store.fileIds()) ?? []
            videoRefs = (try? store.refKeys(kinds: [.videoFrame])) ?? []
        }
        func classify(_ refs: [String]) -> [String: ItemKind] {
            var ambiguous: [String] = []
            for r in refs {
                if cache[r] != nil { continue }
                if fileIds.contains(r) { cache[r] = .file }
                else if videoRefs.contains(r) { cache[r] = .videoFrame }
                else { ambiguous.append(r) }
            }
            if !ambiguous.isEmpty {
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ambiguous, options: nil)
                fetch.enumerateObjects { a, _, _ in
                    self.cache[a.localIdentifier] = a.mediaType == .video ? .videoFrame : .photo
                }
                for r in ambiguous where self.cache[r] == nil { self.cache[r] = .photo }
            }
            var out: [String: ItemKind] = [:]
            for r in refs { out[r] = cache[r] ?? .photo }
            return out
        }
    }

    /// SigLIP2 官方零样本模板("This is a photo of X.")对短语类查询更友好;
    /// 词典命中的精选英文描述套模板,未命中(长句/自由文本)保持原文。
    /// ⚠️ 提示词集成(多模板平均)虽是通行做法,但会平移全库语义分数、
    /// 现有门槛按单模板分布标定——要上必须先在设备上 A/B 单独验证
    /// (批93/批106 的 averagedQuery 平均向量已被批107 交集路径替代删除,
    /// 259 装机实测熔合概念与 AND 意图错位;配方与教训见 git 历史 d7f520d)
    private static func siglipQuery(_ concept: String, nativeZh: Bool = false) -> String {
        if let en = QueryUnderstanding.english(for: concept) {
            // 批95:so400m 激活时中文词直用原文模板壳(原生路径,依据与边界
            // 见 ModelManager.preferNativeZhQuery);标准版仍走词典英译。
            // ⚠️ 复核修正:词典含少量 ASCII 键(pdf→document/qq),「词典命中
            // 必是中文」不成立——nativeZh 只对含非 ASCII 字符的词生效,
            // ASCII 键保持英译(document 泛词是文档截图的刻意选择)
            if nativeZh, !concept.unicodeScalars.allSatisfy(\.isASCII) {
                return QueryUnderstanding.nativeSiglipQuery(concept)
            }
            return "This is a photo of \(en)."
        }
        // 第六十九批英文证件桥:表内词(id/identity→身份证 族)套对应词的
        // 词典英文——身份证族统一 "id card" 模板,比裸 "identity"/"id" 的
        // 弱嵌入强(218 实测裸 id 语义 15 条全是噪声)
        if let zh = QueryUnderstanding.chineseCounterpart(for: concept),
           let en = QueryUnderstanding.english(for: zh) {
            return "This is a photo of \(en)."
        }
        // 第四十四批:裸英文单词/短语(词典没收的,如直接打 woman/man/beach)
        // 同样套官方零样本模板——短裸词分数系统性偏低,照片通道 0.30/0.18
        // 档几乎全灭(2026-10-21 用户实测 woman 仅 2 张;中文「女人」因词典
        // 命中带模板反而正常)。长句(>3 词)保持原文,与词典路径同规
        let isAscii = concept.unicodeScalars.allSatisfy(\.isASCII)
        let wordCount = concept.split(separator: " ").count
        if isAscii, wordCount >= 1, wordCount <= 3 {
            return "This is a photo of \(concept)."
        }
        // 批98+批102:落空的中文词(so400m 原生路径)套原生壳。批98 只给
        // 「的」短语开壳;253 装机实锤「阳光/骑车」(词典未收的普通词)裸搜
        // 与 AND 组合双死——落空裸嵌弱带全灭。扩到全部非 ASCII 落空词:
        // 能走到此处的词已经过专名闸(run≥2 华研机器人类)/人名闸/数字闸
        // 三道筛,剩下正是该救的普通词;财报专名类被闸挡在前面,噪声面
        // 可控,地板/尾线照旧兜底。标准版 nativeZh=false 走原裸嵌路径
        if nativeZh, !isAscii {
            return QueryUnderstanding.nativeSiglipQuery(concept)
        }
        return concept
    }

    /// 单词语义命中(SigLIP 视觉,门槛 0.14 + 0.08 弱信号兜底 + 相对尾部截断)。
    /// gateScale=激活模型的门槛缩放系数(so400m 0.87,第六十一批校准)
    private func semanticHits(_ rawTerm: String, siglip: SigLIPEmbedder?, topK: Int,
                              within: Set<String>? = nil,
                              scope: SearchScope = .all,
                              gateScale: Float = 1.0,
                              nativeZh: Bool = false,
                              andRecall: Bool = false) -> [SearchHit] {
        guard let siglip else { return [] }
        // 第七十二批:繁→简归一(在通道入口自守,调用方无须预处理)——内部
        // 专名闸/词典模板/证件豁免/性别逻辑全按简体键运转,繁体原串会绕过
        // 校准裸嵌入(駕駛證 254 文档噪声实测);幂等,简体词零变化
        let term = QueryUnderstanding.simplified(rawTerm)
        // 第七十一批:文件范围早退(与 searchScoped 守卫收窄配套)——文件
        // 只有 Gemma 空间向量,SigLIP 通道无文件数据;此前靠 searchScoped
        // 顶部守卫兜着,守卫删后这里必须自守(顺带免去空 kinds 的 SQL IN())
        if scope == .file { return [] }
        // 第六十二批:专名词整词跳过视觉(与主管线 semanticTerms 同规)——
        // 整词原文兜底=SigLIP 品类语义噪声带捞全库,只走文字通道
        if QueryUnderstanding.isProperNoun(term) { return [] }
        // 第六十四批人名闸:零命中(isDictionaryUnknown)+命中用户标签=人名,
        // 且非性别词 → 跳过视觉(标签/OCR 照常;依据与判据边界见主管线
        // semanticTerms 注释);无标签零命中词(财报类)不受影响,照走多语言
        if QueryUnderstanding.isDictionaryUnknown(term),
           QueryUnderstanding.genderOpposite(of: term) == nil,
           !((try? store.searchUserTags(query: term)) ?? []).isEmpty {
            return []
        }
        // 第六十三批性别词深潜(与主管线 genderDive 同规):第三档地板+豁免
        // 尾线,性别裁决交给下方对比边距(低分段实测依据见主管线注释)
        let genderDive = QueryUnderstanding.genderContrast(of: term) != nil
        guard let q = try? siglip.embedQuery(Self.siglipQuery(term, nativeZh: nativeZh)) else { return [] }
        // 第三十九批:按范围取 kinds(与主管线 imageKinds 同构,.file→[] 空)——
        // 此前全部写死 [.photo],视频范围下视频帧向量从不被查询
        let searchKinds: [ItemKind] = scope == .all ? [.photo, .videoFrame]
            : (scope == .photo ? [.photo] : (scope == .video ? [.videoFrame] : []))
        // 结果内核验:分数只在基数集合内算,门槛放宽、相对截断以集合内最佳为基准——
        // 否则全局排名的尾部截断会在取交集前把候选砍光(「行情图 ∩ 语义:电脑屏幕=0」事故:
        // 全库最高分是实拍屏幕,行情截图全被 85% 线砍掉,交集自然为空)
        if let within, !within.isEmpty {
            var hits: [SearchHit]
            if genderDive {
                // 深潜:结果内第三档地板直取(0.03),性别裁决交对比边距
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                    minScore: 0.03 * gateScale, among: within)) ?? [])
            } else {
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                minScore: 0.20 * gateScale, among: within)) ?? [])
            // 降档=薄判(tierMinResults,第六十二批,与主流程阶梯同规)
            if hits.count < Self.tierMinResults {
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                    minScore: 0.10 * gateScale, among: within)) ?? [])
            }
            if hits.count < Self.tierMinResults {
                // 结果内第三档:0.20/0.10 在命中带上悬空 → 结果内语义词恒空;
                // 地板 0.03(结果内=点名小集合,相对排名优先,与主流程照片第三档同规),
                // 尾部由 best×0.95 低置信尾线截断
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                    minScore: 0.03 * gateScale, among: within)) ?? [])
            }
            }
            let best = hits.first?.score ?? 0
            // 第五十批:性别词对比过滤(结果内同规,见 QueryUnderstanding.genderOpposites;
            // 第五十五批:对比串=文献标准形式,正反向对称打分,线 0.010 见标定注释)
            if let pair = QueryUnderstanding.genderContrast(of: term),
               let qFwd = try? siglip.embedQuery(pair.fwd),
               let qOpp = try? siglip.embedQuery(pair.opp) {
                let cands = Set(hits.map(\.refKey))
                func litScores(_ query: [Float]) -> [String: Float] {
                    Dictionary(((try? store.search(space: siglip.space, kinds: searchKinds, query: query,
                                        limit: max(topK, cands.count * 8), minScore: 0, among: cands)) ?? [])
                        .map { ($0.refKey, $0.score) },
                    uniquingKeysWith: { max($0, $1) })   // 视频多帧同 refKey,保最高分
                }
                let fwdScores = litScores(qFwd)
                let oppScores = litScores(qOpp)
                hits = Self.dedupByRef(hits.filter {
                    ((fwdScores[$0.refKey] ?? $0.score) - (oppScores[$0.refKey] ?? 0)) >= 0.010
                })
            }
            // 批104⑤:语义查询摘要行(逐词路径·结果内分支)
            // 批105②:移到尾线切割前——原记在切割后,常规路径 min=幸存集
            // 最小值,被砍分数不可见且与主管线(切割前记录)口径不一致;
            // 对比/尾线两个滤子交换顺序,集合等价,行为零变化
            IndexingCoordinator.DiagLog.shared.event(
                String(format: "语义[%@] 路径=%@(结果内) 命中=%d best=%.3f min=%.3f",
                       term, andRecall ? "AND宽召回" : "常规",
                       hits.count, hits.first?.score ?? 0, hits.map(\.score).min() ?? 0))
            // 深潜豁免尾线(低分段真命中会被 best×0.85 一刀砍光,与主管线同规);
            // 批104②:AND 宽召回在结果内分支同享免尾线(与全局路径同规,交集即精度)
            if best > 0, !genderDive, !andRecall { hits = hits.filter { $0.score >= best * 0.85 } }
            return videoFrameGuard(hits, term: term, gateScale: gateScale)
        }
        var hits: [SearchHit]
        if genderDive {
            // 深潜:第三档地板直取(0.08),性别裁决交对比边距
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                minScore: 0.08 * gateScale)) ?? [])
        } else if andRecall {
            // 批103:AND 逐词宽召回——交集即精度,逐词不再各自收紧(阶梯+尾线
            // 都是单词查询的防噪工具,组合查询里它们抢交集的活,把弱向量词
            // (阳光:裸搜语义仅9张)砍到交集饿死——254 装机 马尔代夫 阳光=1)。
            // 第三档地板直取+免尾线;证件豁免同规;OR 模式不走此路(并集
            // 仍需逐词精度)
            let exempt = QueryUnderstanding.floorExemptWords.contains(term)
                || QueryUnderstanding.chineseCounterpart(for: term) != nil
            let floor: Float = exempt ? 0.04 : 0.08 * gateScale
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                minScore: floor)) ?? [])
        } else {
        hits = Self.dedupByRef((try? store.search(
            space: siglip.space, kinds: searchKinds, query: q, limit: topK,
            minScore: 0.25 * gateScale)) ?? [])
        // 降档=薄判(tierMinResults,第六十二批,与主流程阶梯同规)
        if hits.count < Self.tierMinResults {
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                minScore: 0.15 * gateScale)) ?? [])
        }
        if hits.count < Self.tierMinResults,
           // 第六十八批:改反空兜底(与主流程照片第三档同规,判据见
           // floorTierPermitted);and:/or: 每词必并 OCR 命中(第十六批),
           // 文字覆盖探测与该通道同源。第六十九批:表内英文证件词经桥
           // 享有与中文证件类同等的第三档豁免
           // 第七十二批二修(验收 medium):探测与 ocrUnionHits 同口径扩为
           // 原词+繁→简转换词——繁体词只有原词命中 OCR 时(月结单印繁体),
           // 只探测归一后的简体词会误判未覆盖,批68 要挡的底网噪声被放进
           // 批91:证件豁免词的地板固定不随模型缩放(so400m 0.070
           // 捞不到真卡,与主流程同改;双通道对称面);批92:0.05→0.04
           // 给 ANE 嵌入抖动留余量(见 SigLIPEmbedder.queryVecCache)
           Self.floorTierPermitted(
               among: nil, semanticOnly: false, upperTierEmpty: hits.isEmpty,
               aliasExempt: QueryUnderstanding.floorExemptWords.contains(term)
                   || QueryUnderstanding.chineseCounterpart(for: term) != nil,
               textCovered: !QueryUnderstanding.scriptVariants(of: rawTerm).allSatisfy {
                   ((try? store.searchOCR(query: $0)) ?? []).isEmpty }) {
            let exempt = QueryUnderstanding.floorExemptWords.contains(term)
                || QueryUnderstanding.chineseCounterpart(for: term) != nil
            let floor: Float = exempt ? 0.04 : 0.08 * gateScale
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                minScore: floor)) ?? [])
        }
        }
        let best = hits.first?.score ?? 0
        // 第五十批:性别词对比过滤(与主管线视觉通道同规;第五十五批:对比串
        // =文献标准「a photo of a woman/man」形式+正反向对称打分+线0.010,
        // 真实照片标定见 QueryUnderstanding.genderEnglish 注释)
        if let pair = QueryUnderstanding.genderContrast(of: term),
           let qFwd = try? siglip.embedQuery(pair.fwd),
           let qOpp = try? siglip.embedQuery(pair.opp) {
            let cands = Set(hits.map(\.refKey))
            func litScores(_ query: [Float]) -> [String: Float] {
                Dictionary(((try? store.search(space: siglip.space, kinds: searchKinds, query: query,
                                    limit: max(topK, cands.count * 8), minScore: 0, among: cands)) ?? [])
                    .map { ($0.refKey, $0.score) },
                uniquingKeysWith: { max($0, $1) })   // 视频多帧同 refKey,保最高分
            }
            let fwdScores = litScores(qFwd)
            let oppScores = litScores(qOpp)
            hits = Self.dedupByRef(hits.filter {
                ((fwdScores[$0.refKey] ?? $0.score) - (oppScores[$0.refKey] ?? 0)) >= 0.010
            })
        }
        // 批104⑤:语义查询摘要行(逐词路径·全局分支)
        // 批105②:移到尾线切割前(与主管线/结果内分支同口径,被砍分数可见;
        // 对比/尾线两个滤子交换顺序,集合等价,行为零变化)
        IndexingCoordinator.DiagLog.shared.event(
            String(format: "语义[%@] 路径=%@ 命中=%d best=%.3f min=%.3f",
                   term, andRecall ? "AND宽召回" : "常规",
                   hits.count, hits.first?.score ?? 0, hits.map(\.score).min() ?? 0))
        // 低置信收紧与主流程同规:最佳 <0.13 = 无真匹配,只留贴近最佳的一小撮;
        // 深潜豁免(性别词的噪声闸=对比边距,与主管线同规);
        // 批103:AND 宽召回免尾线(交集即精度)
        if best > 0, !genderDive, !andRecall {
            hits = hits.filter { $0.score >= best * (best >= 0.13 * gateScale ? 0.85 : 0.95) }
        }
        return videoFrameGuard(hits, term: term, gateScale: gateScale)
    }

    /// 视频帧实据护栏(第四十批,与主管线 392-400 同规):SigLIP 对视频帧
    /// 区分度不足,0.08-0.15 是噪声带(2026-10-12 马尔代夫实测)——帧标签
    /// 与查询词有交集或分数 ≥0.15 强信号才入场,否则剔除视频帧命中
    /// gateScale=SigLIP 门槛缩放(so400m 0.87,验收拦下强信号线 0.15 漏缩)
    private func videoFrameGuard(_ hits: [SearchHit], term: String,
                                  gateScale: Float = 1.0) -> [SearchHit] {
        guard hits.contains(where: { $0.kind == .videoFrame }) else { return hits }
        let tokens = QueryUnderstanding.labelTokens(term)
        var labelRefs = Set<String>()
        if !tokens.isEmpty, let rows = try? store.allVideoFrameLabels() {
            for row in rows {
                let labelTokens = Set(row.title.lowercased()
                    .components(separatedBy: CharacterSet(charactersIn: ", ")))
                if !tokens.intersection(labelTokens).isEmpty { labelRefs.insert(row.refKey) }
            }
        }
        return hits.filter { $0.kind != .videoFrame
            || labelRefs.contains($0.refKey) || $0.score >= 0.15 * gateScale }
    }

    /// 按连接词合并各词命中列表:and=交集,or/无=并集(去重保序),最后相对化显示分
    private func combine(_ perTerm: [[SearchHit]], connector: QuerySpec.Connector) -> [SearchHit] {
        let lists = perTerm.filter { !$0.isEmpty }
        guard !lists.isEmpty else { return [] }
        var out: [SearchHit] = []
        var placed = Set<String>()
        if connector == .and {
            var common = Set(lists[0].map(\.refKey))
            for l in lists.dropFirst() { common.formIntersection(Set(l.map(\.refKey))) }
            guard !common.isEmpty else { return [] }
            for l in lists {
                for h in l where common.contains(h.refKey) && !placed.contains(h.refKey) {
                    placed.insert(h.refKey)
                    out.append(h)
                }
            }
        } else {
            for l in lists {
                for h in l where !placed.contains(h.refKey) {
                    placed.insert(h.refKey)
                    out.append(h)
                }
            }
        }
        return normalized(out)
    }

    /// 相对化显示分(第一名 100%)
    private func normalized(_ hits: [SearchHit]) -> [SearchHit] {
        let maxS = hits.map(\.score).max() ?? 1
        guard maxS > 0 else { return hits }
        return hits.map {
            SearchHit(kind: $0.kind, refKey: $0.refKey, frameIndex: $0.frameIndex,
                      space: $0.space, title: $0.title, date: $0.date,
                      score: $0.score / maxS, color: $0.color)
        }
    }

    /// 通道内按 refKey 去重(视频多帧/文件多块),供 RRF 按名次计分
    private static func dedupByRef(_ hits: [SearchHit]) -> [SearchHit] {
        var best: [String: SearchHit] = [:]
        for hit in hits {
            if let cur = best[hit.refKey] {
                if hit.score > cur.score { best[hit.refKey] = hit }
            } else {
                best[hit.refKey] = hit
            }
        }
        return best.values.sorted { $0.score > $1.score }
    }

    private static func assetPasses(_ asset: PHAsset, dateRange: QueryDateRange?,
                                    place: CLCircularRegion?) -> Bool {
        if let dateRange {
            guard let d = asset.creationDate, d >= dateRange.start, d < dateRange.end else { return false }
        }
        if let place {
            guard let loc = asset.location, place.contains(loc.coordinate) else { return false }
        }
        return true
    }

    /// 文件时间过滤:优先文件系统的修改/创建时间,拿不到退回导入时间
    private func fileMatchesFilters(_ record: ImportedFile, dateRange: QueryDateRange?) -> Bool {
        guard let dateRange else { return true }
        if let url = try? imports.resolveURL(record) {
            defer { url.stopAccessingSecurityScopedResource() }
            if let v = try? url.resourceValues(forKeys: [.contentModificationDateKey, .creationDateKey]) {
                let d = v.contentModificationDate ?? v.creationDate ?? record.addedAt
                return d >= dateRange.start && d < dateRange.end
            }
        }
        return record.addedAt >= dateRange.start && record.addedAt < dateRange.end
    }

    private func resolve(_ hits: [SearchHit], dateRange: QueryDateRange? = nil,
                         place: CLCircularRegion? = nil) -> [DisplayHit] {
        let assetIDs = Set(hits.filter {
            $0.kind == .photo || $0.kind == .videoFrame || $0.kind == .photoLabel || $0.kind == .videoFrameLabel
        }.map { $0.refKey })

        var assets: [String: PHAsset] = [:]
        if !assetIDs.isEmpty {
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: Array(assetIDs), options: nil)
            fetch.enumerateObjects { asset, _, _ in assets[asset.localIdentifier] = asset }
        }

        var out: [DisplayHit] = []
        // 颜色一次批量查回(第六十八批):逐条 colorForRef 是每条一次全表扫,
        // 大结果集线性放大(id 实测 1,690 条/22s);文件分支不走颜色,不参与
        let colorMap = store.colors(for: hits.filter {
            $0.kind != .face && $0.kind != .file && $0.kind != .fileChunk
        }.map { $0.refKey })
        for hit in hits {
            switch hit.kind {
            case .face:
                continue   // 人脸向量只做归组数据源,不进搜索结果
            case .photo, .photoLabel, .userTag, .videoFrame, .videoFrameLabel:
                guard let asset = assets[hit.refKey] else { continue }
                guard Self.assetPasses(asset, dateRange: dateRange, place: place) else { continue }
                let name = asset.mediaType == .video ? "视频" : "照片"
                out.append(DisplayHit(id: "\(hit.kind.rawValue)-\(hit.refKey)",
                                      refKey: hit.refKey,
                                      target: .asset(asset),
                                      kind: hit.kind,
                                      title: hit.title ?? name,
                                      score: hit.score,
                                      date: asset.creationDate,
                                      color: hit.color ?? colorMap[hit.refKey]))
            case .file, .fileChunk:
                guard let file = try? store.file(id: hit.refKey) else { continue }
                guard fileMatchesFilters(file, dateRange: dateRange) else { continue }
                out.append(DisplayHit(id: "\(hit.kind.rawValue)-\(hit.refKey)",
                                      refKey: hit.refKey,
                                      target: .file(file),
                                      kind: hit.kind,
                                      title: hit.title ?? file.name,
                                      score: hit.score,
                                      date: hit.date,
                                      color: hit.color))
            }
        }
        return out
    }
}
