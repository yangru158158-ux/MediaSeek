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
        enum Channel { case any, text, semantic }
        let connector: Connector
        let channel: Channel
        let body: String
        var isExplicit: Bool { channel != .any || connector != .none }
        var semanticOnly: Bool { channel == .semantic && connector == .none }
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
        var query = QueryUnderstanding.core(spec.body)
        guard !query.isEmpty else { return .empty }

        let siglip = await MainActor.run { models.siglip }
        lastSemanticTopScore = nil

        // 自然语言时间/地点:普通查询与结果内都参与(组合语法/前缀语法不参与)。
        // 时间词命中即剥离——结果内同样生效:从当前基数集里按拍摄/创建时间筛(与第一次搜索同构);
        // 地点「在XX」仅普通查询(结果内挂地理编码徒增异步与误伤),失败原样保留。
        var dateRange: QueryDateRange?
        var placeRegion: CLCircularRegion?
        var placeLabel: String?
        if spec.channel == .any, spec.connector == .none {
            let t = TimeQueryParser.parse(query)
            if let r = t.range {
                dateRange = r
                query = t.remaining.trimmingCharacters(in: .whitespaces)
            }
            if within == nil {
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

        // 裸地名是「加法」不是「过滤」:内容通道与普通搜索完全一致(不得挂 GPS
        // 过滤——照片没有定位、截图在本乡本土,挂了就会把 38 个命中杀成 0),
        // GPS 地点浏览(该地拍摄的照片/视频,不依赖索引)只做并集排前
        if let placeRegion, placeLabel != nil, !query.isEmpty,
           spec.channel == .any, spec.connector == .none, within == nil {
            let main = try await runBareSearch(query: query, spec: spec, scope: scope, topK: topK,
                                               siglip: siglip, dateRange: dateRange,
                                               placeRegion: nil, placeLabel: placeLabel, among: nil)
            var browseHits: [DisplayHit] = []
            photo.fetchAllAssets().enumerateObjects { asset, _, _ in
                guard browseHits.count < topK else { return }
                if scope == .file { return }   // 文件无 GPS,地点浏览只列相册内容
                if scope == .photo, asset.mediaType != .image { return }
                if scope == .video, asset.mediaType != .video { return }
                guard Self.assetPasses(asset, dateRange: dateRange, place: placeRegion) else { return }
                let isVideo = asset.mediaType == .video
                browseHits.append(DisplayHit(
                    id: "\(isVideo ? ItemKind.videoFrame.rawValue : ItemKind.photo.rawValue)-\(asset.localIdentifier)",
                    refKey: asset.localIdentifier,
                    target: .asset(asset),
                    kind: isVideo ? .videoFrame : .photo,
                    title: isVideo ? "视频" : "照片",
                    score: 1.0, date: asset.creationDate,
                    color: self.store.colorForRef(asset.localIdentifier)))
            }
            // 地点浏览命中(拍摄时间倒序,fetchAllAssets 即此顺序)排前,内容命中随后
            var merged = browseHits
            var seen = Set(browseHits.map { $0.refKey })
            for h in main.hits where !seen.contains(h.refKey) { merged.append(h) }
            return SearchOutcome(hits: merged, textIDs: main.textIDs,
                                 timeLabel: dateRange?.label, placeLabel: placeLabel)
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
            if let allFiles = try? imports.allFiles() {
                for f in allFiles where hits.count < topK {
                    if fileMatchesFilters(f, dateRange: dateRange) {
                        hits.append(SearchHit(kind: .file, refKey: f.id, frameIndex: 0,
                                              space: "browse", title: f.name,
                                              date: f.addedAt, score: 1.0, color: nil))
                    }
                }
            }
            return SearchOutcome(hits: resolve(hits), textIDs: [],
                                 timeLabel: dateRange?.label, placeLabel: placeLabel)
        }

        // 第四十一批:此分支只接管裸搜(或任何 spec 的文件范围)——前缀查询
        // (and:/or:/文字:)在结果内也要落到下方 searchScoped 保持 AND/排他
        // 语义,与全库搜索一致;此前被拦进主管线,文字: 在结果内多词退化成
        // 并集。文件范围例外:searchScoped 有 scope 守卫拦截,保持主管线
        if let within, !within.isEmpty,
           scope == .file || (spec.channel == .any && spec.connector == .none) {
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
                                 timeLabel: nil, placeLabel: nil)
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
    /// 命中再交集,已验收实现,不另造一套防漂移);单词与文件范围(file 被
    /// searchScoped 守卫拦截)保持原主管线;or: 前缀仍可显式取并集。
    /// 已知收窄(有意):全部范围多词改道后不跑 Gemma 文件语义通道——旧裸搜
    /// 对文件是整句模糊匹配,与逐词 AND 精确语义不兼容,文件改靠 OCR 原文
    /// (PDF/CSV/XLSX/TXT 内容均已入库 ocr_text)双词精确命中。
    /// 空态分数提示(lastSemanticTopScore)仅 语义: 查询展示,而 语义: 已被
    /// 下方 channel 校验挡回主管线(主管线自己写该字段),本函数无需写
    private func runBareSearch(query: String, spec: QuerySpec, scope: SearchScope,
                               topK: Int, siglip: SigLIPEmbedder?, dateRange: QueryDateRange?,
                               placeRegion: CLCircularRegion?, placeLabel: String?,
                               among: Set<String>? = nil) async throws -> SearchOutcome {
        // 只对裸搜(channel == .any)生效:语义: 前缀必须保持排他(只要语义
        // 命中),不得被改道 and: 并入 OCR——验收 medium 实锤(此前只看
        // scope 和词数,语义:多词被误改道且未在提交说明声明)
        if spec.channel == .any, scope != .file, Self.splitTerms(query).count >= 2 {
            let andSpec = QuerySpec(connector: .and, channel: .any, body: query)
            let scoped = try await searchScoped(spec: andSpec, scope: scope, topK: topK,
                                                siglip: siglip, within: among,
                                                dateRange: dateRange)
            return SearchOutcome(hits: scoped.hits, textIDs: scoped.textIDs,
                                 timeLabel: dateRange?.label, placeLabel: placeLabel)
        }
        return try await runMainSearch(query: query, spec: spec, semanticOnly: spec.semanticOnly,
                                       scope: scope, topK: topK, siglip: siglip,
                                       dateRange: dateRange, placeRegion: placeRegion,
                                       placeLabel: placeLabel, among: among)
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
        let visualTerms = terms.filter { !Self.containsDigit($0) }
        let concept = visualTerms.isEmpty ? query : visualTerms.joined(separator: " ")

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
        if !imageKinds.isEmpty, spec.channel != .text, let siglip {
            let q = try siglip.embedQuery(Self.siglipQuery(concept))
            // 照片与视频帧分开走各自的门槛阶梯:分数分布不同,且照片行为已验收冻结
            var hits: [SearchHit] = []
            if imageKinds.contains(.photo) {
                let floor: Float = semanticOnly ? 0.25 : 0.30
                hits = Self.dedupByRef(try store.search(
                    space: siglip.space, kinds: [.photo], query: q, limit: topK,
                    minScore: floor, among: among))
                // 弱信号兜底:品牌词/抽象词(「微信」「聊天」)对截图的视觉相似度
                // 天然低于常规门槛,一刀切会整词清零(2026-10-10 用户实测三个词全空)
                if hits.isEmpty {
                    hits = Self.dedupByRef(try store.search(
                        space: siglip.space, kinds: [.photo], query: q, limit: topK,
                        minScore: 0.18, among: among))
                }
                if hits.isEmpty {
                    // 第三档分两种:全局地板 0.08(高于无关带上沿、低于命中带下沿);
                    // 结果内 0.03——用户点名的小集合里相对排名优先,弱分真匹配(证件照
                    // 对「身份证」只有 0.05-0.08)也要浮顶,噪声由低置信尾线 0.95 收
                    let floor: Float = among != nil ? 0.03 : 0.08
                    hits = Self.dedupByRef(try store.search(
                        space: siglip.space, kinds: [.photo], query: q, limit: topK,
                        minScore: floor, among: among))
                }
            }
            if imageKinds.contains(.videoFrame) {
                // 真尺度校准:纯 SigLIP 的视频帧区分度不足——古庙绿崖 0.11 vs 真海滩 0.13-0.14,
                // 相对标定尾线必然放进噪声(2026-10-12 马尔代夫实测)。按 113 确立的通道优先级
                // (标签零幻觉 > OCR > SigLIP 兜底),视频帧入场必须有实据:
                // 帧标签与查询词有交集,或分数 ≥0.15 的强信号
                var vHits = Self.dedupByRef(try store.search(
                    space: siglip.space, kinds: [.videoFrame], query: q, limit: topK,
                    minScore: 0.08, among: among))
                vHits = vHits.filter { videoLabelRefs.contains($0.refKey) || $0.score >= 0.15 }
                hits.append(contentsOf: vHits)
                hits = Self.dedupByRef(hits)
            }
            // 兜底后仍空 = 全库最高分都不到 0.08,记下最高原始分供空态提示
            if hits.isEmpty {
                lastSemanticTopScore = (try? store.search(
                    space: siglip.space, kinds: imageKinds, query: q, limit: 1,
                    minScore: -1, among: among))?.first?.score
            }
            // 第五十批:性别词对比过滤——SigLIP 绝对分区分男女弱(0.08 地板
            // 男女同放行,用户实测搜 woman 杨如程玉婷混列),相对差才可靠
            // (实测:女照 woman分-man分=+0.021 / 男照 -0.027~-0.038 / 中性
            // -0.004)。反义词同跑,只留正向分领先 ≥0.012 的命中;合影两性
            // 分数相抵(差≈0)也被滤掉——性别词搜索要的是单一性别主导的照片。
            // 反查用 among=本组命中集(第五十一批):限 topK 排位时大库漏取
            // 反向分会按 0 放行穿透过滤;限定在本组内取分,命中集 ≤topK 必全取到
            if let opp = QueryUnderstanding.genderOpposite(of: concept),
               let qOpp = try? siglip.embedQuery(Self.siglipQuery(opp)) {
                let cands = Set(hits.map(\.refKey))
                let oppScores = Dictionary((try store.search(space: siglip.space, kinds: imageKinds, query: qOpp,
                                  limit: max(topK, cands.count), minScore: 0, among: cands))
                    .map { ($0.refKey, $0.score) },
                uniquingKeysWith: { max($0, $1) })   // 视频多帧同 refKey,保最高反向分
                hits = Self.dedupByRef(hits.filter {
                    ($0.score - (oppScores[$0.refKey] ?? 0)) >= 0.012
                })
            }
            // 相对尾部截断:远弱于头部的长尾(如「人」里混入的屏幕翻拍照)直接砍掉。
            // 低置信收紧(真尺度 2026-10-12):整组最佳 <0.13 = 没有真匹配,列表全是
            // 人像/风景基线噪声(同一张海景人像 马尔代夫0.09/护照0.12/身份证0.12),
            // 此时尾线收到 best×0.95 只留贴近最佳的一小撮,不灌整屏
            let best = hits.first?.score ?? 0
            let tailFactor: Float = best >= 0.13 ? 0.85 : 0.95
            channels.append((1.0, best > 0 ? hits.filter { $0.score >= best * tailFactor } : hits))
        }

        // 用户标签通道:中文子串精确匹配(文字: 排他,不参与)
        if spec.channel != .text, imageKinds.contains(.photo) {
            // 第二十一批:多词按「或」逐词命中标签——整串「杨如 男」永远匹配
            // 不到标签「杨如」,他本人的照片在裸搜多词下一张出不来
            var tagRefs = Set<String>()
            for t in (terms.isEmpty ? [concept] : terms) {
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
                // 标题优先取语义通道命中(「含…文字」标记只留给纯 OCR 命中)
                let winner: SearchHit
                if cur.hit.space == "ocr", hit.space != "ocr" { winner = hit }
                else { winner = hit.score > cur.hit.score ? hit : cur.hit }
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
        if !semanticOnly, scope != .file {
            let allowedKinds = Self.scopeAllowedKinds(scope)
            let classifier = OCRRefClassifier(store: store)
            var seen = Set<String>()
            var rank = 0
            // 第二十一批:恢复「每词独立命中」(build 32 的「或」语义在 CI 88
            // 重构时窄化成了「只有含数字的词走 OCR」)——纯中文多词(杨如 男)
            // 整串带空格当子串查,健康码印的是「杨如，男」永远 miss,用户实测
            // 裸搜搜不出、and: 反而搜得出。逐词并集,与 and:/or: 通道同构
            let ocrTerms = terms.isEmpty ? [query] : terms
            for t in ocrTerms where !t.isEmpty {
                // 证件类 OCR 别名:人像面不印证件名(身份证正面只有「公民身份号码」等字段),
                // 只搜证件名会漏掉全部正面照——补字段级别名并集保召回
                let variants = [t] + (QueryUnderstanding.ocrAliases[t] ?? [])
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
                              dateRange: QueryDateRange? = nil) async throws -> (hits: [DisplayHit], textIDs: Set<String>) {
        guard scope != .file else { return ([], []) }   // 文件范围无照片/视频(第三十九批:视频范围原被误拦)
        let terms = Self.splitTerms(spec.body)
        guard !terms.isEmpty else { return ([], []) }
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
                // 内联 switch(其 .file 分支因上方 guard 永不可达)
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
            return (resolve(combined, dateRange: dateRange), textIDs)
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
                // 结果内时语义分数限定在基数集合内算(同 searchWithin 的修正)
                var hits = semanticHits(t, siglip: siglip, topK: topK, within: within, scope: scope)
                if let within { hits = hits.filter { within.contains($0.refKey) } }
                // 「像或含字」同构(第十六批):裸搜模式里每个词 = OCR∪视觉任一
                // 命中即算,and:/or: 自动通道此前只送语义——证件/文件类词语义
                // 天生弱(0.05-0.08 过不了地板),and:身份证 正面 恒空。每词并入
                // OCR(含证件别名)命中再按连接词交并;同照片双命中时语义条目
                // 优先(带真实分),OCR 只补缺口
                var seen = Set(hits.map(\.refKey))
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
                    for r in ((try? store.searchUserTags(query: t)) ?? [])
                        .filter({ within?.contains($0) ?? true }) where !seen.contains(r) {
                        seen.insert(r)
                        hits.append(SearchHit(kind: .photo, refKey: r, frameIndex: 0,
                                              space: "userTag", title: nil, date: nil,
                                              score: 0.9, color: nil))
                    }
                }
                perTerm.append(hits)
            }
        }
        var combined = combine(perTerm, connector: spec.connector)
        // AND 模式逐照片合成标题(第十七批):combine 去重只保留第一个词的命中
        // 条目,and:杨如 男 的卡片全部显示「含「杨如」文字」——第二个词的
        // 命中只字不提,用户看不出交集凭什么成立(照片明明同时含「男」)。
        // 逐照片列出每词的命中方式(文字/标签/语义)
        if spec.connector == .and, terms.count > 1 {
            var info: [String: [String]] = [:]
            for (t, list) in zip(terms, perTerm) {
                for h in list {
                    let way = h.space == "ocr" ? "文字"
                        : (h.space == "userTag" ? "标签" : "语义")
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
        return (resolve(combined, dateRange: dateRange), textIDs)
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
        for term in [t] + (QueryUnderstanding.ocrAliases[t] ?? []) {
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
    private static func siglipQuery(_ concept: String) -> String {
        if let en = QueryUnderstanding.english(for: concept) {
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
        return concept
    }

    /// 单词语义命中(SigLIP 视觉,门槛 0.14 + 0.08 弱信号兜底 + 相对尾部截断)
    private func semanticHits(_ term: String, siglip: SigLIPEmbedder?, topK: Int,
                              within: Set<String>? = nil,
                              scope: SearchScope = .all) -> [SearchHit] {
        guard let siglip else { return [] }
        guard let q = try? siglip.embedQuery(Self.siglipQuery(term)) else { return [] }
        // 第三十九批:按范围取 kinds(与主管线 imageKinds 同构,.file→[] 空)——
        // 此前全部写死 [.photo],视频范围下视频帧向量从不被查询
        let searchKinds: [ItemKind] = scope == .all ? [.photo, .videoFrame]
            : (scope == .photo ? [.photo] : (scope == .video ? [.videoFrame] : []))
        // 结果内核验:分数只在基数集合内算,门槛放宽、相对截断以集合内最佳为基准——
        // 否则全局排名的尾部截断会在取交集前把候选砍光(「行情图 ∩ 语义:电脑屏幕=0」事故:
        // 全库最高分是实拍屏幕,行情截图全被 85% 线砍掉,交集自然为空)
        if let within, !within.isEmpty {
            var hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                minScore: 0.20, among: within)) ?? [])
            if hits.isEmpty {
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                    minScore: 0.10, among: within)) ?? [])
            }
            if hits.isEmpty {
                // 结果内第三档:0.20/0.10 在命中带上悬空 → 结果内语义词恒空;
                // 地板 0.03(结果内=点名小集合,相对排名优先,与主流程照片第三档同规),
                // 尾部由 best×0.95 低置信尾线截断
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: searchKinds, query: q, limit: topK,
                    minScore: 0.03, among: within)) ?? [])
            }
            let best = hits.first?.score ?? 0
            if best > 0 { hits = hits.filter { $0.score >= best * 0.85 } }
            // 第五十批:性别词对比过滤(结果内同规,见 QueryUnderstanding.genderOpposites;
            // 反查 among=本组命中集,第五十一批大库 topK 边界修复同主管线)
            if let opp = QueryUnderstanding.genderOpposite(of: term),
               let qOpp = try? siglip.embedQuery(Self.siglipQuery(opp)) {
                let cands = Set(hits.map(\.refKey))
                let oppScores = Dictionary(((try? store.search(space: siglip.space, kinds: searchKinds, query: qOpp,
                                    limit: max(topK, cands.count), minScore: 0, among: cands)) ?? [])
                    .map { ($0.refKey, $0.score) },
                uniquingKeysWith: { max($0, $1) })   // 视频多帧同 refKey,保最高反向分
                hits = Self.dedupByRef(hits.filter {
                    ($0.score - (oppScores[$0.refKey] ?? 0)) >= 0.012
                })
            }
            return videoFrameGuard(hits, term: term)
        }
        var hits = Self.dedupByRef((try? store.search(
            space: siglip.space, kinds: searchKinds, query: q, limit: topK, minScore: 0.25)) ?? [])
        if hits.isEmpty {
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK, minScore: 0.15)) ?? [])
        }
        if hits.isEmpty {
            // 真尺度地板 0.08(与主流程照片第三档同规)
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: searchKinds, query: q, limit: topK, minScore: 0.08)) ?? [])
        }
        let best = hits.first?.score ?? 0
        // 低置信收紧与主流程同规:最佳 <0.13 = 无真匹配,只留贴近最佳的一小撮
        if best > 0 { hits = hits.filter { $0.score >= best * (best >= 0.13 ? 0.85 : 0.95) } }
        // 第五十批:性别词对比过滤(与主管线视觉通道同规,实测依据见
        // QueryUnderstanding.genderOpposites 注释;反查 among=本组命中集,
        // 第五十一批大库 topK 边界修复同主管线)
        if let opp = QueryUnderstanding.genderOpposite(of: term),
           let qOpp = try? siglip.embedQuery(Self.siglipQuery(opp)) {
            let cands = Set(hits.map(\.refKey))
            let oppScores = Dictionary(((try? store.search(space: siglip.space, kinds: searchKinds, query: qOpp,
                                    limit: max(topK, cands.count), minScore: 0, among: cands)) ?? [])
                .map { ($0.refKey, $0.score) },
                uniquingKeysWith: { max($0, $1) })   // 视频多帧同 refKey,保最高反向分
            hits = Self.dedupByRef(hits.filter {
                ($0.score - (oppScores[$0.refKey] ?? 0)) >= 0.012
            })
        }
        return videoFrameGuard(hits, term: term)
    }

    /// 视频帧实据护栏(第四十批,与主管线 392-400 同规):SigLIP 对视频帧
    /// 区分度不足,0.08-0.15 是噪声带(2026-10-12 马尔代夫实测)——帧标签
    /// 与查询词有交集或分数 ≥0.15 强信号才入场,否则剔除视频帧命中
    private func videoFrameGuard(_ hits: [SearchHit], term: String) -> [SearchHit] {
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
            || labelRefs.contains($0.refKey) || $0.score >= 0.15 }
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
                                      color: hit.color ?? store.colorForRef(hit.refKey)))
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
