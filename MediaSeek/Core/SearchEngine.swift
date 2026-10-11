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
        var s = raw
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
        query.components(separatedBy: CharacterSet(charactersIn: " ,,、,/"))
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

        let semanticOnly = spec.semanticOnly
        let siglip = await MainActor.run { models.siglip }
        lastSemanticTopScore = nil

        // 自然语言时间/地点:仅普通查询(文字:/语义:/组合语法不参与)。
        // 时间词命中即剥离;地点「在XX」需地理编码成功才剥离,失败原样保留。
        var dateRange: QueryDateRange?
        var placeRegion: CLCircularRegion?
        var placeLabel: String?
        if spec.channel == .any, spec.connector == .none, within == nil {
            let t = TimeQueryParser.parse(query)
            if let r = t.range {
                dateRange = r
                query = t.remaining.trimmingCharacters(in: .whitespaces)
            }
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
        }

        // 裸地名是「加法」不是「过滤」:内容通道与普通搜索完全一致(不得挂 GPS
        // 过滤——照片没有定位、截图在本乡本土,挂了就会把 38 个命中杀成 0),
        // GPS 地点浏览(该地拍摄的照片/视频,不依赖索引)只做并集排前
        if let placeRegion, placeLabel != nil, !query.isEmpty,
           spec.channel == .any, spec.connector == .none, within == nil {
            let main = try await runMainSearch(query: query, spec: spec, semanticOnly: semanticOnly,
                                               scope: scope, topK: topK, siglip: siglip,
                                               dateRange: dateRange, placeRegion: nil,
                                               placeLabel: placeLabel)
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

        if let within, !within.isEmpty {
            // 结果内 = 同一条主管线 + 基数集合约束:三维度/排名/标题与第一次搜索
            // 完全一致(旧 searchWithin 是特设核验流程,通道与主流程不同步,
            // 反复出病:门槛悬空、漏标签通道,已删)
            return try await runMainSearch(query: query, spec: spec, semanticOnly: semanticOnly,
                                           scope: scope, topK: topK, siglip: siglip,
                                           dateRange: nil, placeRegion: nil, placeLabel: nil,
                                           among: within)
        }
        if spec.channel == .text || spec.connector != .none {
            return SearchOutcome(hits: try await searchScoped(spec: spec, scope: scope,
                                   topK: topK, siglip: siglip, within: nil),
                                 textIDs: [], timeLabel: nil, placeLabel: nil)
        }
        return try await runMainSearch(query: query, spec: spec, semanticOnly: semanticOnly,
                                       scope: scope, topK: topK, siglip: siglip,
                                       dateRange: dateRange, placeRegion: placeRegion,
                                       placeLabel: placeLabel)
    }

    private func runMainSearch(query: String, spec: QuerySpec, semanticOnly: Bool,
                               scope: SearchScope, topK: Int, siglip: SigLIPEmbedder?,
                               dateRange: QueryDateRange?, placeRegion: CLCircularRegion?,
                               placeLabel: String?, among: Set<String>? = nil) async throws -> SearchOutcome {
        // among = 结果内基数(第一次搜索的结果集合):非 nil 时所有通道只在此集合内
        // 取命中,管线其余部分(通道权重/RRF 排名/三维度切分)与全局搜索完全一致
        // 多词查询:「2026 电脑屏幕」→ 含数字的词走 OCR 文字匹配,
        // 其余词走视觉/标签;「或」语义:照片满足任一条件(像/含字)即显示
        let terms = Self.splitTerms(query)
        let textTerms = terms.filter(Self.containsDigit)
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
        let tokens = Self.queryTokens(query: concept)
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
                    if m > 0 { scored.append((row.refKey, m, row.title)) }
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
            }
            if imageKinds.contains(.videoFrame) {
                let vFloor: Float = semanticOnly ? 0.25 : 0.22
                var vHits = Self.dedupByRef(try store.search(
                    space: siglip.space, kinds: [.videoFrame], query: q, limit: topK,
                    minScore: vFloor, among: among))
                if vHits.isEmpty {
                    vHits = Self.dedupByRef(try store.search(
                        space: siglip.space, kinds: [.videoFrame], query: q, limit: topK,
                        minScore: 0.12, among: among))
                }
                if vHits.isEmpty {
                    // 诊断实锤(2026-10-11):SigLIP2 在本库的真实分布 max 仅 0.07 上下
                    // (电脑屏幕 照片0.088/帧0.076,马尔代夫 照片0.090/帧0.076),
                    // 绝对门槛 0.12-0.30 全部悬空;视频帧又没有标签/OCR 兜底,一悬空即全空。
                    // 改相对标定:0.04 取候选(远高于底噪),尾部由下方 best×0.85 截断兜住
                    vHits = Self.dedupByRef(try store.search(
                        space: siglip.space, kinds: [.videoFrame], query: q, limit: topK,
                        minScore: 0.04, among: among))
                }
                hits.append(contentsOf: vHits)
                hits = Self.dedupByRef(hits)
            }
            // 兜底后仍空 = 全库最高分都不到 0.08,记下最高原始分供空态提示
            if hits.isEmpty {
                lastSemanticTopScore = (try? store.search(
                    space: siglip.space, kinds: imageKinds, query: q, limit: 1,
                    minScore: -1, among: among))?.first?.score
            }
            // 相对尾部截断:远弱于头部的长尾(如「人」里混入的屏幕翻拍照)直接砍掉
            let best = hits.first?.score ?? 0
            channels.append((1.0, best > 0 ? hits.filter { $0.score >= best * 0.85 } : hits))
        }

        // 用户标签通道:中文子串精确匹配(文字: 排他,不参与)
        if spec.channel != .text, imageKinds.contains(.photo) {
            let tagRefs = ((try? store.searchUserTags(query: concept)) ?? [])
                .filter { among?.contains($0) ?? true }
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
            if among == nil, scope != .file, fileHits.count > 1 {
                let scores = fileHits.map(\.score).sorted()
                let median = scores[scores.count / 2]
                let best = scores.last ?? 0
                if best < 0.40 || best - median < 0.10 { fileHits = [] }
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
        if !semanticOnly, scope == .all || scope == .photo {
            // OCR 命中可能来自导入文件(PDF OCR 原文):按 files 表区分 文件/照片
            let fileIds = (try? store.fileIds()) ?? []
            let videoRefs = (try? store.refKeys(kinds: [.videoFrame])) ?? []
            // OCR 命中的 ref 可能是 照片/文件/视频(视频帧 OCR):按库内身份归类
            func ocrKind(_ ref: String) -> ItemKind {
                if scope == .all, fileIds.contains(ref) { return .file }
                if videoRefs.contains(ref) { return .videoFrame }
                return .photo
            }
            var seen = Set<String>()
            var rank = 0
            for t in (textTerms.isEmpty ? [query] : textTerms) where !t.isEmpty {
                for ref in (try? store.searchOCR(query: t)) ?? []
                where !seen.contains(ref.refKey) && (among?.contains(ref.refKey) ?? true) {
                    seen.insert(ref.refKey)
                    add(SearchHit(kind: ocrKind(ref.refKey), refKey: ref.refKey, frameIndex: 0,
                                  space: "ocr", title: "含「\(t)」:\(Self.ocrSnippet(ref.text, t))",
                                  date: nil, score: 1.0, color: nil),
                        weight: 8.0, rank: rank)
                    rank += 1
                }

                // 长中文词整句 LIKE 会因语序漏匹配(搜「医学出生证明」,证书上印的
                // 是「出生医学证明」)→ 二字滑窗片段 OR,≥2 个片段命中才算,按数排序
                if textTerms.isEmpty, query.count >= 4 {
                    var grams: [String] = []
                    var seenGram = Set<String>()
                    let chars = Array(query)
                    for i in 0..<(chars.count - 1) {
                        let g = String(chars[i...i + 1])
                        if seenGram.insert(g).inserted { grams.append(g) }
                    }
                    var counts: [String: Int] = [:]
                    for g in grams {
                        for ref in (try? store.searchOCR(query: g)) ?? []
                        where among?.contains(ref.refKey) ?? true {
                            counts[ref.refKey, default: 0] += 1
                        }
                    }
                    let multi = counts.filter { $0.value >= 2 }
                        .sorted { $0.value > $1.value }
                        .prefix(topK)
                    for (i, e) in multi.enumerated() {
                        add(SearchHit(kind: ocrKind(e.key), refKey: e.key, frameIndex: 0,
                                      space: "ocr", title: "含「\(query)」相关文字(命中\(e.value)处)",
                                      date: nil, score: 1.0, color: nil),
                            weight: 5.0, rank: rank + i)
                    }
                }
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

    /// 查询词集合:清洗后的原文分词 + 词典英译分词,再做一跳同义扩展(供标签 token 比对)。
    /// 长查询(≥3 字)剔除泛化词:people/document 这类标签什么照片都能沾,只制造噪声
    private static func queryTokens(query: String) -> Set<String> {
        var tokens = Set(query.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty })
        if let en = QueryUnderstanding.english(for: query) {
            for t in en.lowercased().split(separator: " ") { tokens.insert(String(t)) }
        }
        tokens = QueryUnderstanding.expandedTokens(tokens)
        if query.count >= 3 {
            tokens.subtract(QueryUnderstanding.genericTokens)
        }
        return tokens
    }


    /// 显式语法流程:通道(文字/语义/自动)× 连接词(and/or)自由组合
    private func searchScoped(spec: QuerySpec, scope: SearchScope,
                              topK: Int, siglip: SigLIPEmbedder?, within: Set<String>? = nil) async throws -> [DisplayHit] {
        guard scope == .all || scope == .photo else { return [] }   // v1 只作用于照片
        let terms = Self.splitTerms(spec.body)
        guard !terms.isEmpty else { return [] }

        // 文字通道:每个词独立查 OCR,连接词决定 AND/OR。
        // and 模式标题列出全部命中词(交集里每张都含全部词,不能只显示第一个)
        // 结果内模式:只在这些照片的文字索引里查
        if spec.channel == .text {
            let fileIds = (try? store.fileIds()) ?? []
            let videoRefs = (try? store.refKeys(kinds: [.videoFrame])) ?? []
            // OCR 命中的 ref 可能是 照片/文件/视频(视频帧 OCR):按库内身份归类
            func ocrKind(_ ref: String) -> ItemKind {
                if scope == .all, fileIds.contains(ref) { return .file }
                if videoRefs.contains(ref) { return .videoFrame }
                return .photo
            }
            var perTerm: [[SearchHit]] = []
            for t in terms {
                let refs: [(refKey: String, text: String)] = within.map { among in
                    ((try? store.searchOCR(query: t, among: Array(among))) ?? []).map { ($0, "") }
                } ?? (((try? store.searchOCR(query: t)) ?? []).prefix(topK).map { ($0.refKey, $0.text) })
                let connectorIsAnd = (spec.connector == .and)
                perTerm.append(refs.enumerated().map { rank, ref in
                    let base = within != nil
                        ? "结果内文字命中「\(t)」"
                        : (connectorIsAnd
                           ? "文字全含:\(terms.joined(separator: " ")) —「\(t)」\(Self.ocrSnippet(ref.text, t))"
                           : "文字命中「\(t)」:\(Self.ocrSnippet(ref.text, t))")
                    return SearchHit(kind: ocrKind(ref.refKey), refKey: ref.refKey, frameIndex: 0, space: "ocr",
                              title: base,
                              date: nil, score: Float(1.0 / Double(rank + 1)), color: nil)
                })
            }
            return resolve(combine(perTerm, connector: spec.connector))
        }

        // 语义/自动通道:数字词→文字,其余词→视觉
        var perTerm: [[SearchHit]] = []
        for t in terms {
            if Self.containsDigit(t) {
                let refs: [String] = within.map { among in
                    (try? store.searchOCR(query: t, among: Array(among))) ?? []
                } ?? ((try? store.searchOCR(query: t)) ?? []).prefix(topK).map(\.refKey)
                perTerm.append(refs.enumerated().map { rank, refKey in
                    SearchHit(kind: .photo, refKey: refKey, frameIndex: 0, space: "ocr",
                              title: "含「\(t)」文字", date: nil,
                              score: Float(1.0 / Double(rank + 1)), color: nil)
                })
            } else {
                // 结果内时语义分数限定在基数集合内算(同 searchWithin 的修正)
                var hits = semanticHits(t, siglip: siglip, topK: topK, within: within)
                if let within { hits = hits.filter { within.contains($0.refKey) } }
                perTerm.append(hits)
            }
        }
        return resolve(combine(perTerm, connector: spec.connector))
    }

    /// SigLIP2 官方零样本模板("This is a photo of X.")对短语类查询更友好;
    /// 词典命中的精选英文描述套模板,未命中(长句/自由文本)保持原文。
    /// ⚠️ 提示词集成(多模板平均)虽是通行做法,但会平移全库语义分数、
    /// 现有门槛按单模板分布标定——要上必须先在设备上 A/B 单独验证
    private static func siglipQuery(_ concept: String) -> String {
        if let en = QueryUnderstanding.english(for: concept) {
            return "This is a photo of \(en)."
        }
        return concept
    }

    /// 单词语义命中(SigLIP 视觉,门槛 0.14 + 0.08 弱信号兜底 + 相对尾部截断)
    private func semanticHits(_ term: String, siglip: SigLIPEmbedder?, topK: Int,
                              within: Set<String>? = nil) -> [SearchHit] {
        guard let siglip else { return [] }
        guard let q = try? siglip.embedQuery(Self.siglipQuery(term)) else { return [] }
        // 结果内核验:分数只在基数集合内算,门槛放宽、相对截断以集合内最佳为基准——
        // 否则全局排名的尾部截断会在取交集前把候选砍光(「行情图 ∩ 语义:电脑屏幕=0」事故:
        // 全库最高分是实拍屏幕,行情截图全被 85% 线砍掉,交集自然为空)
        if let within, !within.isEmpty {
            var hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: [.photo], query: q, limit: topK,
                minScore: 0.20, among: within)) ?? [])
            if hits.isEmpty {
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: [.photo], query: q, limit: topK,
                    minScore: 0.10, among: within)) ?? [])
            }
            if hits.isEmpty {
                // SigLIP2 本库真实分布 max 0.07-0.09(2026-10-11 诊断实锤),
                // 0.20/0.10 全悬空 → 结果内语义词恒空;0.04 相对档,
                // 尾部由下方 best×0.85(以集合内最佳为基准)截断
                hits = Self.dedupByRef((try? store.search(
                    space: siglip.space, kinds: [.photo], query: q, limit: topK,
                    minScore: 0.04, among: within)) ?? [])
            }
            let best = hits.first?.score ?? 0
            if best > 0 { hits = hits.filter { $0.score >= best * 0.85 } }
            return hits
        }
        var hits = Self.dedupByRef((try? store.search(
            space: siglip.space, kinds: [.photo], query: q, limit: topK, minScore: 0.25)) ?? [])
        if hits.isEmpty {
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: [.photo], query: q, limit: topK, minScore: 0.15)) ?? [])
        }
        if hits.isEmpty {
            hits = Self.dedupByRef((try? store.search(
                space: siglip.space, kinds: [.photo], query: q, limit: topK, minScore: 0.04)) ?? [])
        }
        let best = hits.first?.score ?? 0
        if best > 0 { hits = hits.filter { $0.score >= best * 0.85 } }
        return hits
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
