import Foundation
import Photos

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
final class SearchEngine {
    private let store: VectorStore
    private let models: ModelManager
    private let photo: PhotoLibraryService
    private let imports: ImportLibrary

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

    func search(_ rawQuery: String, scope: SearchScope, topK: Int = 4000)   // 4000≈全库:SQLite 向量检索本就全表扫描,加大 LIMIT 几乎零成本 async throws -> [DisplayHit] {
        let raw = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return [] }
        let spec = Self.parseQuery(raw)
        let query = QueryUnderstanding.core(spec.body)
        guard !query.isEmpty else { return [] }

        // 显式语法(文字:/and:/or:/and 语义: 等)走独立流程;纯「语义:词」走主流程但跳过 OCR
        let semanticOnly = spec.semanticOnly
        if spec.channel == .text || spec.connector != .none {
            let siglip = await MainActor.run { models.siglip }
            return try await searchScoped(spec: spec, scope: scope, topK: topK, siglip: siglip)
        }

        // 多词查询:「2026 电脑屏幕」→ 含数字的词走 OCR 文字匹配,
        // 其余词走视觉/标签;「或」语义:照片满足任一条件(像/含字)即显示
        let terms = Self.splitTerms(query)
        let textTerms = terms.filter(Self.containsDigit)
        let visualTerms = terms.filter { !Self.containsDigit($0) }
        let concept = visualTerms.isEmpty ? query : visualTerms.joined(separator: " ")

        let imageKinds: [ItemKind] = scope == .all || scope == .photo
            ? [.photo] : (scope == .video ? [.videoFrame] : [])
        // Gemma 语义只负责文件名/文件内容——照片标签的余弦虚高且区分度低
        // (「猫」vs「people, adult」和 vs「cat」差不了几分),会顶掉真命中
        let fileKinds: [ItemKind] = scope == .all || scope == .file ? [.file, .fileChunk] : []

        var channels: [(weight: Double, hits: [SearchHit])] = []
        let siglip = await MainActor.run { models.siglip }
        let gemma = await MainActor.run { models.gemma }

        // 标签精确通道(优先):词典英译+同义词与 Vision 英文标签 token 级比对,零幻觉
        var labelHits: [SearchHit] = []
        if !imageKinds.isEmpty {
            let tokens = Self.queryTokens(query: concept)
            if !tokens.isEmpty, let rows = try? store.allPhotoLabels() {
                var scored: [(refKey: String, matches: Int, title: String)] = []
                for row in rows {
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
        }
        if !labelHits.isEmpty { channels.append((1.6, labelHits)) }

        // 视觉通道:SigLIP2 原生多语言(分词实验已证),中文原文直接查;
        // 不再用语义路由——泛化标签(document/screenshot)会被几乎任何词
        // 以 ≥0.60 命中,造成整库截图照灌进结果(「医学出生证明」事故)
        // 显式「语义:」查询门槛放宽到 0.14(用户点名要语义召回)
        if !imageKinds.isEmpty, let siglip {
            let q = try siglip.embedQuery(QueryUnderstanding.english(for: concept) ?? concept)
            let floor = semanticOnly ? 0.14 : 0.18
            let hits = Self.dedupByRef(try store.search(
                space: siglip.space, kinds: imageKinds, query: q, limit: topK, minScore: Float(floor)))
            // 相对尾部截断:远弱于头部的长尾(如「人」里混入的屏幕翻拍照)直接砍掉
            let best = hits.first?.score ?? 0
            channels.append((1.0, best > 0 ? hits.filter { $0.score >= best * 0.85 } : hits))
        }

        // 用户标签通道:中文子串精确匹配
        if scope != .file, let tagRefs = try? store.searchUserTags(query: concept), !tagRefs.isEmpty {
            channels.append((1.6, tagRefs.map {
                SearchHit(kind: .photo, refKey: $0, frameIndex: 0,
                          space: "userTag", title: nil, date: nil, score: 1.0, color: nil)
            }))
        }

        // 文件语义通道:Gemma 文本嵌入查文件名/内容
        if !fileKinds.isEmpty, let gemma {
            let q = try gemma.embedQuery(concept)
            channels.append((1.0, Self.dedupByRef(try store.search(
                space: gemma.space, kinds: fileKinds, query: q, limit: topK, minScore: 0.12))))
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
            var seen = Set<String>()
            var rank = 0
            for t in (textTerms.isEmpty ? [query] : textTerms) where !t.isEmpty {
                for ref in (try? store.searchOCR(query: t)) ?? [] where !seen.contains(ref.refKey) {
                    seen.insert(ref.refKey)
                    add(SearchHit(kind: .photo, refKey: ref.refKey, frameIndex: 0,
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
                        for ref in (try? store.searchOCR(query: g)) ?? [] {
                            counts[ref.refKey, default: 0] += 1
                        }
                    }
                    let multi = counts.filter { $0.value >= 2 }
                        .sorted { $0.value > $1.value }
                        .prefix(topK)
                    for (i, e) in multi.enumerated() {
                        add(SearchHit(kind: .photo, refKey: e.key, frameIndex: 0,
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
        guard !fused.isEmpty else { return [] }

        // 显示分 = 相对融合分(第一名 100%)
        let maxScore = fused.values.map { $0.score }.max() ?? 1.0
        let merged: [SearchHit] = fused.values.sorted { $0.score > $1.score }.map {
            SearchHit(kind: $0.hit.kind, refKey: $0.hit.refKey, frameIndex: $0.hit.frameIndex,
                      space: $0.hit.space, title: $0.hit.title, date: $0.hit.date,
                      score: Float($0.score / maxScore), color: $0.hit.color)
        }
        return resolve(merged)
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
                              topK: Int, siglip: SigLIPEmbedder?) async throws -> [DisplayHit] {
        guard scope == .all || scope == .photo else { return [] }   // v1 只作用于照片
        let terms = Self.splitTerms(spec.body)
        guard !terms.isEmpty else { return [] }

        // 文字通道:每个词独立查 OCR,连接词决定 AND/OR。
        // and 模式标题列出全部命中词(交集里每张都含全部词,不能只显示第一个)
        if spec.channel == .text {
            var perTerm: [[SearchHit]] = []
            for t in terms {
                let refs = (try? store.searchOCR(query: t)) ?? []
                let connectorIsAnd = (spec.connector == .and)
                perTerm.append(refs.prefix(60).enumerated().map { rank, ref in
                    let title = connectorIsAnd
                        ? "文字全含:\(terms.joined(separator: " ")) —「\(t)」\(Self.ocrSnippet(ref.text, t))"
                        : "文字命中「\(t)」:\(Self.ocrSnippet(ref.text, t))"
                    return SearchHit(kind: .photo, refKey: ref.refKey, frameIndex: 0, space: "ocr",
                              title: title,
                              date: nil, score: Float(1.0 / Double(rank + 1)), color: nil)
                })
            }
            return resolve(combine(perTerm, connector: spec.connector))
        }

        // 语义/自动通道:数字词→文字,其余词→视觉
        var perTerm: [[SearchHit]] = []
        for t in terms {
            if Self.containsDigit(t) {
                let refs = (try? store.searchOCR(query: t)) ?? []
                perTerm.append(refs.prefix(40).enumerated().map { rank, ref in
                    SearchHit(kind: .photo, refKey: ref.refKey, frameIndex: 0, space: "ocr",
                              title: "含「\(t)」文字", date: nil,
                              score: Float(1.0 / Double(rank + 1)), color: nil)
                })
            } else {
                perTerm.append(semanticHits(t, siglip: siglip, topK: topK))
            }
        }
        return resolve(combine(perTerm, connector: spec.connector))
    }

    /// 单词语义命中(SigLIP 视觉,门槛 0.18 + 相对尾部截断)
    private func semanticHits(_ term: String, siglip: SigLIPEmbedder?, topK: Int) -> [SearchHit] {
        guard let siglip else { return [] }
        guard let q = try? siglip.embedQuery(QueryUnderstanding.english(for: term) ?? term) else { return [] }
        var hits = Self.dedupByRef((try? store.search(
            space: siglip.space, kinds: [.photo], query: q, limit: topK, minScore: 0.18)) ?? [])
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

    private func resolve(_ hits: [SearchHit]) -> [DisplayHit] {
        let assetIDs = Set(hits.filter {
            $0.kind == .photo || $0.kind == .videoFrame || $0.kind == .photoLabel
        }.map { $0.refKey })

        var assets: [String: PHAsset] = [:]
        if !assetIDs.isEmpty {
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: Array(assetIDs), options: nil)
            fetch.enumerateObjects { asset, _, _ in assets[asset.localIdentifier] = asset }
        }

        var out: [DisplayHit] = []
        for hit in hits {
            switch hit.kind {
            case .photo, .photoLabel, .userTag, .videoFrame:
                guard let asset = assets[hit.refKey] else { continue }
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
