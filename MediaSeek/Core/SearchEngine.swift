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

    func search(_ rawQuery: String, scope: SearchScope, topK: Int = 120) async throws -> [DisplayHit] {
        let raw = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return [] }
        let query = QueryUnderstanding.core(raw)
        guard !query.isEmpty else { return [] }

        // 多词查询:「2026 电脑屏幕」→ 含数字的词走 OCR 文字匹配(AND),
        // 其余词走视觉/标签;混合查询的结果须同时满足两边
        let terms = query.components(separatedBy: CharacterSet(charactersIn: " ,、,/"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        func hasDigit(_ t: String) -> Bool {
            t.unicodeScalars.contains { $0.value >= 48 && $0.value <= 57 }
        }
        let textTerms = terms.filter(hasDigit)
        let visualTerms = terms.filter { !hasDigit($0) }
        let concept = visualTerms.isEmpty ? query : visualTerms.joined(separator: " ")

        let imageKinds: [ItemKind] = scope == .all || scope == .photo
            ? [.photo] : (scope == .video ? [.videoFrame] : [])
        // Gemma 语义只负责文件名/文件内容——照片标签的余弦虚高且区分度低
        // (「猫」vs「people, adult」和 vs「cat」差不了几分),会顶掉真命中
        let fileKinds: [ItemKind] = scope == .all || scope == .file ? [.file, .fileChunk] : []

        var channels: [(weight: Double, hits: [SearchHit])] = []
        let siglip = await MainActor.run { models.siglip }
        let gemma = await MainActor.run { models.gemma }

        // 标签语义路由(通用中文入口):任意查询 → Gemma 多语言相似度 →
        // 库内实际存在的英文标签(封闭集合,惰性建向量表)→ 再按 token 精确捞照片。
        // 只用语义"选标签",照片匹配仍然精确,零幻觉;词典只是它的兜底。
        var routedEnglish: String?
        if !imageKinds.isEmpty, let gemma {
            if let routed = Self.routeLabels(query: concept, gemma: gemma, store: store) {
                routedEnglish = routed.enQuery
                if !routed.hits.isEmpty { channels.append((1.5, routed.hits)) }
            }
        }

        // 视觉通道:SigLIP2 以英文图文对训练;词典 → 路由结果 → 原文,三级取英文
        if !imageKinds.isEmpty, let siglip {
            let q = try siglip.embedQuery(QueryUnderstanding.english(for: concept) ?? routedEnglish ?? concept)
            channels.append((1.0, Self.dedupByRef(try store.search(
                space: siglip.space, kinds: imageKinds, query: q, limit: topK, minScore: 0.12))))
        }

        // 标签精确通道:查询词(含词典英译)与 Vision 英文标签做 token 级比对,零幻觉
        if !imageKinds.isEmpty {
            let tokens = Self.queryTokens(query: concept)
            if !tokens.isEmpty, let rows = try? store.allPhotoLabels() {
                var scored: [(refKey: String, matches: Int)] = []
                for row in rows {
                    let labelTokens = Set(row.title.lowercased()
                        .components(separatedBy: CharacterSet(charactersIn: ", ")))
                    let m = tokens.intersection(labelTokens).count
                    if m > 0 { scored.append((row.refKey, m)) }
                }
                let hits = scored.sorted { $0.matches > $1.matches }.prefix(topK).map {
                    SearchHit(kind: .photo, refKey: $0.refKey, frameIndex: 0,
                              space: "label", title: nil, date: nil,
                              score: Float($0.matches), color: nil)
                }
                if !hits.isEmpty { channels.append((1.6, Array(hits))) }
            }
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

        // OCR 文字精确通道:数字词组合按 AND 求交;混合查询再与视觉结果取交集
        var ocrMust: Set<String>?
        if scope == .all || scope == .photo {
            if !textTerms.isEmpty {
                var acc: Set<String>? = nil
                for t in textTerms {
                    let set = Set(((try? store.searchOCR(query: t)) ?? []).map(\.refKey))
                    acc = (acc ?? set).intersection(set)
                    if acc?.isEmpty == true { break }
                }
                guard let acc, !acc.isEmpty else { return [] }   // AND 无解 → 诚实空
                let title = "含「\(textTerms.joined(separator: " "))」文字"
                for (i, ref) in acc.sorted().prefix(topK).enumerated() {
                    add(SearchHit(kind: .photo, refKey: ref, frameIndex: 0,
                                  space: "ocr", title: title,
                                  date: nil, score: 1.0, color: nil),
                        weight: 8.0, rank: i)
                }
                if !visualTerms.isEmpty { ocrMust = acc }   // 混合查询:视觉命中须同时含文字
            } else if query.count >= 2 {
                let refs = (try? store.searchOCR(query: query)) ?? []
                for (i, ref) in refs.enumerated() {
                    add(SearchHit(kind: .photo, refKey: ref.refKey, frameIndex: 0,
                                  space: "ocr", title: "含「\(query)」文字",
                                  date: nil, score: 1.0, color: nil),
                        weight: 8.0, rank: i)
                }
            }
        }

        for ch in channels {
            for (rank, hit) in ch.hits.enumerated() { add(hit, weight: ch.weight, rank: rank) }
        }
        guard !fused.isEmpty else { return [] }
        if let ocrMust {
            fused = fused.filter { ocrMust.contains($0.key) }
            guard !fused.isEmpty else { return [] }   // 视觉命中里没有同时含文字的 → 诚实空
        }

        // 显示分 = 相对融合分(第一名 100%)
        let maxScore = fused.values.map { $0.score }.max() ?? 1.0
        let merged: [SearchHit] = fused.values.sorted { $0.score > $1.score }.map {
            SearchHit(kind: $0.hit.kind, refKey: $0.hit.refKey, frameIndex: $0.hit.frameIndex,
                      space: $0.hit.space, title: $0.hit.title, date: $0.hit.date,
                      score: Float($0.score / maxScore), color: $0.hit.color)
        }
        return resolve(merged)
    }

    /// 查询词集合:清洗后的原文分词 + 词典英译分词,再做一跳同义扩展(供标签 token 比对)
    private static func queryTokens(query: String) -> Set<String> {
        var tokens = Set(query.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty })
        if let en = QueryUnderstanding.english(for: query) {
            for t in en.lowercased().split(separator: " ") { tokens.insert(String(t)) }
        }
        return QueryUnderstanding.expandedTokens(tokens)
    }

    /// 语义路由:取语义最近的至多 3 个库内标签(余弦 ≥ 0.60),按 token 精确捞照片;
    /// 顺带返回标签英文串给视觉通道当查询。词向量表惰性补齐,每次最多嵌 200 条。
    private static func routeLabels(query: String, gemma: GemmaTextEmbedder, store: VectorStore)
        -> (enQuery: String, hits: [SearchHit])? {
        guard let labels = try? store.allDistinctLabels(), !labels.isEmpty else { return nil }
        var vocab = (try? store.labelVocab()) ?? []
        let known = Set(vocab.map { $0.label })
        let missing = labels.filter { !known.contains($0) }.prefix(200)
        for label in missing {
            guard let v = try? gemma.embedDocument(label) else { continue }
            try? store.saveLabelVec(label, vec: v)
            vocab.append((label: label, vec: v))
        }
        guard !vocab.isEmpty, let qv = try? gemma.embedQuery(query) else { return nil }
        let picked = vocab
            .map { (label: $0.label, cos: Self.cosine($0.vec, qv)) }
            .filter { $0.cos >= 0.60 }
            .sorted { $0.cos > $1.cos }
            .prefix(3)
        guard !picked.isEmpty else { return nil }
        let chosen = Set(picked.map { $0.label })
        let tokens = Set(picked.flatMap {
            $0.label.lowercased().components(separatedBy: CharacterSet(charactersIn: ", "))
        })
        let rows = (try? store.allPhotoLabels()) ?? []
        var hits: [SearchHit] = []
        for row in rows {
            let lt = Set(row.title.lowercased().components(separatedBy: CharacterSet(charactersIn: ", ")))
            if !lt.isDisjoint(with: tokens) {
                hits.append(SearchHit(kind: .photo, refKey: row.refKey, frameIndex: 0,
                                      space: "label", title: nil, date: nil, score: 1.0, color: nil))
            }
        }
        let en = picked.map { $0.label }.sorted().joined(separator: " ")
        return (en, hits)
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        guard na > 0, nb > 0 else { return 0 }
        return dot / (na.squareRoot() * nb.squareRoot())
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
