import SwiftUI
import Photos

/// 人物归组(方案B):人脸身份向量贪心聚类 → 分组浏览/命名/合并。
/// 命名 = 给组内照片写用户标签,搜索人名即可命中(复用现有 userTag 通道);
/// 检测/对齐/嵌入在索引期完成,这里只做向量聚类与展示
struct PeopleView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var mode = "balanced"
    @State private var scanning = false
    @State private var progress: (done: Int, total: Int)?
    @State private var clusters: [[String]] = []     // 每组 = 组合相同的照片 refKey
    @State private var clusterScores: [[Float]] = [] // 与 clusters 对齐:照片与所属人物簇锚的相似度
    @State private var groupSections: [[(title: String, refs: [String], scores: [Float])]] = []   // 组内分节(只有TA/与某人/…)
    @State private var withOthersCounts: [Int] = [] // 与 clusters 对齐:含他人合影的张数(行副标题)
    @State private var groupNames: [String] = []     // 组的现有名字(成员标签众数,未命名时空),与 clusters 对齐
    @State private var assets: [String: PHAsset] = [:]
    @State private var faceTotal = 0
    @State private var singletonCount = 0
    @State private var totalGroups = 0

    // 阈值(与锚的余弦,r50 ArcFace 尺度)。CI 真实照片实测(3 人各 3 张含油画
    // 肖像/不同年代):同人最低 0.484(油画×照片)、跨人最高 0.162——真实安全
    // 带是 0.16~0.48,原先 0.68/0.58/0.45 定得过高,同人难样本够不到阈值就
    // 自成一组(140 用户实测同一人分好几组)。阈值扫描 0.50 及以下全部正确,
    // 取 0.48 留 3 倍于跨人上限的余量
    private let thresholds: [String: Float] = ["exact": 0.60, "balanced": 0.48, "loose": 0.38]
    /// 组间合并的连接阈值 = 入组阈值 − 0.14(下限 0.28)。真实数据同人最低
    /// 0.48/跨人最高 0.16,中间 0.2~0.45 是空带——连接放空带下段(平衡 0.34):
    /// 跨人不可能串组(质心~0.13),同人的跨年代/跨地点子组能并拢
    /// (145 实测「相同两人因拍摄时间地点不同被拆两组」——0.42 还是偏高)
    private func linkThreshold(_ t: Float) -> Float { max(t - 0.14, 0.28) }
    /// 垃圾剔除线:合并完成后与组锚相似低于它的成员移出组(成为单例)。
    /// 真实数据同人最低 0.48(油画×照片),0.18 远在其下,不会误剔真人;
    /// 模糊/误检脸(141 的「最弱 -6%」)由此清出大组
    private let expelFloor: Float = 0.18

    var body: some View {
        NavigationStack {
            Form {
                setupSection
                if scanning, let p = progress, p.total > 0 {
                    Section {
                        ProgressView(value: Double(p.done) / Double(p.total))
                        Text("已处理 \(p.done)/\(p.total) 张脸")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if !clusters.isEmpty {
                    resultsSection
                } else if !scanning, faceTotal == 0 {
                    Section {
                        Text(emptyHint).font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("人物归组")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear { faceTotal = app.store.faceStats().faces }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("关闭") { dismiss() }
                }
            }
        }
    }

    // MARK: - 设置区

    private var emptyHint: String {
        if app.models.state(for: .face) != .ready {
            return "人脸模型未安装:请先在「设置」导入包含人脸模型(FaceEmbedder)的新模型包,再回资料库点「同步新增内容」补人脸数据。"
        }
        return "库内还没有人脸数据:回资料库点「同步新增内容」补全(约 30-60 分钟,可中断可续)。"
    }

    private var setupSection: some View {
        Section("归组设置") {
            Picker("模式", selection: $mode) {
                Text("精确").tag("exact")
                Text("平衡").tag("balanced")
                Text("宽松").tag("loose")
            }
            .pickerStyle(.segmented)
            let stats = app.store.faceStats()
            Text(faceTotal > 0
                 ? "已入库人脸 \(faceTotal) 张(\(stats.photos) 张照片)"
                 : "覆盖已索引照片(\(stats.done) 张已做人脸处理)")
                .font(.caption2).foregroundStyle(.secondary)
            Button {
                start()
            } label: {
                if scanning {
                    HStack { ProgressView(); Text("扫描中…") }
                } else {
                    Label(faceTotal < 2 ? "人脸数据不足,先同步补全" : "开始归组",
                          systemImage: "person.2")
                }
            }
            .disabled(scanning || faceTotal < 2 || app.models.state(for: .face) != .ready)
            Text("同一人的照片归为一组,点开按「只有TA/与某人/与更多人」分节;命名后搜索名字即命中。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - 结果区

    private var resultsSection: some View {
        Section {
            ForEach(clusters.indices, id: \.self) { i in
                row(i)
            }
        } header: {
            Text(totalGroups > clusters.count
                 ? "人物 \(totalGroups) 组(显示最大 \(clusters.count) 组)"
                 : "人物 \(totalGroups) 组")
        } footer: {
            Text(singletonCount > 0
                 ? "另有 \(singletonCount) 张脸未成组(单人单照)。组行显示含他人合影张数,点开看分节;命名后搜索名字即命中。" : "组行显示含他人合影张数,点开看分节;命名后搜索名字即命中。")
        }
    }

    private func row(_ i: Int) -> some View {
        let refs = clusters[i]
        let named = groupNames.indices.contains(i) && !groupNames[i].isEmpty
        let others = withOthersCounts.indices.contains(i) ? withOthersCounts[i] : 0
        // 组内最弱相似度:该组照片与所属人物簇锚的最低相似,低=组可疑
        let weakest = clusterScores.indices.contains(i)
            ? clusterScores[i].filter { $0 < 0.999 }.min() : nil
        return NavigationLink {
            ClusterDetailView(refs: refs,
                              existingName: named ? groupNames[i] : "",
                              sections: groupSections.indices.contains(i) ? groupSections[i] : [])
        } label: {
            HStack(spacing: 12) {
                Group {
                    if let a = assets[refs.first ?? ""] {
                        AssetThumbView(asset: a, pixel: 160)
                    } else {
                        Rectangle().fill(Color(.systemGray5))
                    }
                }
                .frame(width: 52, height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                VStack(alignment: .leading, spacing: 3) {
                    Text(named ? groupNames[i] : "未命名")
                        .font(.subheadline)
                        .fontWeight(named ? .semibold : .regular)
                        .foregroundStyle(named ? .primary : .secondary)
                    Text("\(refs.count) 张照片\(others > 0 ? " · \(others) 张含他人" : "")\(weakest.map { String(format: " · 最弱 %.0f%%", $0 * 100) } ?? "")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - 扫描

    private func start() {
        guard !scanning else { return }
        // 守卫:人脸模型换了但脸数据还没按新模型重嵌——此时聚类的是旧模型向量,
        // 新旧嵌入空间不通用,聚出来的必然混人。重建中(库被清空重嵌)不拦,
        // 聚的是已完成部分的新向量
        let want = UserDefaults.standard.integer(forKey: "faceModelSize")
        let indexed = UserDefaults.standard.integer(forKey: "faceIndexedSize")
        if want > 0, want != indexed, app.store.faceStats().faces > 0 {
            app.notify("人脸模型已更换,旧数据还没重建完:回资料库等「同步」跑完再归组(约 30-60 分钟,可中断可续)")
            return
        }
        scanning = true
        progress = (0, 1)
        let threshold = thresholds[mode] ?? 0.38
        let link = linkThreshold(threshold)
        let store = app.store
        Task.detached(priority: .userInitiated) {
            do {
                let data = try store.allFaceVectors()
                let raw = DuplicateScanner.scan(
                    keys: data.keys, flat: data.flat, dim: data.dim, threshold: threshold) { done, total in
                    Task { @MainActor in progress = (done, total) }
                }
                // 组间合并:贪心聚类「一锚定终身」会把同一人按角度/年代拆成多组
                // (145 实测相同两人因拍摄时间不同裂成两簇)。组质心连接+离群
                // 照片吸收+垃圾剔除:连接降到空带下段,跨年代子组与离群照并拢
                let groupedIdx = Set(raw.flatMap { $0.map(\.index) })
                let singletons = (0..<data.keys.count).filter { !groupedIdx.contains($0) }
                let scored = DuplicateScanner.mergeGroups(
                    raw, flat: data.flat, dim: data.dim,
                    linkThreshold: link, expelFloor: expelFloor,
                    singletons: singletons) { done, total in
                    Task { @MainActor in progress = (done, max(total, 1)) }
                }
                // 人物为组 + 组内合影分节(145 组合分组实测:组合当分组键太脆——
                // 多检/漏检一张脸、同人跨年代裂成两簇,组合就变,组数爆炸;用户
                // 2026-10-13 反馈「单人混进合影/相同两人被拆/组数特别多」)。
                // 顶层一行=一个人(组数有界);详情按「只有TA/与某人/与另两人…」
                // 分节:单人绝不混进合影,相同的两个人天然聚在同一节,识别稍有
                // 出入只影响个别照片的落节,不会裂出新组
                var photoClusters: [String: Set<Int>] = [:]
                var clusterPhotos: [[(ref: String, score: Float)]] = []
                for (ci, g) in scored.enumerated() {
                    var best: [String: Float] = [:]
                    var order: [String] = []
                    for m in g {
                        let r = data.refs[m.index]
                        if best[r] == nil { order.append(r) }
                        if m.score > (best[r] ?? -1) { best[r] = m.score }
                        photoClusters[r, default: []].insert(ci)
                    }
                    clusterPhotos.append(order.map { ($0, best[$0] ?? 1) })
                }
                let clusterOrder = scored.indices.sorted { clusterPhotos[$0].count > clusterPhotos[$1].count }
                let shown = clusterOrder.prefix(300)
                let groupedFaces = scored.reduce(0) { $0 + $1.count }
                let singles = data.refs.count - groupedFaces
                let totalGroups = scored.count

                // 照片资产与簇名(簇名 = 簇内照片已有用户标签的众数,二次扫描自动带名)
                let shownRefs = Array(Set(shown.flatMap { clusterPhotos[$0].map { $0.ref } }))
                var fetched: [String: PHAsset] = [:]
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: shownRefs, options: nil)
                fetch.enumerateObjects { a, _, _ in fetched[a.localIdentifier] = a }
                let tagMap = store.userTagsForRefs(shownRefs)
                var clusterName: [Int: String] = [:]
                for ci in shown {
                    var counts: [String: Int] = [:]
                    for p in clusterPhotos[ci] { for t in tagMap[p.ref] ?? [] { counts[t, default: 0] += 1 } }
                    clusterName[ci] = counts.max { $0.value < $1.value }?.key ?? ""
                }

                var outClusters: [[String]] = []
                var outScores: [[Float]] = []
                var outNames: [String] = []
                var outSections: [[(title: String, refs: [String], scores: [Float])]] = []
                var outWithOthers: [Int] = []
                for ci in shown {
                    let alive = clusterPhotos[ci].filter { fetched[$0.ref] != nil }
                    guard !alive.isEmpty else { continue }
                    outClusters.append(alive.map { $0.ref })
                    outScores.append(alive.map { $0.score })
                    outNames.append(clusterName[ci] ?? "")
                    // 分节:只有TA / 与某一人(按同框对象分节) / 与另两人 / 与更多人
                    var onlyR: [String] = []; var onlyS: [Float] = []
                    var twoR: [String] = []; var twoS: [Float] = []
                    var manyR: [String] = []; var manyS: [Float] = []
                    var partner: [Int: ([String], [Float])] = [:]
                    for p in alive {
                        let others = (photoClusters[p.ref] ?? []).subtracting([ci])
                        switch others.count {
                        case 0: onlyR.append(p.ref); onlyS.append(p.score)
                        case 1:
                            let k = others.first!
                            partner[k, default: ([], [])].0.append(p.ref)
                            partner[k]!.1.append(p.score)
                        case 2: twoR.append(p.ref); twoS.append(p.score)
                        default: manyR.append(p.ref); manyS.append(p.score)
                        }
                    }
                    var sections: [(title: String, refs: [String], scores: [Float])] = []
                    if !onlyR.isEmpty { sections.append(("只有 TA(\(onlyR.count))", onlyR, onlyS)) }
                    let sortedPartners = partner.sorted { $0.value.0.count > $1.value.0.count }
                    for (k, v) in sortedPartners.prefix(12) {
                        let n = clusterName[k].flatMap { $0.isEmpty ? nil : $0 } ?? "未命名人物"
                        sections.append(("与 \(n)(\(v.0.count))", v.0, v.1))
                    }
                    if sortedPartners.count > 12 {   // 同框对象过多:节封顶,余量并入「与更多人」
                        for (_, v) in sortedPartners.dropFirst(12) { manyR += v.0; manyS += v.1 }
                    }
                    if !twoR.isEmpty { sections.append(("与另 2 人(\(twoR.count))", twoR, twoS)) }
                    if !manyR.isEmpty { sections.append(("与更多人(\(manyR.count))", manyR, manyS)) }
                    outSections.append(sections)
                    outWithOthers.append(alive.count - onlyR.count)
                }

                await MainActor.run {
                    scanning = false
                    progress = nil
                    clusters = outClusters
                    clusterScores = outScores
                    groupNames = outNames
                    groupSections = outSections
                    withOthersCounts = outWithOthers
                    assets = fetched
                    singletonCount = singles
                    self.totalGroups = totalGroups
                    faceTotal = data.refs.count
                    if clusters.isEmpty {
                        app.notify("没有找到可归组的人脸(试试「宽松」模式)")
                    }
                }
            } catch {
                await MainActor.run { scanning = false; app.fail(error) }
            }
        }
    }
}

/// 人物详情:照片按「只有TA / 与某人 / 与另两人 / 与更多人」分节,
/// 命名写用户标签(搜索人名即命中)
struct ClusterDetailView: View {
    let refs: [String]          // 该人物全部照片(命名打标签的范围)
    let sections: [(title: String, refs: [String], scores: [Float])]
    @EnvironmentObject var app: AppModel

    @State private var assets: [String: PHAsset] = [:]
    @State private var naming = false
    @State private var nameText = ""
    @State private var applying = false
    @State private var currentName: String
    @State private var selectedHit: DisplayHit?

    private let columns = [GridItem(.flexible(), spacing: 6),
                           GridItem(.flexible(), spacing: 6),
                           GridItem(.flexible(), spacing: 6)]

    init(refs: [String], existingName: String,
         sections: [(title: String, refs: [String], scores: [Float])]) {
        self.refs = refs
        self.sections = sections
        _currentName = State(initialValue: existingName)
    }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(sections.indices, id: \.self) { si in
                    let sec = sections[si]
                    Text(sec.title)
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 2)
                    LazyVGrid(columns: columns, spacing: 6) {
                        ForEach(Array(sec.refs.enumerated()), id: \.element) { i, ref in
                            cell(ref: ref,
                                 score: sec.scores.indices.contains(i) ? sec.scores[i] : 1)
                        }
                    }
                }
            }
            .padding(12)
        }
        .navigationTitle(currentName.isEmpty ? "人物 \(refs.count) 张" : currentName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(applying ? "命名中…" : "命名") {
                    nameText = currentName
                    naming = true
                }
                .disabled(applying)
            }
        }
        .alert("命名这组人物", isPresented: $naming) {
            TextField("人名,如 张三", text: $nameText)
            Button("命名(\(refs.count) 张照片打标签)") { applyName() }
            Button("取消", role: .cancel) {}
        } message: {
            Text("给组内每张照片写入同名标签;之后在搜索页输入该名字即可全部找到")
        }
        .sheet(item: $selectedHit) { hit in
            DetailSheet(hit: hit)
        }
        .onAppear {
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: refs, options: nil)
            var map: [String: PHAsset] = [:]
            fetch.enumerateObjects { a, _, _ in map[a.localIdentifier] = a }
            assets = map
        }
    }

    private func cell(ref: String, score: Float) -> some View {
        Group {
            if let a = assets[ref] {
                AssetThumbView(asset: a, pixel: 240)
                    .frame(height: 124)
                    .clipped()
                    .onTapGesture {
                        selectedHit = DisplayHit(
                            id: "face-\(ref)", refKey: ref, target: .asset(a),
                            kind: .photo,
                            title: "人物:\(currentName.isEmpty ? "未命名" : currentName)",
                            score: 1, date: a.creationDate, color: nil)
                    }
            } else {
                Rectangle().fill(Color(.systemGray5)).frame(height: 124)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(alignment: .bottomLeading) {
            // 角标按分数判定基准(≈1),不按下标位置
            if score < 0.999 {
                Text(String(format: "%.0f%%", score * 100))
                    .font(.caption2).monospacedDigit()
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                    .foregroundStyle(score >= 0.5 ? .white : .orange)
            } else {
                Text("基准")
                    .font(.caption2)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                    .foregroundStyle(.white)
            }
        }
    }

    private func applyName() {
        let name = nameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !applying else { return }
        applying = true
        let refs = self.refs
        Task {
            do {
                for ref in refs {
                    try await app.addUserTag(refKey: ref, tag: name)
                }
                await MainActor.run {
                    applying = false
                    currentName = name
                    app.notify("已给 \(refs.count) 张照片打上「\(name)」标签,搜索「\(name)」即可找到")
                }
            } catch {
                await MainActor.run { applying = false; app.fail(error) }
            }
        }
    }
}
