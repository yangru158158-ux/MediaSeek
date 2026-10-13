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
    @State private var clusters: [[String]] = []     // 每组 = 去重后的照片 refKey
    @State private var clusterScores: [[Float]] = [] // 与 clusters 对齐:成员与组锚的相似度(锚=1)
    @State private var groupNames: [String] = []     // 组的现有名字(成员标签众数,未命名时空),与 clusters 对齐
    @State private var assets: [String: PHAsset] = [:]
    @State private var selection = Set<Int>()
    @State private var mergeMode = false
    @State private var faceTotal = 0
    @State private var singletonCount = 0
    @State private var totalGroups = 0

    // 阈值(与锚的余弦):135 用户实测大杂烩组后收紧——MobileFaceNet 同人清晰照
    // 一般 ≥0.55,模糊小脸/证件翻拍质量差会把分数压到跨人区间,宁拆勿合
    private let thresholds: [String: Float] = ["exact": 0.60, "balanced": 0.50, "loose": 0.40]

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
                ToolbarItem(placement: .topBarTrailing) {
                    if !clusters.isEmpty, !scanning {
                        Button(mergeMode ? "合并所选(\(selection.count))" : "合并") { toggleMerge() }
                            .font(.footnote)
                            .disabled(mergeMode && selection.count < 2)
                    }
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
            Text("同一人的照片归为一组;给组命名后,搜索名字即可找到全部照片。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - 结果区

    private var resultsSection: some View {
        Section {
            if mergeMode {
                HStack {
                    Button("全选") { selection = Set(clusters.indices) }.font(.footnote)
                    Spacer()
                    Button("取消") { mergeMode = false; selection.removeAll() }.font(.footnote)
                }
                .buttonStyle(.plain)
            }
            ForEach(clusters.indices, id: \.self) { i in
                row(i)
            }
        } header: {
            Text(totalGroups > clusters.count
                 ? "人物 \(totalGroups) 组(显示最大 \(clusters.count) 组)"
                 : "人物 \(clusters.count) 组")
        } footer: {
            Text(singletonCount > 0
                 ? "另有 \(singletonCount) 张脸未成组(单人单照);同人被拆成多组时,用「合并」归拢。" : "")
        }
    }

    private func row(_ i: Int) -> some View {
        let refs = clusters[i]
        let named = groupNames.indices.contains(i) && !groupNames[i].isEmpty
        let title = named ? groupNames[i] : "未命名"
        // 组内最弱相似度(去掉锚=1 后的最小值):链式聚类的最弱连接,低=组可疑
        let weakest = clusterScores.indices.contains(i)
            ? clusterScores[i].dropFirst().min() : nil
        return NavigationLink {
            ClusterDetailView(refs: refs,
                              existingName: named ? groupNames[i] : "",
                              scores: clusterScores.indices.contains(i) ? clusterScores[i] : [])
        } label: {
            HStack(spacing: 12) {
                if mergeMode {
                    Image(systemName: selection.contains(i) ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(selection.contains(i) ? .blue : .secondary)
                        .onTapGesture { togglePick(i) }
                }
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
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(named ? .semibold : .regular)
                        .foregroundStyle(named ? .primary : .secondary)
                    Text("\(refs.count) 张照片\(weakest.map { String(format: " · 最弱 %.0f%%", $0 * 100) } ?? "")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private func togglePick(_ i: Int) {
        if selection.contains(i) { selection.remove(i) } else { selection.insert(i) }
    }

    private func toggleMerge() {
        if !mergeMode {
            mergeMode = true
            selection.removeAll()
        } else if selection.count >= 2 {
            var merged: [String] = []
            var mergedSc: [Float] = []
            var seen = Set<String>()
            for i in selection.sorted() {
                guard clusters.indices.contains(i) else { continue }
                let sc = clusterScores.indices.contains(i) ? clusterScores[i] : []
                for (k, r) in clusters[i].enumerated() where seen.insert(r).inserted {
                    merged.append(r)
                    mergedSc.append(sc.indices.contains(k) ? sc[k] : 0)
                }
            }
            let keep = clusters.indices.filter { !selection.contains($0) }
            let keepNames = keep.map { groupNames.indices.contains($0) ? groupNames[$0] : "" }
            let keepScores = keep.map { clusterScores.indices.contains($0) ? clusterScores[$0] : [] }
            clusters = [merged] + keep.map { clusters[$0] }
            clusterScores = [mergedSc] + keepScores
            groupNames = [""] + keepNames
            totalGroups -= selection.count - 1
            selection.removeAll()
            mergeMode = false
            app.notify("已合并为 1 组(\(merged.count) 张照片)")
        }
    }

    // MARK: - 扫描

    private func start() {
        guard !scanning else { return }
        scanning = true
        progress = (0, 1)
        let threshold = thresholds[mode] ?? 0.45
        let store = app.store
        Task.detached(priority: .userInitiated) {
            do {
                let data = try store.allFaceVectors()
                let scored = DuplicateScanner.scan(
                    keys: data.keys, flat: data.flat, dim: data.dim, threshold: threshold) { done, total in
                    Task { @MainActor in progress = (done, total) }
                }
                // 脸组 → 照片组(组内同照片多脸去重,保留与锚相似度);按人数降序,封顶 300 组防长列表
                var photoGroups: [[String]] = []
                var groupScores: [[Float]] = []
                for g in scored {
                    var refs: [String] = []
                    var scs: [Float] = []
                    var seen = Set<String>()
                    for m in g where seen.insert(data.refs[m.index]).inserted {
                        refs.append(data.refs[m.index])
                        scs.append(m.score)
                    }
                    if !refs.isEmpty { photoGroups.append(refs); groupScores.append(scs) }
                }
                photoGroups.sort { $0.count > $1.count }
                let groupedFaces = scored.reduce(0) { $0 + $1.count }
                let singles = data.refs.count - groupedFaces
                let totalGroups = photoGroups.count
                let shown = Array(photoGroups.prefix(300))
                let shownScores = Array(groupScores.prefix(300))

                // 照片资产与组名(组名 = 成员已有用户标签的众数,二次扫描自动带名)
                let shownRefs = shown.flatMap { $0 }
                var fetched: [String: PHAsset] = [:]
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: shownRefs, options: nil)
                fetch.enumerateObjects { a, _, _ in fetched[a.localIdentifier] = a }
                let tagMap = store.userTagsForRefs(Array(Set(shownRefs)))

                var outClusters: [[String]] = []
                var outScores: [[Float]] = []
                var outNames: [String] = []
                for (gi, g) in shown.enumerated() {
                    // 资产已不在相册的成员剔除(分数按下标同步)
                    let aliveIdx = g.indices.filter { fetched[g[$0]] != nil }
                    guard !aliveIdx.isEmpty else { continue }
                    outClusters.append(aliveIdx.map { g[$0] })
                    let sc = shownScores.indices.contains(gi)
                        ? aliveIdx.map { shownScores[gi].indices.contains($0) ? shownScores[gi][$0] : 0 }
                        : aliveIdx.map { _ in 0 }
                    outScores.append(sc)
                    var counts: [String: Int] = [:]
                    for r in aliveIdx.map({ g[$0] }) { for t in tagMap[r] ?? [] { counts[t, default: 0] += 1 } }
                    outNames.append(counts.max { $0.value < $1.value }?.key ?? "")
                }

                await MainActor.run {
                    scanning = false
                    progress = nil
                    clusters = outClusters
                    clusterScores = outScores
                    groupNames = outNames
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

/// 组详情:成员照片网格 + 命名(写用户标签,搜索人名即命中)
struct ClusterDetailView: View {
    let refs: [String]
    var scores: [Float] = []   // 成员与组锚的相似度(锚=1);角标展示,低于 ~50% 的可疑
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

    init(refs: [String], existingName: String, scores: [Float] = []) {
        self.refs = refs
        self.scores = scores
        _currentName = State(initialValue: existingName)
    }

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 6) {
                ForEach(Array(refs.enumerated()), id: \.element) { i, ref in
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
                        if i > 0, scores.indices.contains(i) {
                            Text(String(format: "%.0f%%", scores[i] * 100))
                                .font(.caption2).monospacedDigit()
                                .padding(.horizontal, 4).padding(.vertical, 2)
                                .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                                .foregroundStyle(scores[i] >= 0.5 ? .white : .orange)
                        } else if i == 0 {
                            Text("基准")
                                .font(.caption2)
                                .padding(.horizontal, 4).padding(.vertical, 2)
                                .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                                .foregroundStyle(.white)
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
