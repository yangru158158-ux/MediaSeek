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
    @State private var groupPeople: [Int] = []       // 与 clusters 对齐:组内人物数(1=单人,2+=合影)
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
    /// 组间合并的连接阈值 = 入组阈值 − 0.06(143 用户实测效果不错,维持);
    /// 质心=全组「平均脸」:同人的各子组质心都逼近本人均值,照片特别多的人
    /// (如近 2000 张)靠它全并成一组;个别模糊/误检脸被平均掉,拦不住合并
    private func linkThreshold(_ t: Float) -> Float { max(t - 0.06, 0.30) }
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
            Text("照片按「里面是哪些人」归组:单人照归本人组,同一对/组人的合影另成一组;命名后搜索名字即可命中。")
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
                 ? "另有 \(singletonCount) 张脸未成组(单人单照)。单人照在前、合影按人数分档在后;给组合影命名(如「我和妈妈」)后,搜索名字即可命中。" : "单人照在前、合影按人数分档在后;给组合影命名后,搜索名字即可命中。")
        }
    }

    private func row(_ i: Int) -> some View {
        let refs = clusters[i]
        let named = groupNames.indices.contains(i) && !groupNames[i].isEmpty
        let title = named ? groupNames[i] : (peopleCount(i) > 1 ? "\(peopleCount(i)) 人合影" : "未命名")
        // 组内最弱相似度:该组照片与所属人物簇锚的最低相似,低=组可疑。
        // 排除锚(=1)而不是按位置 dropFirst——锚可能不在显示位
        let weakest = clusterScores.indices.contains(i)
            ? clusterScores[i].filter { $0 < 0.999 }.min() : nil
        return NavigationLink {
            ClusterDetailView(refs: refs,
                              existingName: named ? groupNames[i] : "",
                              peopleCount: peopleCount(i),
                              scores: clusterScores.indices.contains(i) ? clusterScores[i] : [])
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
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(named ? .semibold : .regular)
                        .foregroundStyle(named ? .primary : .secondary)
                    Text("\(refs.count) 张照片\(peopleCount(i) > 1 ? " · \(peopleCount(i)) 人" : "")\(weakest.map { String(format: " · 最弱 %.0f%%", $0 * 100) } ?? "")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private func peopleCount(_ i: Int) -> Int {
        groupPeople.indices.contains(i) ? max(groupPeople[i], 1) : 1
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
                // 组间合并:贪心聚类「一锚定终身」会把同一人按角度/遮挡拆成多组。
                // 组质心连接(照片多的人全并拢)+垃圾剔除(模糊/误检脸清出大组)
                let scored = DuplicateScanner.mergeGroups(
                    raw, flat: data.flat, dim: data.dim,
                    linkThreshold: link, expelFloor: expelFloor) { done, total in
                    Task { @MainActor in progress = (done, max(total, 1)) }
                }
                // 组合分组(用户 2026-10-13 提议,143 人物聚类已准):每张照片按
                // 「包含哪些人物簇」归组——只含 A 的照片归 A 组,含 A+B 的合影归
                // 「A×B」组,以此类推。合影不再混进单人组:两人合影里另一个人
                // 不是本人时,这张照片进的是「A×B」组而非 A 的组
                var photoClusters: [String: Set<Int>] = [:]
                var photoBestScore: [String: Float] = [:]   // 照片在各簇中的最高分(角标用)
                for (ci, g) in scored.enumerated() {
                    for m in g {
                        let r = data.refs[m.index]
                        photoClusters[r, default: []].insert(ci)
                        if m.score > (photoBestScore[r] ?? -1) { photoBestScore[r] = m.score }
                    }
                }
                var compo: [Set<Int>: (refs: [String], scores: [Float])] = [:]
                for (ref, cs) in photoClusters {
                    compo[cs, default: ([], [])].refs.append(ref)
                    compo[cs]!.scores.append(photoBestScore[ref] ?? 1)
                }
                // 排序:单人组在前,多人在后;同档按张数降序;封顶 300 组防长列表
                let order = compo.keys.sorted { a, b in
                    if a.count != b.count { return a.count < b.count }
                    return compo[a]!.refs.count > compo[b]!.refs.count
                }
                let shown = order.prefix(300)
                let groupedFaces = scored.reduce(0) { $0 + $1.count }
                let singles = data.refs.count - groupedFaces
                let totalGroups = compo.count

                // 照片资产与组名(组名 = 成员已有用户标签的众数,二次扫描自动带名)
                let shownRefs = shown.flatMap { compo[$0]!.refs }
                var fetched: [String: PHAsset] = [:]
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: shownRefs, options: nil)
                fetch.enumerateObjects { a, _, _ in fetched[a.localIdentifier] = a }
                let tagMap = store.userTagsForRefs(Array(Set(shownRefs)))

                var outClusters: [[String]] = []
                var outScores: [[Float]] = []
                var outNames: [String] = []
                var outPeople: [Int] = []
                for key in shown {
                    let entry = compo[key]!
                    // 资产已不在相册的成员剔除(分数按下标同步)
                    let aliveIdx = entry.refs.indices.filter { fetched[entry.refs[$0]] != nil }
                    guard !aliveIdx.isEmpty else { continue }
                    outClusters.append(aliveIdx.map { entry.refs[$0] })
                    outScores.append(aliveIdx.map { entry.scores[$0] })
                    var counts: [String: Int] = [:]
                    for r in aliveIdx.map({ entry.refs[$0] }) { for t in tagMap[r] ?? [] { counts[t, default: 0] += 1 } }
                    outNames.append(counts.max { $0.value < $1.value }?.key ?? "")
                    outPeople.append(key.count)
                }

                await MainActor.run {
                    scanning = false
                    progress = nil
                    clusters = outClusters
                    clusterScores = outScores
                    groupNames = outNames
                    groupPeople = outPeople
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
    var scores: [Float] = []   // 照片与所属人物簇锚的相似度(锚=1);角标展示,低于 ~50% 的可疑
    var peopleCount = 1        // 组合人物数(1=单人,2+=合影)
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

    init(refs: [String], existingName: String, peopleCount: Int = 1, scores: [Float] = []) {
        self.refs = refs
        self.peopleCount = peopleCount
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
                        // 角标按分数判定基准(≈1),不按下标位置——合并后锚不一定在第 0 位
                        if scores.indices.contains(i), scores[i] < 0.999 {
                            Text(String(format: "%.0f%%", scores[i] * 100))
                                .font(.caption2).monospacedDigit()
                                .padding(.horizontal, 4).padding(.vertical, 2)
                                .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                                .foregroundStyle(scores[i] >= 0.5 ? .white : .orange)
                        } else if scores.indices.contains(i) {
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
        .navigationTitle(currentName.isEmpty
                         ? (peopleCount > 1 ? "合影 \(refs.count) 张" : "人物 \(refs.count) 张")
                         : currentName)
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
