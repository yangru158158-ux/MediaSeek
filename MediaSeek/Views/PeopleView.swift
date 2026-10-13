import SwiftUI
import Photos

/// 人物组合分组:①人脸向量聚类确定「总共几个人」;②每张照片按检出脸
/// 归出「主角是哪几个人」的集合;③照片归入且只归入该集合对应的组——
/// abc 三个不同的人,至多 {a}{b}{c}{a×b}{a×c}{b×c}{a×b×c} 七组(用户定稿
/// 的本体论;146 的「人物为组+组内分节」把合影同时算进主角组和分节,
/// 照片被重复归属,用户实测推翻)。
/// 防碎机制:只出现过一两次的路人脸/误检脸不成「人物」,不参与组合键
/// (「判断一张照片主角是哪几个人」——路人不是主角),组合数因此有界。
/// 命名 = 给「这个人」出现的全部照片(含合照)写同名标签,搜索名字即命中
struct ComboGroup: Identifiable {
    let id: Int
    let members: [Int]        // 主角人物簇 id;空 = 全是路人/未识别脸
    let refs: [String]
    let scores: [Float]       // 每张照片在其主角簇内的最高相似度
}

struct PeopleView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var mode = "balanced"
    @State private var scanning = false
    @State private var progress: (done: Int, total: Int)?
    @State private var groups: [ComboGroup] = []
    @State private var personNames: [Int: String] = [:]   // 簇 → 显示名(标签覆盖达标)
    @State private var personRefs: [Int: [String]] = [:]  // 簇 → 该人全部照片(命名打标签范围)
    @State private var assets: [String: PHAsset] = [:]
    @State private var faceTotal = 0
    @State private var totalPersons = 0
    @State private var unknownCount = 0
    @State private var showingSmall = false
    @State private var confirmClearTags = false

    // 阈值(与锚的余弦,r50 ArcFace 尺度)。CI 真实照片实测(3 人各 3 张含油画
    // 肖像/不同年代):同人最低 0.484(油画×照片)、跨人最高 0.162——真实安全
    // 带是 0.16~0.48,原先 0.68/0.58/0.45 定得过高,同人难样本够不到阈值就
    // 自成一组(140 用户实测同一人分好几组)。阈值扫描 0.50 及以下全部正确,
    // 取 0.48 留 3 倍于跨人上限的余量
    private let thresholds: [String: Float] = ["exact": 0.60, "balanced": 0.48, "loose": 0.38]
    /// 组间合并的连接阈值 = 入组阈值 − 0.14(下限 0.28)。真实数据同人最低
    /// 0.48/跨人最高 0.16,中间 0.2~0.45 是空带——连接放空带下段(平衡 0.34):
    /// 跨人不可能串组,同人的跨年代/跨地点子组能并拢
    private func linkThreshold(_ t: Float) -> Float { max(t - 0.14, 0.28) }
    /// 垃圾剔除线:合并完成后与组锚相似低于它的成员移出组(成为单例)。
    /// 真实数据同人最低 0.48(油画×照片),0.18 远在其下,不会误剔真人;
    /// 模糊/误检脸(141 的「最弱 -6%」)由此清出大组
    private let expelFloor: Float = 0.18
    /// 「人物」门槛:簇至少覆盖 3 张不同照片。更小的簇是一次性路人/海报误检,
    /// 不参与组合键——这是组合数不被噪声撑爆的关键(145 组合分组的教训)
    private let personMinPhotos = 3
    /// 簇内内聚底线:成员对簇质心(平均脸)的平均相似度低于它 → 整簇解散。
    /// 148 实测打印脸/模糊脸被质心连接链式串成「假人物」巨组(第三组 154 张
    /// 里婴儿/明星/界面截屏全有);真人物该均值 ~0.6-0.75,假簇 ~0.3-0.45
    private let coherenceFloor: Float = 0.45
    /// 人名认定的标签票数门槛:≥3 张 且 过半。148 实测教训:只看覆盖率时,
    /// 两三张的小簇带 1 张污染标签=覆盖率 100%,满屏一人组都叫程玉婷/杨如
    /// (146 命名中断留下的错标签);有最低票数后小簇一律「未命名」
    private func personNameCoverage(_ n: Int, _ total: Int) -> Bool { n >= 3 && n * 2 >= total }

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
                if !groups.isEmpty {
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
            .confirmationDialog("确定清除全部人名标签?", isPresented: $confirmClearTags,
                                titleVisibility: .visible) {
                Button("清除全部人名标签", role: .destructive) { clearTags() }
                Button("取消", role: .cancel) {}
            } message: {
                Text("删除现有的人物命名(分组结构不受影响,组名全部变回「未命名」);确认单人分组无误后再重新命名")
            }
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
            Button(role: .destructive) {
                confirmClearTags = true
            } label: {
                Label("清除全部人名标签", systemImage: "tag.slash")
            }
            Text("先聚类确定总共几个人,再按每张照片「主角是哪几个人」精确归组:abc 三个人至多 a、b、c、a×b、a×c、b×c、a×b×c 七组;合影只出现在组合组,不再算进单人组。命名后搜索名字即命中。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - 结果区

    private var resultsSection: some View {
        Section {
            ForEach(majorGroups) { g in
                row(g)
            }
            if !minorGroups.isEmpty {
                Button {
                    showingSmall.toggle()
                } label: {
                    Label(showingSmall ? "收起更小的组"
                                       : "还有 \(minorGroups.count) 个不足 \(foldThreshold) 张的小组",
                          systemImage: showingSmall ? "chevron.up" : "chevron.down")
                        .font(.subheadline)
                }
            }
            if showingSmall {
                ForEach(minorGroups) { g in
                    row(g)
                }
            }
        } header: {
            Text("人物 \(totalPersons) 人 · \(groups.count) 组")
        } footer: {
            Text(unknownCount > 0
                 ? "主角只出现一两次的路人不参与分组;全部主角都无法识别的 \(unknownCount) 张在「未识别人物」。不足 \(foldThreshold) 张的组默认折叠。命名后搜索名字即命中。"
                 : "主角只出现一两次的路人不参与分组。不足 \(foldThreshold) 张的组默认折叠。点开组可为其中任一人物命名,命名后搜索名字即命中。")
        }
    }

    /// 少于 10 张的组默认折叠隐藏(用户指定),主列表只看大组
    private let foldThreshold = 10
    private var majorGroups: [ComboGroup] { groups.filter { $0.refs.count >= foldThreshold } }
    private var minorGroups: [ComboGroup] { groups.filter { $0.refs.count < foldThreshold } }

    private func personDisplay(_ ci: Int) -> String {
        let n = personNames[ci] ?? ""
        return n.isEmpty ? "未命名" : n
    }

    private func groupTitle(_ g: ComboGroup) -> String {
        guard !g.members.isEmpty else { return "未识别人物" }
        let parts = g.members.map(personDisplay)
        return parts.count == 1 ? parts[0] : parts.joined(separator: " × ")
    }

    private func row(_ g: ComboGroup) -> some View {
        // 缩略图取组内相似度最高的一张(最标准的正面脸)
        let bestIdx = g.scores.indices.max(by: { g.scores[$0] < g.scores[$1] }) ?? 0
        let thumbRef = g.refs.indices.contains(bestIdx) ? g.refs[bestIdx] : g.refs.first
        let weakest = g.scores.filter { $0 < 0.999 }.min()
        return NavigationLink {
            ClusterDetailView(members: g.members,
                              refs: g.refs, scores: g.scores,
                              personNames: $personNames, personRefs: personRefs,
                              title: groupTitle(g))
        } label: {
            HStack(spacing: 12) {
                Group {
                    if let a = thumbRef.flatMap({ assets[$0] }) {
                        AssetThumbView(asset: a, pixel: 160)
                    } else {
                        Rectangle().fill(Color(.systemGray5))
                    }
                }
                .frame(width: 52, height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                VStack(alignment: .leading, spacing: 3) {
                    Text(groupTitle(g))
                        .font(.subheadline)
                        .fontWeight(g.members.count == 1 ? .semibold : .regular)
                        .foregroundStyle(g.members.count == 1 ? .primary : .secondary)
                    Text("\(g.members.count > 1 ? "\(g.members.count) 人合影 · " : "")\(g.refs.count) 张照片\(weakest.map { String(format: " · 最弱 %.0f%%", $0 * 100) } ?? "")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - 扫描

    /// 清除全部人名标签(取消命名):聚类只看 face 向量,组结构不变,
    /// 本地组名清空即回到「未命名」;重新归组也不受影响
    private func clearTags() {
        do {
            try app.store.deleteAllUserTags()
            personNames = [:]
            app.notify("已清除全部人名标签;确认单人分组无误后,再逐组重新命名")
        } catch {
            app.fail(error)
        }
    }

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
                // 组间合并:贪心聚类「一锚定终身」会把同一人按角度/年代拆成多组。
                // 组质心连接+离群照片吸收+垃圾剔除(143 定稿)
                let groupedIdx = Set(raw.flatMap { $0.map(\.index) })
                let singletons = (0..<data.keys.count).filter { !groupedIdx.contains($0) }
                let scored = DuplicateScanner.mergeGroups(
                    raw, flat: data.flat, dim: data.dim,
                    linkThreshold: link, expelFloor: expelFloor,
                    singletons: singletons, coherenceFloor: coherenceFloor) { done, total in
                    Task { @MainActor in progress = (done, max(total, 1)) }
                }

                // ① 簇 → 照片集合(同照片多脸去重)与每照片最高分
                var refClusters: [String: Set<Int>] = [:]
                var clusterBest: [Int: [String: Float]] = [:]
                for (ci, g) in scored.enumerated() {
                    for m in g {
                        let r = data.refs[m.index]
                        refClusters[r, default: []].insert(ci)
                        if m.score > (clusterBest[ci]?[r] ?? -1) {
                            clusterBest[ci, default: [:]][r] = m.score
                        }
                    }
                }
                var clusterRefSets: [Int: Set<String>] = [:]
                for (ci, best) in clusterBest { clusterRefSets[ci] = Set(best.keys) }
                // ② 谁是「人物」:簇覆盖 ≥3 张不同照片;更小簇是一次性路人/误检脸
                let persons = Set(clusterRefSets.filter { $0.value.count >= personMinPhotos }.keys)
                // ③ 照片按主角集合精确归组:属于且只属于「主角是哪几个人」对应组
                var comboMap: [Set<Int>: (refs: [String], scores: [Float])] = [:]
                var unknownRefs: [String] = []
                var unknownScores: [Float] = []
                for (r, cs) in refClusters {
                    let key = cs.intersection(persons)
                    let best = cs.compactMap { clusterBest[$0]?[r] }.max() ?? Float(1)
                    if key.isEmpty {
                        unknownRefs.append(r); unknownScores.append(best)
                    } else {
                        var entry = comboMap[key] ?? ([], [])
                        entry.refs.append(r)
                        entry.scores.append(best)
                        comboMap[key] = entry
                    }
                }
                var ordered: [(members: [Int], refs: [String], scores: [Float])] =
                    comboMap.map { (Array($0.key).sorted(), $0.value.refs, $0.value.scores) }
                // 按组内照片数降序(用户指定);未识别组垫底
                ordered.sort { $0.refs.count > $1.refs.count }
                if !unknownRefs.isEmpty {
                    ordered.append(([], unknownRefs, unknownScores))
                }
                // 已删除照片不出场(资产取不到的顺带剔除),空组丢弃
                let allRefs = Array(Set(ordered.flatMap { $0.refs }))
                var fetched: [String: PHAsset] = [:]
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: allRefs, options: nil)
                fetch.enumerateObjects { a, _, _ in fetched[a.localIdentifier] = a }
                // ④ 人名:该人照片里覆盖率 ≥40% 的标签众数(无门槛众数会污染,147 教训)
                let tagMap = store.userTagsForRefs(allRefs)
                var names: [Int: String] = [:]
                var refsOf: [Int: [String]] = [:]
                for ci in persons {
                    let refs = Array(clusterRefSets[ci] ?? []).sorted()
                    refsOf[ci] = refs
                    var counts: [String: Int] = [:]
                    for r in refs { for t in tagMap[r] ?? [] { counts[t, default: 0] += 1 } }
                    if let best = counts.max(by: { $0.value < $1.value }),
                       personNameCoverage(best.value, refs.count) {
                        names[ci] = best.key
                    }
                }
                var outGroups: [ComboGroup] = []
                for (i, o) in ordered.enumerated() {
                    let alive = o.refs.filter { fetched[$0] != nil }
                    guard !alive.isEmpty else { continue }
                    let aliveSet = Set(alive)
                    let scores = o.scores.enumerated()
                        .filter { aliveSet.contains(o.refs[$0.offset]) }.map(\.element)
                    outGroups.append(ComboGroup(id: i, members: o.members,
                                                refs: alive, scores: scores))
                }
                let unknownAlive = outGroups.first { $0.members.isEmpty }?.refs.count ?? 0

                await MainActor.run {
                    scanning = false
                    progress = nil
                    groups = outGroups
                    personNames = names
                    personRefs = refsOf
                    assets = fetched
                    totalPersons = persons.count
                    unknownCount = unknownAlive
                    faceTotal = data.refs.count
                    if outGroups.isEmpty {
                        app.notify("没有找到可归组的人脸(试试「宽松」模式)")
                    }
                }
            } catch {
                await MainActor.run { scanning = false; app.fail(error) }
            }
        }
    }
}

/// 组合组详情:照片网格 + 「本组人物」命名区。
/// 命名对象是「人」不是「组」:给该人物出现的全部照片(含其他合照)打同名
/// 标签,搜索名字即命中全部;147 的教训是不能按组名显示(标签众数会污染)
struct ClusterDetailView: View {
    let members: [Int]
    let refs: [String]
    let scores: [Float]
    @Binding var personNames: [Int: String]
    let personRefs: [Int: [String]]
    let title: String

    @EnvironmentObject var app: AppModel
    @State private var assets: [String: PHAsset] = [:]
    @State private var naming = false            // 146 真机验证过的纯 Bool 弹出方式
    @State private var namingMember: Int?
    @State private var nameText = ""
    @State private var applying = false
    @State private var selectedHit: DisplayHit?

    private let columns = [GridItem(.flexible(), spacing: 6),
                           GridItem(.flexible(), spacing: 6),
                           GridItem(.flexible(), spacing: 6)]

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                if !members.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("本组人物")
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(.secondary)
                        ForEach(members, id: \.self) { mi in
                            Button {
                                guard !applying else { return }
                                nameText = personNames[mi] ?? ""
                                namingMember = mi
                                naming = true
                            } label: {
                                HStack {
                                    Image(systemName: "person.crop.circle")
                                    Text(displayName(mi))
                                        .fontWeight(personNames[mi] != nil ? .semibold : .regular)
                                    Spacer()
                                    Text(applying && namingMember == mi ? "命名中…" : "命名 >")
                                        .font(.footnote)
                                        .foregroundStyle(.blue)
                                }
                            }
                            .disabled(applying)
                        }
                    }
                    .padding(.horizontal, 2)
                }
                LazyVGrid(columns: columns, spacing: 6) {
                    ForEach(Array(refs.enumerated()), id: \.element) { i, ref in
                        cell(ref: ref,
                             score: scores.indices.contains(i) ? scores[i] : 1)
                    }
                }
            }
            .padding(12)
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // 单人组的命名主入口放工具栏(146 同款位置,醒目);
            // 组合组从「本组人物」逐行命名
            if members.count == 1 {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(applying ? "命名中…" : "命名") {
                        nameText = personNames[members[0]] ?? ""
                        namingMember = members[0]
                        naming = true
                    }
                    .disabled(applying)
                }
            }
        }
        .alert("命名这个人物", isPresented: $naming) {
            TextField("人名,如 张三", text: $nameText)
            Button("命名") { applyName() }
            Button("取消", role: .cancel) {}
        } message: {
            let n = namingMember.flatMap { personRefs[$0] }?.count ?? 0
            Text("给该人物出现的全部 \(n) 张照片(含其他合照)写入同名标签;搜索该名字即可全部找到")
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

    private func displayName(_ ci: Int) -> String {
        let n = personNames[ci] ?? ""
        return n.isEmpty ? "未命名" : n
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
                            title: "人物:\(title)",
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
        guard let mi = namingMember, !applying else { return }
        let name = nameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        applying = true
        let refs = personRefs[mi] ?? []
        Task {
            do {
                // 批量通道:向量只嵌一次(此前逐张推理,命名两千张要几十分钟)
                try await app.addUserTag(name, toRefs: refs)
                await MainActor.run {
                    applying = false
                    personNames[mi] = name
                    namingMember = nil
                    app.notify("已给 \(refs.count) 张照片打上「\(name)」标签,搜索「\(name)」即可找到")
                }
            } catch {
                await MainActor.run { applying = false; app.fail(error) }
            }
        }
    }
}
