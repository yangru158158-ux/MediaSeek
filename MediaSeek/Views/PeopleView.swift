import SwiftUI
import Photos
import Accelerate
import UIKit

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
    /// 该组照片都有「主角级但认不出是谁」的同行脸(侧脸/模糊,身份匹配不可行)。
    /// 单人组只放真单人照(用户三轮定稿:认出的进组合组,认不出的进「与其他人」,
    /// 纯背景小脸不算)。157 教训:必须逐张分流,不能组级打标
    var withOthers: Bool = false
}

/// 归组结果快照(落库 people_results 表)。人名不存——水化时按当前标签重推,
/// 重命名/清标签后重开面板即自动生效;指纹随快照存,人脸数据一变作废
private struct PeopleSnapshot: Codable {
    struct Group: Codable {
        var members: [Int]
        var refs: [String]
        var scores: [Float]
        var withOthers: Bool
    }
    var groups: [Group]
    var personRefs: [Int: [String]]
    var totalPersons: Int
    var unknownCount: Int
    var savedAt: Date?   // 旧快照无此字段,可选保证可解码
    /// 每张keyed照片的归类理由 [ref: [reason, note]](165 起;旧快照无此字段=nil,
    /// 单人组详情页据此标注「为什么算单人」)
    var reasons: [String: [Int]]?
    /// 逐脸匹配明细(166 起):ref → 「共N脸: c3@0.65>c7@0.21S | …」,
    /// 长按照片可读,把每张合影漏进单人组的裁决依据摆到台面上
    var details: [String: String]?
}

/// 归组扫描共享状态:挂在 AppModel 一份,视图只观察。
/// 160 的教训:结果放在 sheet 的 @State 里,用户长扫描期间关面板等全局弹窗
/// (两轮实测都是这个习惯),完成时结果写进已销毁的视图,重开面板=全新空视图,
/// 「扫描完没有结果生成」。搬进共享对象后:关了再开照常显示进度/结果;
/// 跨视图实例的防重复扫描守卫也才真正生效(@State 私有,守不住新实例)
@MainActor
final class PeopleModel: ObservableObject {
    @Published var scanning = false
    @Published var progress: (done: Int, total: Int)?
    @Published var groups: [ComboGroup] = []
    @Published var personNames: [Int: String] = [:]   // 簇 → 显示名(标签覆盖达标)
    @Published var personRefs: [Int: [String]] = [:]  // 簇 → 该人全部照片(命名打标签范围)
    @Published var assets: [String: PHAsset] = [:]
    @Published var totalPersons = 0
    @Published var unknownCount = 0
    /// 当前结果的产生时间(扫描完成/快照水化),结果页顶部标注——
    /// 没有它,自动恢复的结果看起来像"凭空冒出"(163 用户反馈)
    @Published var lastScanAt: Date?
    /// 每张照片的归类理由(165 起):单人组详情页标注「为什么算单人」——
    /// 0 全部强匹配 / 1 弱分回退(note=被吞脸最低分×100) / 2 iCloud取图失败 /
    /// 3 复检未检出人脸。用户报"合影还在单人组"时一眼定位到哪条路径
    @Published var reasons: [String: (reason: Int, note: Int)] = [:]
    /// 逐脸匹配明细(166 起,与 reasons 同源):长按照片显示每张脸的匹配分与裁决
    @Published var details: [String: String] = [:]
}

struct PeopleView: View {
    @EnvironmentObject var app: AppModel

    var body: some View { PeoplePanel(m: app.people) }
}

struct PeoplePanel: View {
    @ObservedObject var m: PeopleModel
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var faceTotal = 0
    @State private var showingSmall = false
    @State private var confirmClearTags = false
    @State private var blockedCount = 0

    // 精确档固定参数(153 用户定稿:去掉三档,只要精确;拿不准一律进不确定组)。
    // r50 ArcFace 尺度,CI 真实照片实测:同人最低 0.484、跨人最高 0.162,
    // 中间 0.2~0.45 是空带
    private let threshold: Float = 0.60                     // 入组(对锚)
    private var link: Float { max(threshold - 0.14, 0.28) } // 组间质心连接(空带下段)
    /// 垃圾剔除线:合并完成后与组锚相似低于它的成员移出组(成为单例);
    /// 模糊/误检脸(141 的「最弱 -6%」)由此清出大组
    private let expelFloor: Float = 0.18
    /// 逐脸核验底线(对簇质心/平均脸)= 真实同人下限 0.484 取整:
    /// 低于它 = 比最难的真同人样本还不像,只能是别人/打印脸/误检 → 不确定组
    private let verifyFloor: Float = 0.48
    /// 「人物」门槛:簇至少覆盖 3 张不同照片。更小的簇是一次性路人/海报误检,
    /// 不参与组合键——这是组合数不被噪声撑爆的关键(145 组合分组的教训)
    private let personMinPhotos = 3
    /// 簇内内聚底线:成员对簇质心(平均脸)的平均相似度过低 → 整簇解散
    /// (簇级兜底;148 实测打印脸被质心连接链式串成「假人物」巨组的教训)
    private let coherenceFloor: Float = 0.45
    /// 簇内双峰拆分线(第十二批):两个长相相近的人(实测两个从未同框的
    /// 小孩)被质心连接并成一簇,混合质心落在两人中间、verify/coherence
    /// 全放行——对每簇做余弦 2-均值,两半质心相似度低于该线才拆。
    /// 同人按姿态/年代分的两半质心都逼近本人平均脸(≥0.65+);两个相近的
    /// 人各为一半约 0.5。误拆(一个真人被撕成两簇)→升 0.60~0.62;相近的
    /// 人仍并在一起 → 降 0.55。**第二十二批 0.58→0.62:用户实测程玉婷被
    /// 撕成 3+ 碎片(三个「程玉婷 与其他人」组并存 434/25/21),收紧**
    private let splitFloor: Float = 0.62
    /// 人名认定门槛(≥3 票且过半)在 deriveNames 里:扫描完成与快照水化
    /// 共用同一判据。148 教训:只看覆盖率时,小簇带 1 张污染标签=覆盖率 100%

    var body: some View {
        NavigationStack {
            Form {
                // 结果置顶:扫完当场/重开面板,第一屏就是结果列表——结果排在
                // 设置区下面时,完成后停在设置区视角的用户会以为"没有结果"
                if !m.groups.isEmpty {
                    resultsSection
                } else if !m.scanning, faceTotal == 0 {
                    Section {
                        Text(emptyHint).font(.footnote).foregroundStyle(.secondary)
                    }
                }
                if m.scanning, let p = m.progress, p.total > 0 {
                    Section {
                        ProgressView(value: Double(p.done) / Double(p.total))
                        Text("已处理 \(p.done)/\(p.total)")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                setupSection
            }
            .navigationTitle("人物归组")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                faceTotal = app.store.faceStats().faces
                blockedCount = app.store.blockedFacePhotoCount()
                hydrateIfNeeded()
            }
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
            let stats = app.store.faceStats()
            Text(faceTotal > 0
                 ? "已入库人脸 \(faceTotal) 张(\(stats.photos) 张照片)"
                 : "覆盖已索引照片(\(stats.done) 张已做人脸处理)")
                .font(.caption2).foregroundStyle(.secondary)
            Button {
                start()
            } label: {
                if m.scanning {
                    HStack { ProgressView(); Text("扫描中…") }
                } else {
                    Label(faceTotal < 2 ? "人脸数据不足,先同步补全" : "开始归组",
                          systemImage: "person.2")
                }
            }
            .disabled(m.scanning || faceTotal < 2 || app.models.state(for: .face) != .ready)
            if blockedCount > 0 {
                Button {
                    restoreBlocked()
                } label: {
                    Label("恢复 \(blockedCount) 张已排除照片(重新参与归组)",
                          systemImage: "arrow.uturn.backward")
                }
            }
            Button(role: .destructive) {
                confirmClearTags = true
            } label: {
                Label("清除全部人名标签", systemImage: "tag.slash")
            }
            Text("精确档:入库即过滤——截图、低质量脸、大侧脸一律不参与(苹果官方质量分把关);每张脸都必须与「平均脸」核验达标才归属,拿不准的一律进「不确定人物」。归组时现场补认合影里的第二个人:认出是谁就进对应组合组(如 程玉婷 × 杨如);有明显同行脸但认不出的(侧脸/模糊)进「某某 与其他人」;纯单人照才留在单人组。重新命名会覆盖旧名。组内点「排除」可手动移除错误照片,命名后搜索名字即命中。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - 结果区

    private var resultsSection: some View {
        Section {
            if let t = m.lastScanAt {
                HStack {
                    Text("上次归组 \(Self.timeFmt.string(from: t))——结果已存库,重开不需要重扫")
                        .font(.caption2).foregroundStyle(.tertiary)
                    Spacer()
                    // 常驻重扫入口:结果已水化时也一眼能找到(165 教训:水化旧结果
                    // 置顶+文案「不需要重扫」,用户找不到重扫机会,以为问题没修)
                    Button("重新归组") { start() }
                        .font(.caption2)
                        .disabled(m.scanning || faceTotal < 2 || app.models.state(for: .face) != .ready)
                }
            }
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
            Text("人物 \(m.totalPersons) 人 · \(m.groups.count) 组")
        } footer: {
            Text(m.unknownCount > 0
                 ? "截图/低质量/大侧脸不入库;每张脸都与平均脸核验达标才归属,拿不准的 \(m.unknownCount) 张在「不确定人物」。角标=与平均脸的相似度。不足 \(foldThreshold) 张的组默认折叠。归组含合影第二人现场补认(2048px 复检,约多花几分钟);认不出的同行人照片归「与其他人」。"
                 : "截图/低质量/大侧脸不入库;每张脸都与平均脸核验达标才归属,拿不准的进「不确定人物」。角标=与平均脸的相似度。不足 \(foldThreshold) 张的组默认折叠。归组含合影第二人现场补认(2048px 复检,约多花几分钟);认不出的同行人照片归「与其他人」。")
        }
    }

    /// 少于 10 张的组默认折叠隐藏(用户指定),主列表只看大组
    private let foldThreshold = 10
    private static let timeFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MM-dd HH:mm"
        return f
    }()
    private var majorGroups: [ComboGroup] { m.groups.filter { $0.refs.count >= foldThreshold } }
    private var minorGroups: [ComboGroup] { m.groups.filter { $0.refs.count < foldThreshold } }

    private func personDisplay(_ ci: Int) -> String {
        let n = m.personNames[ci] ?? ""
        return n.isEmpty ? "未命名" : n
    }

    private func groupTitle(_ g: ComboGroup) -> String {
        guard !g.members.isEmpty else { return "不确定人物" }
        let parts = g.members.map(personDisplay)
        let base = parts.count == 1 ? parts[0] : parts.joined(separator: " × ")
        return g.withOthers ? "\(base) 与其他人" : base
    }

    private func row(_ g: ComboGroup) -> some View {
        // 缩略图取组内相似度最高的一张(最标准的正面脸)
        let bestIdx = g.scores.indices.max(by: { g.scores[$0] < g.scores[$1] }) ?? 0
        let thumbRef = g.refs.indices.contains(bestIdx) ? g.refs[bestIdx] : g.refs.first
        let weakest = g.scores.filter { $0 < 0.999 }.min()
        return NavigationLink {
            ClusterDetailView(members: g.members,
                              refs: g.refs, scores: g.scores,
                              personNames: $m.personNames, personRefs: m.personRefs,
                              title: groupTitle(g), withOthers: g.withOthers,
                              reasons: m.reasons,
                              details: m.details,
                              onChanged: {
                                  blockedCount = app.store.blockedFacePhotoCount()
                                  start()
                              })
        } label: {
            HStack(spacing: 12) {
                Group {
                    if let a = thumbRef.flatMap({ m.assets[$0] }) {
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
            m.personNames = [:]
            app.notify("已清除全部人名标签;确认单人分组无误后,再逐组重新命名")
        } catch {
            app.fail(error)
        }
    }

    /// 恢复全部被排除的照片:重新参与归组并立即重扫
    private func restoreBlocked() {
        do {
            try app.store.unblockAllFacePhotos()
            blockedCount = 0
            app.notify("已恢复全部被排除的照片,重新归组中")
            start()
        } catch {
            app.fail(error)
        }
    }

    /// 合影第二人现场补认(见 start() 调用处注释)。对每张 keyed 照片:主角级脸
    /// (宽度 ≥ 最大脸 30%,纯背景小脸不算)无闸门检测+对齐+嵌入,与人物质心按
    /// 双判据匹配。返回每张照片 (认出的人物簇集合, 未识别主角级脸数, 归类理由,
    /// 理由附注)。⚠️ 匹配门槛低于入库核验线 0.48:被核验移出/闸门滤掉的第二
    /// 人脸(侧/糊)对本人质心常在 0.32-0.47——用 0.48 认人=逻辑死锁(158 实测)。
    /// 双判据防误认:≥0.40 直接认;0.32-0.40 需比第二名人物高出 0.12(跨人真实
    /// 分布 ~0.16-0.30)。认不出的侧/糊脸(身份匹配不可行)由调用方归「与其他人」
    /// 桶——用户定稿:认不出的不能装作单人。
    /// 归类理由:0 全部强匹配 / 1 弱分回退(note=被吞脸最低分×100) /
    /// 2 取图失败(不落缓存) / 3 复检未检出人脸
    private static func recoverCompanions(refs: [String], baseKeys: [String: Set<Int>],
                                          centroids: [Int: [Float]], dim: Int,
                                          photo: PhotoLibraryService, assets: [String: PHAsset],
                                          face: FaceEmbedder,
                                          siglipAnchors: (two: [Float], one: [Float])?,
                                          coOccurrence: [Int: [Int: Int]],
                                          clusterMembers: [Int: [Int]], flat: [Float],
                                          cached: [String: (matched: Set<Int>, unexplained: Int, reason: Int, note: Int, detail: String)],
                                          store: VectorStore, ver: String,
                                          progress: @escaping (Int, Int) -> Void) async -> (verdicts: [String: (recovered: Set<Int>, unexplained: Int, reason: Int, note: Int, detail: String)], fetchFails: Int) {
        /// 逐躯干 宽%/面% 对(诊断用):「[100%/100%,81%/42%]」——下轮截图
        /// 即知第二人躯干多大、18% 宽线+15% 面积线卡没卡住,数据驱动调线
        func bodyWidths(_ bodies: [CGRect]) -> String {
            guard let hero = bodies.max(by: { $0.width < $1.width }), hero.width > 0 else { return "" }
            let heroArea = hero.width * hero.height
            let parts = bodies.map { b -> (w: Int, s: String) in
                let w = Int((b.width / hero.width * 100).rounded())
                let a = heroArea > 0 ? Int((b.width * b.height / heroArea * 100).rounded()) : 100
                return (w, "\(w)%/\(a)%")
            }.sorted { $0.w > $1.w }.map { $0.s }
            return "[" + parts.joined(separator: ",") + "]"
        }
        /// 簇内 top-5 近邻均值(K 通道):脸向量与该簇每个库内脸逐个余弦,
        /// 取最高 5 个平均。质心是平均脸,对糊/侧脸系统性压分;近邻对
        /// 「同角度/同糊的历史脸」更公平(InsightFace 标准匹配即 KNN)
        func knnMean(_ ci: Int, vec: [Float]) -> Float? {
            guard let rows = clusterMembers[ci], !rows.isEmpty else { return nil }
            var sims = [Float](); sims.reserveCapacity(rows.count)
            for r in rows {
                var d: Float = 0
                vec.withUnsafeBufferPointer { vb in
                    flat.withUnsafeBufferPointer { fb in
                        vDSP_dotpr(vb.baseAddress!, 1, fb.baseAddress! + r * dim, 1,
                                   &d, vDSP_Length(dim))
                    }
                }
                sims.append(d)
            }
            sims.sort(by: >)
            let k = min(5, sims.count)
            return sims.prefix(k).reduce(0, +) / Float(k)
        }
        var out: [String: (recovered: Set<Int>, unexplained: Int, reason: Int, note: Int, detail: String)] = [:]
        for (k, v) in cached { out[k] = (v.matched, v.unexplained, v.reason, v.note, v.detail) }
        let total = refs.count
        progress(0, total)
        var done = 0
        var idx = 0
        var fetchFails = 0
        var pending: [(ref: String, matched: Set<Int>, unexplained: Int, reason: Int, note: Int, detail: String)] = []
        /// 取下一个待处理照片;命中的缓存直接算完成(被杀后重跑=秒过),
        /// 攒满 50 张落一次盘,进程再被杀也只丢尾部几十张
        func takeRef() -> String? {
            while idx < refs.count {
                let r = refs[idx]; idx += 1
                if cached[r] != nil {
                    done += 1
                    if done % 50 == 0 || done == total { progress(done, total) }
                    continue
                }
                return r
            }
            return nil
        }
        func work(_ ref: String) async -> (String, Set<Int>, Int, Bool, Int, Int, String)? {
            guard let a = assets[ref], let key = baseKeys[ref] else { return nil }
            var img: CGImage? = try? await photo.image(for: a, maxPixel: 2048)
            if img == nil {
                // iCloud 优化存储的老照片:2048px 要联网取回,偶发下载失败会把它
                // 静默当"已确认单人"放行——退半秒重试一次,仍失败则计数上报
                try? await Task.sleep(nanoseconds: 600_000_000)
                img = try? await photo.image(for: a, maxPixel: 2048)
            }
            guard let img else { return (ref, [], 0, true, 2, 0, "取图失败") }
            // 协作线程没有逐任务的 autorelease pool:上千张 × 2048px 的 Vision
            // ObjC 临时对象会一路攒到内存红线被系统杀进程(扫描到末尾莫名
            // 无结果的潜在诱因)——每张处理完立即排干
            return autoreleasepool { () -> (String, Set<Int>, Int, Bool, Int, Int, String)? in
                // 无闸门检测(质量/姿态全关):补认的目的恰是找回被闸门滤掉的第二人;
                // 2048px 与索引同级;下限 30px(157 数人头同款)——30% 比例线才是
                // 「纯背景小脸不算」的语义闸门。⚠️ 双通道并集(170 实测漏径:
                // landmarks 版对贴脸/侧脸双人自拍整张只回 1 脸,rectangles 版
                // 检出率不同,并集只增不减);rectangles-only 的脸无关键点,
                // 走 alignedFace 的外接框退化对齐
                let union = FaceKit.detectUnion(cg: img, maxFaces: 12, minFacePx: 30)
                let faces = union.faces
                // keyed 照片复检未发现 ≥30px 候选脸:与 30% 主角脸过滤分开记录,
                // 长按诊断时不会再把“未检出”和“检出后被尺寸门槛过滤”混为一谈。
                guard !faces.isEmpty else {
                    // 零脸照片(远景/糊脸,168 实测新径)也过躯干兜底:脸检彻底
                    // 失败但躯干可数——两人背影/远景合照连一张脸都检不出,
                    // ≥2 同伴级躯干仍是同行证据,不能默认单人
                    let bodies = FaceKit.detectHumans(cg: img)
                    let significant = FaceRecoveryPolicy.significantBodyCount(bodies)
                    if FaceRecoveryPolicy.companionSuspectedByBody(
                        detectedFaces: 0, significantBodies: significant) {
                        return (ref, [], 1, false, 3, 0,
                                "未检出≥30px候选脸;躯干\(bodies.count)个\(bodyWidths(bodies))/同伴级\(significant)→判有同行")
                    }
                    return (ref, [], 0, false, 3, 0,
                            "未检出≥30px候选脸;躯干\(bodies.count)个\(bodyWidths(bodies))/同伴级\(significant)")
                }
                // 主角级脸 = 宽度 ≥ 最大脸 30%:纯背景小脸不参与匹配也不算“他人”。
                // 不以显著脸数量作提前放行条件;每张显著脸仍逐一匹配。
                let maxW = faces.map { $0.boundingBox.width }.max() ?? 0
                let major = faces.filter { $0.boundingBox.width >= maxW * 0.3 }
                let excludedFaceSizes = faces.filter { $0.boundingBox.width < maxW * 0.3 }.map {
                    let width = $0.boundingBox.width
                    let pixels = Int((width * CGFloat(img.width)).rounded())
                    let percent = maxW > 0 ? Int((width / maxW * 100).rounded()) : 0
                    return "\(pixels)px/\(percent)%"
                }
                let faceSummary = "≥30px rect检出\(union.rectCount)/关键点\(union.landmarkCount)/并集\(faces.count),≥30%主角\(major.count)脸"
                    + (excludedFaceSizes.isEmpty ? "" : ";过滤小脸 " + excludedFaceSizes.joined(separator: ","))
                guard !major.isEmpty else {
                    return (ref, [], 0, false, 3, 0, faceSummary)
                }
                // 先算全部主角级脸的匹配分,再统一裁决(双人认领 veto 需要横向比)
                struct FaceMatch { var ci: Int; var best: Float; var secondCi: Int; var second: Float; var strong: Bool; var embedding: [Float]; var width: CGFloat }
                var fms: [FaceMatch] = []
                var embedFailWidths: [CGFloat] = []
                for obs in major {
                    guard let aligned = FaceKit.alignedFace(source: img, observation: obs),
                          let vec = try? face.embed(aligned: aligned) else {
                        // 对齐/嵌入失败的算未识别(按宽度过背景路人间)
                        embedFailWidths.append(obs.boundingBox.width)
                        continue
                    }
                    var bestCi = -1, secondCi = -1
                    var bestSim: Float = -2, secondSim: Float = -2
                    for (ci, c) in centroids {
                        var dot: Float = 0
                        vDSP_dotpr(vec, 1, c, 1, &dot, vDSP_Length(dim))
                        if dot > bestSim {
                            secondSim = bestSim; secondCi = bestCi
                            bestSim = dot; bestCi = ci
                        } else if dot > secondSim {
                            secondSim = dot; secondCi = ci
                        }
                    }
                    let strong = bestCi >= 0 &&
                        (bestSim >= 0.40 || (bestSim >= 0.32 && bestSim - secondSim >= 0.12))
                    fms.append(FaceMatch(ci: bestCi, best: bestSim,
                                         secondCi: secondCi, second: secondSim,
                                         strong: strong, embedding: vec,
                                         width: obs.boundingBox.width))
                }
                // 同图多脸认领同一主角时,既保留原有分差 veto,也要求强匹配脸
                // 彼此达到逐脸核验线(0.48);否则较弱的脸不能冒充该主角的第二张脸。
                // 裁决 helper 与纯 Swift 回归测试共用,避免生产逻辑和测试漂移。
                let policyMatches = fms.map {
                    FaceRecoveryPolicy.Match(cluster: $0.ci, score: $0.best,
                                             strong: $0.strong, embedding: $0.embedding)
                }
                let vetoed = FaceRecoveryPolicy.vetoedIndices(
                    in: policyMatches, recognizedClusters: key)
                var matched = Set<Int>()
                var unexplainedCount = 0   // 只有「同伴尺寸」的未解释脸才算他人(第十三批)
                var swallowedMin: Float = .infinity   // 被回退吞掉的脸里最低分(诊断用)
                var detailParts: [String] = []
                // 背景路人线:脸宽 < 主脸 45% 且认不出/被降级的 = 远处路人不
                // 算他人(175 实测大量「看起来就是单人」的照片被 30-45% 的
                // 远小路人推进「与其他人」)。≥30% 仍参与匹配——真同伴的
                // 小脸照常能被 S/W/Y/K/J 认出进组合组,识别路径不受影响。
                let strangerFloor = maxW * 0.45
                for (i, f) in fms.enumerated() {
                    let tag: String
                    if vetoed.contains(i) {
                        if f.width >= strangerFloor {
                            tag = "V"   // 双人认领 veto:未识别
                            unexplainedCount += 1
                        } else {
                            tag = "B"   // 背景路人:认领被降级但脸小,不算他人
                        }
                    } else if f.strong {
                        matched.insert(f.ci)
                        tag = "S"
                    } else if f.ci >= 0, key.contains(f.ci),
                              f.best >= 0.25,
                              f.best - f.second >= 0.08 || key.contains(f.secondCi) {
                        // 本人的难脸(重检更糊对不上线)最像组内本人→吞掉不算"他人"。
                        // 165 的教训:回退必须带地板+边距——同行者的难脸有自己质心做
                        // 近身竞争者(边距≈0),本人难脸的第二名是无关人物(边距天然大)
                        swallowedMin = min(swallowedMin, f.best)
                        tag = "W"
                    } else if f.ci >= 0,
                              FaceRecoveryPolicy.priorRecoveredByCoOccurrence(
                                best: f.best, second: f.second,
                                coCount: key.reduce(0) { max($0, coOccurrence[$1]?[f.ci] ?? 0) }) {
                        // 同框先验(第九批):171 实测 rect 并集已把第二脸找回,但糊脸
                        // 对质心只有 0.22-0.27(双判据线 0.32 下)全判 U,890 张合影堆在
                        // 「与其他人」。未用的强证据:最像的簇与本照片主角历史高频同框
                        // (杨如×程玉婷数百张)——陌生人成不了主角的高频同框者。
                        // 三重门:同框≥20 + 地板 0.22 + 领先 0.06 → 认作该同行者的难脸
                        matched.insert(f.ci)
                        tag = "Y"
                    } else if f.ci >= 0, let bk = knnMean(f.ci, vec: f.embedding),
                              FaceRecoveryPolicy.priorRecoveredByKNN(
                                bestKNN: bk,
                                secondKNN: f.secondCi >= 0 ? knnMean(f.secondCi, vec: f.embedding) : nil) {
                        // 近邻复核(第九批):质心是全簇平均脸,对糊/侧脸系统性压分;
                        // 改与簇内库内成员逐脸比 top-5 均值——糊脸对「同糊历史脸」的
                        // 分高于对平均脸。Y 先验没接住的低领先样本(171 实测
                        // c1@0.22>c2@0.18)由这里救
                        matched.insert(f.ci)
                        tag = "K"
                    } else {
                        if f.width >= strangerFloor {
                            tag = "U"   // 对不上任何人物也不像组内本人:未识别同行脸
                            unexplainedCount += 1
                        } else {
                            tag = "B"   // 背景路人:远小脸认不出,不算他人
                        }
                    }
                    detailParts.append(String(format: "c%d@%.2f>c%d@%.2f%@",
                                              f.ci, f.best, f.secondCi, f.second, tag))
                }
                for w in embedFailWidths {
                    detailParts.append(w >= strangerFloor ? "嵌入失败U" : "嵌入失败B")
                    if w >= strangerFloor { unexplainedCount += 1 }
                }
                // 未识别的同伴尺寸脸 = 认不出/被降级 且宽 ≥ 主脸 45%
                // (侧/糊同伴的脸)。<45% 的远小路人是 B,不算他人(第十三批)。
                // 镜像/海报同人双脸:嵌入一致(≥0.48)豁免分差 veto(第十一批)
                var unexplained = unexplainedCount
                var bodyNote = ""
                // 躯干兜底(167 实测漏径:双人照复检只出 1 脸——同行者的脸被
                // Vision 检测层整个漏掉,根本没进匹配环节):纯单人判决
                // (恰 1 张主角脸且已解释)再数人体,≥2 个同伴级躯干
                // (宽度 ≥ 最宽躯干 18%,见 FaceRecoveryPolicy.significantBodyCount)
                // → 同行者脸被漏检,不能装作单人
                if major.count == 1 && unexplained == 0 {
                    let bodies = FaceKit.detectHumans(cg: img)
                    let significant = FaceRecoveryPolicy.significantBodyCount(bodies)
                    var suspected = FaceRecoveryPolicy.companionSuspectedByBody(
                        detectedFaces: major.count, significantBodies: significant)
                    var extraNote = ""
                    // SigLIP 计数兜底(第三层):脸并集与躯干都只数出 1(170 实测
                    // 『躯干1个/同伴级1』= 贴脸/重叠构图连躯干检测器也只框出一人)
                    // 时,问整图语义「两人 vs 一人」——照片向量已在索引库,零图像
                    // 推理;判据保守(margin 0.025),诊断带双分数可校准
                    if !suspected, let anchors = siglipAnchors,
                       let pv = store.clipVector(ref: ref), pv.count == anchors.two.count {
                        var two: Float = 0, one: Float = 0
                        vDSP_dotpr(pv, 1, anchors.two, 1, &two, vDSP_Length(pv.count))
                        vDSP_dotpr(pv, 1, anchors.one, 1, &one, vDSP_Length(pv.count))
                        suspected = FaceRecoveryPolicy.companionSuspectedBySigLIP(
                            twoScore: two, oneScore: one)
                        extraNote = String(format: ";siglip two=%.2f/one=%.2f", two, one)
                            + (suspected ? "→判有同行" : "")
                    }
                    if suspected { unexplained = 1 }
                    // 躯干证据无论是否触发都写进诊断(逐躯干宽度百分比:下轮
                    // 截图即知每个躯干多大、18% 线卡没卡住第二人,直接定调线)
                    bodyNote = ";躯干\(bodies.count)个\(bodyWidths(bodies))/同伴级\(significant)"
                        + (suspected ? "→判有同行" : "") + extraNote
                }
                let swallowed = swallowedMin.isFinite
                let reason = swallowed ? 1 : 0
                let note = swallowed ? Int(swallowedMin * 100) : major.count
                return (ref, matched, unexplained, false, reason, note,
                        faceSummary + bodyNote + "; " + detailParts.joined(separator: " | "))
            }
        }
        await withTaskGroup(of: (String, Set<Int>, Int, Bool, Int, Int, String)?.self) { group in
            for _ in 0..<4 {
                if let ref = takeRef() { group.addTask { await work(ref) } }
            }
            while !group.isEmpty {
                if let r = await group.next() {
                    done += 1
                    if done % 10 == 0 || done == total { progress(done, total) }
                    if let (ref, matched, unexplained, failed, reason, note, detail) = r {
                        out[ref] = (matched, unexplained, reason, note, detail)
                        if failed {
                            // 取图失败不落缓存:下次扫描自动重试;先按已入库
                            // 主角归组,完成提示里如实上报数量
                            fetchFails += 1
                        } else {
                            pending.append((ref, matched, unexplained, reason, note, detail))
                            if pending.count >= 50 {
                                store.saveRecoverCache(pending, ver: ver)
                                pending.removeAll()
                            }
                        }
                    }
                }
                if Task.isCancelled { break }
                if let ref = takeRef() {
                    group.addTask { await work(ref) }
                }
            }
        }
        if !pending.isEmpty { store.saveRecoverCache(pending, ver: ver) }
        // —— 聚合转正(第十批 J 通道):二轮扫描本次运行的全部裁决 ——
        // 单张证据不够线的 U 脸(0.18-0.27 糊脸),若在 ≥8 次观测里稳定 best
        // 指向同一簇,跨照片一致性就是身份铁证——嵌入噪声方向随机,真同伴的
        // 糊脸方向稳定。Y 同框先验救不了它:中频同伴的合影从未有一张被认出
        // → 历史同框计数恒为 0(172 实测 c9@0.27>c6@0.20U,分数边距全过 Y 门)。
        // J 只看本次运行内的 (主角,best簇) 配对频次,自闭环不依赖历史;
        // 转正行重写 recover_cache(INSERT OR REPLACE),归组用的是改后的 out。
        var pairCount: [Int: [Int: Int]] = [:]
        for (ref, v) in out {
            guard let key = baseKeys[ref] else { continue }
            for f in FaceRecoveryPolicy.parseFaceSegments(v.detail)
            where f.tag == "U" && f.ci >= 0 {
                for k in key { pairCount[k, default: [:]][f.ci, default: 0] += 1 }
            }
        }
        var promoted: [(ref: String, matched: Set<Int>, unexplained: Int, reason: Int, note: Int, detail: String)] = []
        for (ref, v) in Array(out) {
            guard let key = baseKeys[ref] else { continue }
            var matched = v.recovered, unexplained = v.unexplained, detail = v.detail
            for f in FaceRecoveryPolicy.parseFaceSegments(v.detail)
            where f.tag == "U" && f.ci >= 0 {
                let count = key.reduce(0) { max($0, pairCount[$1]?[f.ci] ?? 0) }
                // 破平局(第十四批):0.23>0.22 的二义脸在分数上分不出两个候选,
                // 借主角的历史同框破局——杨如数百张 vs 无关候选接近 0。两个都
                // 常同框的伙伴打平时(如 300 vs 280)不赌,诚实留 U
                let coBest = key.reduce(0) { max($0, coOccurrence[$1]?[f.ci] ?? 0) }
                let coSecond = f.secondCi >= 0
                    ? key.reduce(0) { max($0, coOccurrence[$1]?[f.secondCi] ?? 0) } : 0
                // 第二名是主角自己的簇(第二十二批护栏):这张脸在「某同伴 vs
                // 本人」之间撕扯(镜像/糊脸对大质心虚高)——程玉婷的糊脸 best
                // 落到高频同框的杨如簇、second 落回她自己,同框先验会把它判给
                // 杨如,单人照凭空多出一个"同伴"。此况不赌,诚实留 U
                let secondIsProtagonist = f.secondCi >= 0 && key.contains(f.secondCi)
                guard FaceRecoveryPolicy.aggregatedRecovered(
                    pairCount: count, best: f.best, second: f.second)
                        || (!secondIsProtagonist
                            && FaceRecoveryPolicy.aggregatedTieBreakRecovered(
                                pairCount: count, best: f.best, coBest: coBest, coSecond: coSecond))
                    else { continue }
                matched.insert(f.ci)
                unexplained -= 1
                detail = detail.replacingOccurrences(
                    of: String(format: "c%d@%.2f>c%d@%.2fU", f.ci, f.best, f.secondCi, f.second),
                    with: String(format: "c%d@%.2f>c%d@%.2fJ", f.ci, f.best, f.secondCi, f.second))
            }
            if matched != v.recovered {
                out[ref] = (matched, unexplained, v.reason, v.note, detail)
                promoted.append((ref, matched, unexplained, v.reason, v.note, detail))
            }
        }
        if !promoted.isEmpty { store.saveRecoverCache(promoted, ver: ver) }
        return (out, fetchFails)
    }

    /// 水化:进程重启/重开面板时从库内快照直接恢复归组结果(162 前结果只在
    /// 内存,App 退后台被系统结束进程=十分钟的活全白干,用户被迫重跑)。
    /// 人名按当前标签重推——重命名/清标签无需重扫
    private func hydrateIfNeeded() {
        guard m.groups.isEmpty, !m.scanning else { return }
        // 算法盐 a15:归组/补认逻辑变更必须升盐——旧快照与补认缓存不能复活旧裁决。
        // (人脸行/排除/模型)不含算法版本,163 的旧判决经快照水化+缓存命中在
        // 164 原样复活,用户看到的就是"修了没变化"。第十四批:J 破平局(同框
        // 先验裁决二义脸)改变了 U→J 的转正集合,旧快照/缓存必须作废重判
        let ver = app.store.peopleFingerprint(
            extra: "a15-m\(UserDefaults.standard.integer(forKey: "faceModelSize"))")
        guard let json = app.store.loadPeopleResults(ver: ver),
              let data = json.data(using: .utf8),
              let snap = try? JSONDecoder().decode(PeopleSnapshot.self, from: data) else {
            // 没有可用快照却留有「扫描中」记账 = 上次归组被系统掐死在半路
            if UserDefaults.standard.object(forKey: "peopleScanRunning") != nil {
                app.notify("上次归组未完成(App 退后台被系统结束):点「开始归组」继续,已补认的照片会直接秒过")
            }
            return
        }
        var fetched: [String: PHAsset] = [:]
        let allRefs = snap.groups.flatMap { $0.refs }
        let fetch = PHAsset.fetchAssets(withLocalIdentifiers: allRefs, options: nil)
        fetch.enumerateObjects { a, _, _ in fetched[a.localIdentifier] = a }
        m.groups = snap.groups.enumerated().map { i, g in
            ComboGroup(id: i, members: g.members, refs: g.refs,
                       scores: g.scores, withOthers: g.withOthers)
        }
        m.personRefs = snap.personRefs
        m.personNames = Self.deriveNames(personRefs: snap.personRefs,
                                         tags: app.store.userTagsForRefs(allRefs))
        m.assets = fetched
        m.totalPersons = snap.totalPersons
        m.unknownCount = snap.unknownCount
        m.lastScanAt = snap.savedAt
        var hydrated: [String: (reason: Int, note: Int)] = [:]
        for (k, v) in snap.reasons ?? [:] where v.count >= 2 {
            hydrated[k] = (v[0], v[1])
        }
        m.reasons = hydrated
        m.details = snap.details ?? [:]
        UserDefaults.standard.removeObject(forKey: "peopleScanRunning")
    }

    /// 人名推导:该人照片里 ≥3 票且过半的标签众数(148 教训,无门槛会污染)。
    /// 扫描完成与快照水化共用同一判据,保证两条路名字一致
    private static func deriveNames(personRefs: [Int: [String]],
                                    tags: [String: [String]]) -> [Int: String] {
        var names: [Int: String] = [:]
        for (ci, refs) in personRefs where !refs.isEmpty {
            var counts: [String: Int] = [:]
            for r in refs { for t in tags[r] ?? [] { counts[t, default: 0] += 1 } }
            if let best = counts.max(by: { $0.value < $1.value }),
               best.value >= 3, best.value * 2 >= refs.count {
                names[ci] = best.key
            }
        }
        return names
    }

    private func start() {
        guard !m.scanning else { return }
        // 守卫:人脸模型换了但脸数据还没按新模型重嵌——此时聚类的是旧模型向量,
        // 新旧嵌入空间不通用,聚出来的必然混人。重建中(库被清空重嵌)不拦,
        // 聚的是已完成部分的新向量
        let want = UserDefaults.standard.integer(forKey: "faceModelSize")
        let indexed = UserDefaults.standard.integer(forKey: "faceIndexedSize")
        if want > 0, want != indexed, app.store.faceStats().faces > 0 {
            app.notify("人脸模型已更换,旧数据还没重建完:回资料库等「同步」跑完再归组(约 30-60 分钟,可中断可续)")
            return
        }
        m.scanning = true
        m.progress = (0, 1)
        let store = app.store
        let photoLib = app.photoLib
        let faceModel = app.models.face
        let model = app.people   // 结果写共享模型:面板关了再开不丢(160 教训)
        // 指纹:人脸数据/排除集合/模型任一变化,快照与补认缓存自动作废;
        // a15=算法盐,归组逻辑变更必须升盐(与 hydrateIfNeeded 同串)
        let ver = store.peopleFingerprint(
            extra: "a15-m\(UserDefaults.standard.integer(forKey: "faceModelSize"))")
        // 「扫描中」记账 + 屏幕常亮:退后台被系统结束进程后,重开面板能明确
        // 告知「上次未完成」;常亮防前台锁屏挂起
        UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: "peopleScanRunning")
        UIApplication.shared.isIdleTimerDisabled = true
        Task.detached(priority: .userInitiated) {
            do {
                // SigLIP 计数兜底锚点(第八批):两人/一人各两条 prompt 取均值
                // 再归一——脸并集与躯干都只数出 1 时,问整图语义「几个人」作
                // 最后证据。模型未装(nil)该层静默跳过;嵌入在此一次完成,
                // 归组全程只做向量算术
                var siglipAnchors: (two: [Float], one: [Float])? = nil
                let siglipModel = await MainActor.run { app.models.siglip }
                if let sig = siglipModel {
                    func anchor(_ prompts: [String]) -> [Float]? {
                        var acc: [Float]? = nil
                        for p in prompts {
                            guard let v = try? sig.embedQuery(p) else { return nil }
                            if var a = acc { for i in v.indices { a[i] += v[i] }; acc = a }
                            else { acc = v }
                        }
                        guard var a = acc else { return nil }
                        var ss: Float = 0
                        vDSP_dotpr(a, 1, a, 1, &ss, vDSP_Length(a.count))
                        let n = max(ss.squareRoot(), 1e-8)
                        for i in a.indices { a[i] /= n }
                        return a
                    }
                    if let t = anchor(["This is a photo of two people.",
                                       "This is a photo of a couple."]),
                       let o = anchor(["This is a photo of one person.",
                                       "This is a photo of a single person."]) {
                        siglipAnchors = (t, o)
                    }
                }
                let data = try store.allFaceVectors()
                let raw = DuplicateScanner.scan(
                    keys: data.keys, flat: data.flat, dim: data.dim, threshold: threshold) { done, total in
                    Task { @MainActor in model.progress = (done, total) }
                }
                // 组间合并+垃圾剔除+逐脸核验+内聚解散+双峰拆分(全部在扫描器内完成)
                let groupedIdx = Set(raw.flatMap { $0.map(\.index) })
                let singletons = (0..<data.keys.count).filter { !groupedIdx.contains($0) }
                var scored = DuplicateScanner.mergeGroups(
                    raw, flat: data.flat, dim: data.dim,
                    linkThreshold: link, expelFloor: expelFloor,
                    singletons: singletons,
                    coherenceFloor: coherenceFloor, verifyFloor: verifyFloor,
                    splitFloor: splitFloor) { done, total in
                    Task { @MainActor in model.progress = (done, max(total, 1)) }
                }

                // 同名簇合并(第二十二批):聚类撕人时,同一真人会留下多个
                // 碎片簇——每个碎片各自成组(用户实测「程玉婷 与其他人」
                // 434/25/21 三组并存),碎片间还互相组合出「程玉婷 × 未命名」
                // (332 张混满单人照:镜像双脸各命中一个碎片)。只要多个簇从
                // 用户标签推出同一个人名,成员并回代表簇(簇号最小,确定性);
                // 在路由/质心/同框统计之前合并,下游全部看到合并后的簇
                do {
                    var earlyRefs: [Int: [String]] = [:]
                    for (ci, g) in scored.enumerated() {
                        var s = Set<String>()
                        for m in g { s.insert(data.refs[m.index]) }
                        earlyRefs[ci] = Array(s)
                    }
                    let earlyTags = store.userTagsForRefs(
                        Array(Set(earlyRefs.values.flatMap { $0 })))
                    let plan = FaceRecoveryPolicy.sameNameMergePlan(
                        clusterNames: Self.deriveNames(personRefs: earlyRefs, tags: earlyTags))
                    if !plan.isEmpty {
                        var repIdx: [Int: Int] = [:]
                        var out: [[(index: Int, score: Float)]] = []
                        for (ci, g) in scored.enumerated() {
                            guard plan[ci] == nil else { continue }   // 碎片:成员并入代表
                            repIdx[ci] = out.count
                            out.append(g)
                        }
                        for (frag, rep) in plan.sorted(by: { $0.key < $1.key }) {
                            if let i = repIdx[rep] { out[i].append(contentsOf: scored[frag]) }
                        }
                        scored = out
                    }
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
                // ③ 先取资产(补认要用图),再对「合影嫌疑照片」做第二人现场补认,
                // 最后按主角集合精确归组
                var fetched: [String: PHAsset] = [:]
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: Array(refClusters.keys), options: nil)
                fetch.enumerateObjects { a, _, _ in fetched[a.localIdentifier] = a }

                // 合影第二人现场补认:入库闸门(小脸/质量/姿态)会把合影里第二个人
                // 的脸滤在库外 → 该照片只挂主角、混进单人组(156/158 两轮实测
                // 程玉婷/杨如合影仍挂在单人组)。逐张补认:主角级脸数 > 主角数的
                // 照片,对其主角级脸无闸门检测+对齐+嵌入(2048px 与索引同级),
                // 与人物质心按 158 修定的双判据(≥0.40 或 ≥0.32+领先第二名 0.12)
                // 匹配 → 主角集合扩为真实集合,进对应组合组(杨如×程玉婷);
                // 认不出的(路人/背景脸)不算他人——单人组只放真单人照
                var baseKeys: [String: Set<Int>] = [:]
                for (r, cs) in refClusters {
                    let key = cs.intersection(persons)
                    if !key.isEmpty { baseKeys[r] = key }
                }
                var centroids: [Int: [Float]] = [:]
                for (ci, g) in scored.enumerated() where persons.contains(ci) {
                    var c = [Float](repeating: 0, count: data.dim)
                    for m in g { for d in 0..<data.dim { c[d] += data.flat[m.index * data.dim + d] } }
                    var ss: Float = 0
                    vDSP_dotpr(c, 1, c, 1, &ss, vDSP_Length(data.dim))
                    let n = max(ss.squareRoot(), 1e-8)
                    for d in 0..<data.dim { c[d] /= n }
                    centroids[ci] = c
                }
                // 同框计数(第九批 Y 通道):每对人物在多少张照片里同时被认出——
                // 杨如×程玉婷数百张;陌生人/低频路人到不了 20
                var coOccurrence: [Int: [Int: Int]] = [:]
                for ks in baseKeys.values {
                    let ps = Array(ks).sorted()
                    for i in 0..<ps.count {
                        for j in (i + 1)..<ps.count {
                            coOccurrence[ps[i], default: [:]][ps[j], default: 0] += 1
                            coOccurrence[ps[j], default: [:]][ps[i], default: 0] += 1
                        }
                    }
                }
                // 簇成员行号(第九批 K 通道 KNN 用):脸向量在 flat 里的行
                var clusterMembers: [Int: [Int]] = [:]
                for (ci, g) in scored.enumerated() where persons.contains(ci) {
                    clusterMembers[ci] = g.map { $0.index }
                }
                var recovered: [String: (recovered: Set<Int>, unexplained: Int, reason: Int, note: Int, detail: String)] = [:]
                var fetchFails = 0
                if let faceModel {
                    // 先取补认进度缓存:被系统杀掉后重跑,已处理的照片秒过
                    let cached = store.loadRecoverCache(ver: ver)
                    (recovered, fetchFails) = await Self.recoverCompanions(
                        refs: Array(baseKeys.keys), baseKeys: baseKeys,
                        centroids: centroids, dim: data.dim,
                        photo: photoLib, assets: fetched, face: faceModel,
                        siglipAnchors: siglipAnchors,
                        coOccurrence: coOccurrence, clusterMembers: clusterMembers,
                        flat: data.flat,
                        cached: cached, store: store, ver: ver) { done, total in
                        Task { @MainActor in model.progress = (done, total) }
                    }
                }

                var comboMap: [Set<Int>: (refs: [String], scores: [Float])] = [:]
                var othersMap: [Set<Int>: (refs: [String], scores: [Float])] = [:]
                var unknownRefs: [String] = []
                var unknownScores: [Float] = []
                var reasonsMap: [String: (reason: Int, note: Int)] = [:]
                var detailsMap: [String: String] = [:]
                var statRecovered = 0   // 补认出 ≥1 个新主角的照片数
                var statOthers = 0      // 仍有认不出的主角级同行脸的照片数
                for (r, cs) in refClusters {
                    let libKey = cs.intersection(persons)
                    let rec = recovered[r]
                    if let rec { reasonsMap[r] = (rec.reason, rec.note); detailsMap[r] = rec.detail }
                    var key = libKey
                    if let rec { key.formUnion(rec.recovered) }
                    if let rec, !rec.recovered.isSubset(of: libKey) { statRecovered += 1 }
                    let best = cs.compactMap { clusterBest[$0]?[r] }.max() ?? Float(1)
                    if key.isEmpty {
                        unknownRefs.append(r); unknownScores.append(best)
                    } else if (rec?.unexplained ?? 0) >= 1 {
                        // 主角认得出 + 还有认不出的主角级同行脸(侧/糊,身份匹配
                        // 不可行)→ 「主角 与其他人」。用户三轮定稿:这批不能装作
                        // 单人;逐张分流,不污染纯单人组(157 组级打标教训)
                        var entry = othersMap[key] ?? ([], [])
                        entry.refs.append(r); entry.scores.append(best)
                        othersMap[key] = entry
                        statOthers += 1
                    } else {
                        var entry = comboMap[key] ?? ([], [])
                        entry.refs.append(r); entry.scores.append(best)
                        comboMap[key] = entry
                    }
                }
                var ordered: [(members: [Int], refs: [String], scores: [Float], withOthers: Bool)] =
                    comboMap.map { (Array($0.key).sorted(), $0.value.refs, $0.value.scores, false) }
                    + othersMap.map { (Array($0.key).sorted(), $0.value.refs, $0.value.scores, true) }
                // 不确定组先入列再排序:它也是普通一组,按照片数排(153 排序 bug 教训:
                // 排序后追加会让 44 张的不确定组落在 10 张组后面)
                if !unknownRefs.isEmpty {
                    ordered.append(([], unknownRefs, unknownScores, false))
                }
                // 按组内照片数降序(用户指定)
                ordered.sort { $0.refs.count > $1.refs.count }
                // ④ 人名:该人照片里覆盖率 ≥40% 的标签众数(无门槛众数会污染,147 教训)
                let allRefs = Array(Set(ordered.flatMap { $0.refs }))
                let tagMap = store.userTagsForRefs(allRefs)
                var refsOf: [Int: [String]] = [:]
                for ci in persons {
                    refsOf[ci] = Array(clusterRefSets[ci] ?? []).sorted()
                }
                let names = Self.deriveNames(personRefs: refsOf, tags: tagMap)
                var outGroups: [ComboGroup] = []
                for (i, o) in ordered.enumerated() {
                    // PHAsset 取回整体为空(授权降级/图库瞬断)时不能把全部结果
                    // 滤没——宁可保留组,缩略图占灰块,结果不能凭空消失
                    let alive = fetched.isEmpty ? o.refs : o.refs.filter { fetched[$0] != nil }
                    guard !alive.isEmpty else { continue }
                    let aliveSet = Set(alive)
                    let scores = o.scores.enumerated()
                        .filter { aliveSet.contains(o.refs[$0.offset]) }.map(\.element)
                    outGroups.append(ComboGroup(id: i, members: o.members,
                                                refs: alive, scores: scores, withOthers: o.withOthers))
                }
                let unknownAlive = outGroups.first { $0.members.isEmpty }?.refs.count ?? 0

                // 结果落库(162):进程随时被杀都拿得回来,重开面板直接水化;
                // 完成即清「扫描中」记账
                let snap = PeopleSnapshot(
                    groups: outGroups.map {
                        .init(members: $0.members, refs: $0.refs,
                              scores: $0.scores, withOthers: $0.withOthers)
                    },
                    personRefs: refsOf, totalPersons: persons.count,
                    unknownCount: unknownAlive, savedAt: Date(),
                    reasons: reasonsMap.mapValues { [$0.reason, $0.note] },
                    details: detailsMap)
                if let data = try? JSONEncoder().encode(snap) {
                    store.savePeopleResults(json: String(decoding: data, as: UTF8.self), ver: ver)
                }
                UserDefaults.standard.removeObject(forKey: "peopleScanRunning")

                await MainActor.run {
                    model.scanning = false
                    model.progress = nil
                    model.lastScanAt = Date()
                    UIApplication.shared.isIdleTimerDisabled = false
                    model.groups = outGroups
                    model.personNames = names
                    model.personRefs = refsOf
                    model.assets = fetched
                    model.totalPersons = persons.count
                    model.unknownCount = unknownAlive
                    model.reasons = reasonsMap
                    model.details = detailsMap
                    if outGroups.isEmpty {
                        app.notify("没有找到可归组的人脸")
                    } else if statRecovered > 0 || statOthers > 0 {
                        // 诊断口径:认出=进了对应组合组;未识别同行=进了「与其他人」
                        app.notify(fetchFails > 0
                               ? "合影补认:认出 \(statRecovered) 张进对应组合组,\(statOthers) 张有认不出的同行人(侧脸/模糊)已归「与其他人」;\(fetchFails) 张 iCloud 照片未能复检(已按入库主角归组,下次扫描自动重试)"
                               : "合影补认:认出 \(statRecovered) 张进对应组合组,\(statOthers) 张有认不出的同行人(侧脸/模糊)已归「与其他人」")
                    }
                }
            } catch {
                UserDefaults.standard.removeObject(forKey: "peopleScanRunning")
                await MainActor.run {
                    model.scanning = false
                    UIApplication.shared.isIdleTimerDisabled = false
                    app.fail(error)
                }
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
    /// 「主角 与其他人」组(members 也恰好是 1 个主角,但照片里有认不出的
    /// 同行脸):单人组诊断角标不得在此组显示——第十四批实测「见N脸·均判
    /// 本人」挂满与其他人组且文案失真(组内照片必有 U 脸,并非"全判本人")
    var withOthers: Bool = false
    /// 每张照片的归类理由(单人组标注「为什么算单人」);空=旧快照无数据
    var reasons: [String: (reason: Int, note: Int)] = [:]
    /// 逐脸匹配明细(长按照片弹诊断);空=旧快照/缓存无数据
    var details: [String: String] = [:]
    /// 排除照片后的回调(父级重新归组)
    var onChanged: (() -> Void)?

    @EnvironmentObject var app: AppModel
    @State private var assets: [String: PHAsset] = [:]
    @State private var naming = false            // 146 真机验证过的纯 Bool 弹出方式
    @State private var namingMember: Int?
    @State private var nameText = ""
    @State private var applying = false
    @State private var selectedHit: DisplayHit?
    @State private var showingDebug = false      // 长按照片看逐脸裁决(166 诊断)
    @State private var debugText = ""
    @State private var selecting = false         // 排除模式:点选照片移出本组
    @State private var selected: Set<String> = []
    // 划选(第二十批):排除模式下横划过一排即批量勾选;竖向移动让给滚动
    @State private var swipe = SwipeSelectState()
    @State private var swipeFrames: [String: CGRect] = [:]
    @State private var excluding = false

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
                .coordinateSpace(name: "peopleGrid")
                .onPreferenceChange(SwipeCellFramesKey.self) { swipeFrames = $0 }
            }
            .padding(12)
        }
        .safeAreaInset(edge: .bottom) {
            if selecting {
                HStack {
                    Button(role: .destructive) {
                        excludeSelected()
                    } label: {
                        Label(selected.isEmpty ? "点选要移出的错误照片"
                                               : "移出所选 \(selected.count) 张(不再参与归组)",
                              systemImage: "minus.circle")
                            .fontWeight(.semibold)
                    }
                    .disabled(selected.isEmpty || excluding)
                    Spacer()
                    Text("移出的脸不参与聚类,可在上层恢复")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .padding(12)
                .background(.bar)
            }
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                if selecting {
                    Button("取消") {
                        selecting = false
                        selected = []
                    }
                } else {
                    Button("排除") { selecting = true }
                    // 单人组的命名主入口放工具栏(146 同款位置,醒目);
                    // 组合组从「本组人物」逐行命名
                    if members.count == 1 {
                        Button(applying ? "命名中…" : "命名") {
                            nameText = personNames[members[0]] ?? ""
                            namingMember = members[0]
                            naming = true
                        }
                        .disabled(applying)
                    }
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
        .alert("归类诊断", isPresented: $showingDebug) {
            Button("好", role: .cancel) {}
        } message: {
            // 逐脸裁决:S=强匹配认出 W=回退判本人难脸 V=双人认领降级 U=未识别;
            // c数字=人物簇,@分数=对该簇质心/第二名质心的余弦
            Text(debugText + "\n(S=认出 W=本人难脸 V=同认领降级 U=未识别 Y=同框先验 K=近邻复核 J=聚合转正 B=背景路人)")
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
                        if selecting {
                            if selected.contains(ref) { selected.remove(ref) } else { selected.insert(ref) }
                        } else {
                            selectedHit = DisplayHit(
                                id: "face-\(ref)", refKey: ref, target: .asset(a),
                                kind: .photo,
                                title: "人物:\(title)",
                                score: 1, date: a.creationDate, color: nil)
                        }
                    }
            } else {
                Rectangle().fill(Color(.systemGray5)).frame(height: 124)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .swipeSelectable(id: ref, space: "peopleGrid", enabled: selecting,
                         onSweep: { v in
                             swipe.handle(v, frames: swipeFrames,
                                          isSelected: { selected.contains($0) },
                                          apply: { id, on in
                                              if on { selected.insert(id) } else { selected.remove(id) }
                                          })
                         }, onEnd: { swipe.reset() })
        .overlay(alignment: .topLeading) {
            if selecting {
                Image(systemName: selected.contains(ref) ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(selected.contains(ref) ? .blue : .white)
                    .shadow(radius: 2)
                    .padding(6)
            }
        }
        .overlay(alignment: .bottomLeading) {
            // 角标=与「平均脸」的相似度(与逐脸核验同一标尺,组内成员恒 ≥48%);
            // 不确定组没有归属分,不显示
            if members.isEmpty { EmptyView() }
            else if score < 0.999 {
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
        .overlay(alignment: .topTrailing) {
            // 纯单人组标注「为什么算单人」(165 诊断,166 补盲区):reason≠0 是异常
            // 路径;reason=0 且检出多张脸全判成本人——正常只该发生在镜像/海报/
            // 本人双脸,若用户肉眼看到别人就该长按照片看逐脸明细。
            // 第十四批:限定纯单人组——「主角 与其他人」组的主角也是 1 个,
            // 不排除会把角标误挂到必含 U 脸的合影上,"均判本人"文案失真
            if members.count == 1, !withOthers, let r = reasons[ref] {
                if r.reason != 0 {
                    Text(r.reason == 1 ? String(format: "弱分回退·%d%%", r.note)
                         : r.reason == 2 ? "iCloud未取到"
                         : "复检未检出人脸")
                        .font(.system(size: 9))
                        .padding(.horizontal, 3).padding(.vertical, 2)
                        .background(.orange.opacity(0.85), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.white)
                        .padding(4)
                } else if r.note >= 2 {
                    Text("见\(r.note)脸·均判本人")
                        .font(.system(size: 9))
                        .padding(.horizontal, 3).padding(.vertical, 2)
                        .background(.orange.opacity(0.85), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.white)
                        .padding(4)
                }
            }
        }
        .onLongPressGesture {
            debugText = details[ref] ?? "无逐脸记录(判决来自旧缓存/旧快照),重新归组一次即可生成"
            showingDebug = true
        }
    }

    /// 移出所选:全部人脸退出聚类(face_block),父级重新归组
    private func excludeSelected() {
        guard !selected.isEmpty, !excluding else { return }
        excluding = true
        let refs = Array(selected)
        Task {
            do {
                try app.store.blockFacePhotos(refs)
                await MainActor.run {
                    excluding = false
                    selecting = false
                    selected = []
                    app.notify("已移出 \(refs.count) 张照片,重新归组中")
                    onChanged?()
                }
            } catch {
                await MainActor.run { excluding = false; app.fail(error) }
            }
        }
    }

    private func applyName() {
        guard let mi = namingMember, !applying else { return }
        let name = nameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        applying = true
        let refs = personRefs[mi] ?? []
        // 重命名=覆盖:先删该人身上的旧名标签再写新名,不能累积两个名字
        // (155 用户点名:第二次命名要覆盖第一次)
        let oldName = personNames[mi] ?? ""
        let cleanup = (!oldName.isEmpty && oldName != name) ? oldName : nil
        Task {
            do {
                if let cleanup {
                    try app.store.deleteUserTag(cleanup, refs: refs)
                }
                // 批量通道:向量只嵌一次(此前逐张推理,命名两千张要几十分钟)
                try await app.addUserTag(name, toRefs: refs)
                await MainActor.run {
                    applying = false
                    personNames[mi] = name
                    namingMember = nil
                    app.notify(cleanup != nil
                               ? "已把「\(cleanup)」改名为「\(name)」(\(refs.count) 张照片)"
                               : "已给 \(refs.count) 张照片打上「\(name)」标签,搜索「\(name)」即可找到")
                }
            } catch {
                await MainActor.run { applying = false; app.fail(error) }
            }
        }
    }
}
