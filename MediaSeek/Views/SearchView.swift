import SwiftUI
import Photos

struct SearchView: View {
    @EnvironmentObject var app: AppModel

    @State private var query = ""
    @State private var scope: SearchScope = .all
    @State private var results: [DisplayHit] = []
    @State private var searching = false
    @State private var searchedOnce = false
    @State private var elapsedMs = 0
    @State private var selectedHit: DisplayHit?
    @State private var exporting = false

    private let exampleQueries = ["一只猫", "海边的日落", "有人的合影", "会议纪要", "发票 PDF"]

    private let columns = [GridItem(.adaptive(minimum: 105), spacing: 10)]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                searchField
                scopePicker
                if !app.models.anyReady {
                    modelBanner
                }
                statusLine
                resultGrid
            }
            .navigationTitle("智搜")
            .sheet(item: $selectedHit) { hit in
                DetailSheet(hit: hit)
            }
        }
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("用一句话找照片、视频、文件…", text: $query)
                .submitLabel(.search)
                .onSubmit(runSearch)
            if !query.isEmpty {
                Button {
                    query = ""
                    results = []
                    searchedOnce = false
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
            }
        }
        .padding(10)
        .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var scopePicker: some View {
        Picker("范围", selection: $scope) {
            ForEach(SearchScope.allCases) { s in
                Text(s.rawValue).tag(s)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private var modelBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "brain")
            Text("模型未就绪,请到「设置」导入模型包")
                .font(.footnote)
        }
        .frame(maxWidth: .infinity)
        .padding(10)
        .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal)
    }

    @ViewBuilder
    private var statusLine: some View {
        if searching {
            ProgressView("正在检索…").padding(.vertical, 6)
        } else if searchedOnce {
            HStack(spacing: 10) {
                Text("找到 \(results.count) 个结果 · \(elapsedMs) ms")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                if !results.isEmpty {
                    Spacer()
                    Menu {
                        Button {
                            exportToAlbum()
                        } label: {
                            Label("存入系统相册(新建专辑)", systemImage: "photo.stack")
                        }
                        Button {
                            exportToFiles()
                        } label: {
                            Label("导出到「文件」App", systemImage: "folder")
                        }
                    } label: {
                        if exporting {
                            ProgressView()
                        } else {
                            Label("导出", systemImage: "square.and.arrow.up")
                                .font(.footnote)
                        }
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
        } else {
            chips
        }
    }

    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(exampleQueries, id: \.self) { q in
                    Button(q) {
                        query = q
                        runSearch()
                    }
                    .font(.footnote)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Color(.systemGray6), in: Capsule())
                }
            }
            .padding(.horizontal)
        }
        .padding(.bottom, 4)
    }

    private var resultGrid: some View {
        ScrollView {
            if results.isEmpty && searchedOnce && !searching {
                ContentUnavailableView("没有找到相关内容",
                                       systemImage: "questionmark.folder",
                                       description: Text("试试换一种说法,或先在「资料库」同步索引"))
                    .padding(.top, 60)
            } else {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(results) { hit in
                        Button { selectedHit = hit } label: { HitCell(hit: hit) }
                            .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    private func exportToAlbum() {
        guard !exporting, !results.isEmpty else { return }
        exporting = true
        let q = query.trimmingCharacters(in: .whitespaces)
        Task {
            do {
                let n = try await SearchExporter.saveToAlbum(title: q, hits: results)
                await MainActor.run { app.notify("已把 \(n) 个项目存入相册「搜索·\(q)」") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func exportToFiles() {
        guard !exporting, !results.isEmpty else { return }
        exporting = true
        let q = query.trimmingCharacters(in: .whitespaces)
        Task {
            do {
                let (n, url) = try await SearchExporter.exportToFiles(
                    folderName: q, hits: results,
                    photo: app.photoLib, imports: app.imports)
                await MainActor.run { app.notify("已导出 \(n) 个文件\n位置:「文件」App → 智搜 → 导出 → \(url.lastPathComponent)") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func runSearch() {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return }
        searching = true
        searchedOnce = true
        Task.detached(priority: .userInitiated) { [scope] in
            var hits: [DisplayHit] = []
            var ms = 0
            do {
                let start = Date()
                hits = try await app.search.search(q, scope: scope)
                ms = Int(Date().timeIntervalSince(start) * 1000)
            } catch {
                await app.fail(error)
            }
            await MainActor.run {
                results = hits
                elapsedMs = ms
                searching = false
            }
        }
    }
}
