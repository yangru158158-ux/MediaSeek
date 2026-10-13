import Foundation
import Compression
import Photos
import AVFoundation
import UIKit
import Vision
import PDFKit
import UniformTypeIdentifiers

/// 索引编排器:照片→SigLIP 图向量 + Vision 标签(走 EmbeddingGemma 中文语义),
/// 视频→关键帧向量,文件→文件名/文本块向量。后台顺序处理,实时进度。
@MainActor
final class IndexingCoordinator: ObservableObject {
    enum Phase: Equatable {
        case idle, waitingModel, photos, videos, faces, files, speech, done, failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var processed = 0
    @Published private(set) var total = 0
    @Published private(set) var isRunning = false
    @Published private(set) var errorCount = 0
    @Published private(set) var firstErrorMessage: String?
    @Published private(set) var lastSyncAt: Date?

    /// 跨并发记录第一个处理错误(线程安全)
    final class FirstErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var message: String?
        func set(_ m: String) {
            lock.lock()
            if message == nil { message = m }
            lock.unlock()
        }
        func get() -> String? {
            lock.lock(); defer { lock.unlock() }
            return message
        }
    }

    var videoFramesPerVideo: Int {
        get { UserDefaults.standard.object(forKey: "videoFrames") as? Int ?? 3 }
        set { UserDefaults.standard.set(newValue, forKey: "videoFrames") }
    }

    /// 中文语义标签(Vision+Gemma):增强中文物体查询,但每张照片多两次推理,慢 2-3 倍
    /// 已索引照片数(空态提示用)
    var indexedPhotoCount: Int { store.countPhotos() }

    /// 视频索引覆盖度一行摘要(搜索页视频范围空态展示)
    var videoCoverageSummary: String? {
        guard photo.isAuthorized, let stats = try? store.videoFrameStats() else { return nil }
        let total = photo.totalVideoCount()
        return "视频帧已索引 \(stats.videos) / 相册共 \(total) 部;改抽帧数后同步会自动重索引"
    }

    var chineseLabels: Bool {
        get { UserDefaults.standard.object(forKey: "chineseLabels") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "chineseLabels") }
    }

    private var processedLocal = 0
    private var rerunPending = false

    private let store: VectorStore
    private let models: ModelManager
    private let photo: PhotoLibraryService
    private let imports: ImportLibrary
    private var task: Task<Void, Never>?

    init(store: VectorStore, models: ModelManager, photo: PhotoLibraryService, imports: ImportLibrary) {
        self.store = store
        self.models = models
        self.photo = photo
        self.imports = imports
    }

    var phaseText: String {
        switch phase {
        case .idle: return "未开始"
        case .waitingModel: return "等待模型加载…"
        case .photos: return "正在索引照片"
        case .videos: return "正在索引视频"
        case .faces: return "正在补人脸数据"
        case .files: return "正在索引文件"
        case .speech: return "正在转写视频语音"
        case .done: return "索引完成"
        case .failed(let m): return "失败:\(m)"
        }
    }

    // MARK: - 入口

    func startAutoSyncIfAuthorized() {
        guard photo.isAuthorized else { return }
        runIncremental()
    }

    func runIncremental() {
        startRun(full: false)
    }

    func rebuildAll() {
        startRun(full: true)
    }

    func cancel() {
        task?.cancel()
        task = nil
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func startRun(full: Bool) {
        guard !isRunning else {
            // 已有任务在跑:记一笔,本轮结束自动补跑(治「导入第二轮文件一直待索引」)
            rerunPending = true
            return
        }
        isRunning = true
        processed = 0
        processedLocal = 0
        total = 0
        errorCount = 0
        firstErrorMessage = nil
        phase = models.bothReady ? .photos : .waitingModel
        // 重建期间禁止自动锁屏:保持前台+插电即可连续索引(iOS 后台会冻结 App)
        UIApplication.shared.isIdleTimerDisabled = true
        let errors = FirstErrorBox()
        task = Task { [full, errors] in
            await self.run(full: full, errors: errors)
            self.isRunning = false
            self.task = nil
            UIApplication.shared.isIdleTimerDisabled = false
            // 跑完后若期间有新导入排队,立刻补跑一轮(文件秒级变「已索引」)
            if self.rerunPending {
                self.rerunPending = false
                self.startRun(full: false)
            }
        }
    }

    private func run(full: Bool, errors: FirstErrorBox) async {
        do {
            let ready = await models.waitReady()
            guard ready else {
                phase = .failed("模型未就绪,请先在「设置」中导入模型包")
                return
            }
            // 全量重建不再先清空旧索引:全量覆盖更新,搜索全程不断档
            // (结束时按现有相册清单自动清理已删除照片的旧行)
            // 文件量级小(几十个)先跑完,用户立刻能看到「已索引」;相册大批量随后
            try await syncImportedFiles()
            try await syncPhotoLibrary(forceAll: full, errors: errors)
            // 语音转写(开关制)放最尾:核心索引不被数小时的转写挡道
            try await syncSpeechTranscripts(errors: errors)
            await prebuildLabelVocab()   // 重建完成后预建路由词表,首次搜索不再有一次性延迟

            lastSyncAt = Date()
            phase = .done
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// 把库内去重标签全部嵌入词向量表(语义路由用),量级几百条、一次性
    private func prebuildLabelVocab() async {
        guard let gemma = models.gemma,
              let labels = try? store.allDistinctLabels(), !labels.isEmpty else { return }
        let known = Set(((try? store.labelVocab()) ?? []).map { $0.label })
        // 推理必须放后台线程:主线程逐个嵌几百个标签会卡死主线程,
        // 触发 scene-update 看门狗(10 秒上限)被系统击杀(18:01/18:03 两次崩溃根因)
        let missing = labels.prefix(2000).filter { !known.contains($0) }
        guard !missing.isEmpty else { return }
        await Task.detached(priority: .utility) { [store] in
            for label in missing {
                guard let v = try? gemma.embedDocument(label) else { continue }
                try? store.saveLabelVec(label, vec: v)
                try? await Task.sleep(nanoseconds: 2_000_000)   // 每 2ms 让路,防持续满载
            }
        }.value
    }

    // MARK: - 相册

    private func syncPhotoLibrary(forceAll: Bool = false, errors: FirstErrorBox) async throws {
        let fetch = photo.fetchAllAssets()
        if forceAll {
            // 全量重建 = 给失败黑名单一次重试机会(iCloud 重新下载/网络恢复后)
            UserDefaults.standard.removeObject(forKey: "vidAssetFail")
            UserDefaults.standard.removeObject(forKey: "vidAssetFailCnt")
            // 语音转写:只清失败记账,不清 speechDone(已转写文本独立有效,
            // 全量重建把几小时转写重跑一遍纯属浪费)
            UserDefaults.standard.removeObject(forKey: "speechFail")
            UserDefaults.standard.removeObject(forKey: "speechFailCnt")
        }
        // 图像模型更换检测(so400m 等在线升级):模型加载时记下当前维度
        // (ModelManager 写 clipModelDim),与库内向量维度不一致 = 换过嵌入模型,
        // 旧向量点积无意义——清空 clip 向量(OCR/标签/人脸保留),本轮全量重嵌
        if let wantDim = UserDefaults.standard.object(forKey: "clipModelDim") as? Int,
           store.clipVectorDim() > 0, store.clipVectorDim() != wantDim {
            try store.wipeClipVectors()
        }
        // 人脸模型更换检测:模型加载时记下指纹(faceModelSize,如 mbf→r50 升级),
        // 与库内已嵌指纹不一致 = 换过嵌入模型,旧向量空间不通用——清空重嵌
        let faceWant = UserDefaults.standard.integer(forKey: "faceModelSize")
        let faceIndexed = UserDefaults.standard.integer(forKey: "faceIndexedSize")
        if faceWant > 0, faceWant != faceIndexed {
            if store.faceStats().faces > 0 { try store.wipeFaceVectors() }
            UserDefaults.standard.set(faceWant, forKey: "faceIndexedSize")
        }
        // 全量重建:所有照片视为新照片全量覆盖;增量同步:跳过已索引。
        // 「已索引」只认 clip 向量行本身——标签/人脸行不算数,否则换模型
        // 清空向量后照片仍被当成已索引,一张都不会重建
        var known = forceAll ? Set<String>() : try store.refKeysWithClipVectors()
        // 增量自动补全:已有向量但缺 OCR 原文的照片视为待补(原位覆盖),
        // 跑多少存多少、中断可续,无需全量重建(无字照片有 "" 占位行,不会反复补)
        if !forceAll {
            for key in try store.photoRefKeysMissingOCR() { known.remove(key) }
            // 抽帧数变更:视频记录了「处理时用的抽帧数」,设置真变了才重处理。
            // 不能直接比帧行数——部分帧黑(114 起跳过)后行数天然小于设置值,
            // 直接比会无限重跑(2026-10-11 用户实测 25 部视频反复索引)
            if let cfg = UserDefaults.standard.dictionary(forKey: "vidFrameCfg") as? [String: Int],
               let stats = try? store.videoFrameStats(), stats.videos > 0 {
                for key in stats.perVideo.keys {
                    if let used = cfg[key], used != videoFramesPerVideo { known.remove(key) }
                }
            }
            // 帧OCR自愈:有帧向量但缺 OCR 行的视频原位重处理一次
            // (无文字视频写 "" 占位行,不会循环)
            if let missing = try? store.videoRefKeysMissingOCR(), !missing.isEmpty {
                known.subtract(missing)
            }
            // 帧标签自愈:有帧向量但缺标签行的视频原位重处理一次
            // (标签是视频侧唯一可靠召回通道,2026-10-11 马尔代夫抖音噪声实锤后补)
            if let frameStats = try? store.videoFrameStats(), frameStats.videos > 0,
               let labelRefs = try? store.refKeys(kinds: [.videoFrameLabel]) {
                for key in frameStats.perVideo.keys where !labelRefs.contains(key) {
                    known.remove(key)
                }
            }
        }
        // 人脸补全:已索引但未做过人脸处理的照片(装上人脸模型后的一次性回填)。
        // 走轻量管线(只检测+嵌入,不重跑 SigLIP/OCR/标签),可中断可续;
        // 人脸模型未装时跳过——face_done 不记账,装上后下一轮同步自动开始补
        let faceEmbedder = models.face
        let missingFaces: Set<String> = (!forceAll && faceEmbedder != nil)
            ? try store.photoRefKeysMissingFaces() : []

        let recentlyDeleted = Self.recentlyDeletedIDs()
        // 黑屏录音类视频记账(全帧黑、无可索引内容):跳过抽帧,不再每轮同步反复扫
        let fullyBlack = Set(UserDefaults.standard.stringArray(forKey: "vidAllBlack") ?? [])
        // 视频读取/处理连续失败记账(3 次拉黑):iCloud 未下载或损坏的视频每轮同步
        // 都会被当新视频重做一遍且无声空转(132 用户实测 7 个视频每轮都索引);
        // 拉黑后跳过,「全量重建索引」会清空黑名单重试(下载/网络恢复后可恢复)
        let assetFailed = Set(UserDefaults.standard.stringArray(forKey: "vidAssetFail") ?? [])

        var libIDs = Set<String>()
        var newPhotos: [PHAsset] = []
        var newVideos: [PHAsset] = []
        var faceBackfill: [PHAsset] = []
        fetch.enumerateObjects { asset, _, _ in
            // 「最近删除」里的照片/视频不入库:不新增索引,旧行按 stale 一并清除
            guard recentlyDeleted.contains(asset.localIdentifier) == false else { return }
            // 全黑视频:不进 libIDs(遗留行顺带清除),也不重新处理
            guard !fullyBlack.contains(asset.localIdentifier) else { return }
            // 连续失败已拉黑的视频:同全黑视频处理,不再每轮空转
            guard !assetFailed.contains(asset.localIdentifier) else { return }
            libIDs.insert(asset.localIdentifier)
            if known.contains(asset.localIdentifier) {
                // 已索引但缺人脸数据 → 人脸补全队列(轻量)
                if missingFaces.contains(asset.localIdentifier), asset.mediaType == .image {
                    faceBackfill.append(asset)
                }
                return
            }
            if asset.mediaType == .image {
                newPhotos.append(asset)
            } else if asset.mediaType == .video {
                newVideos.append(asset)
            }
        }
        try store.removeRefs(kinds: [.photo, .photoLabel, .userTag, .videoFrame, .videoFrameLabel, .face], notIn: libIDs)

        total += newPhotos.count + newVideos.count + faceBackfill.count
        // 视频量级小先处理:几分钟后即可被「视频」范围搜到,不必等全部照片
        phase = .videos
        for asset in newVideos {
            try Task.checkCancellation()
            await embedVideo(asset, errors: errors)
            processedLocal += 1
            processed = processedLocal
        }
        // 人脸补全先于新照片:人物归组数据尽早可用,且与照片全量嵌入互不阻塞
        if !faceBackfill.isEmpty, let faceEmbedder {
            phase = .faces
            let fstore = self.store
            let fphoto = self.photo
            var faceAbort: String? = nil
            await withTaskGroup(of: Int.self) { group in
                var index = 0
                let maxConcurrent = 2
                while index < min(maxConcurrent, faceBackfill.count) {
                    let asset = faceBackfill[index]; index += 1
                    group.addTask { await Self.faceOnlyWork(
                        asset: asset, photo: fphoto, store: fstore, face: faceEmbedder, errors: errors) }
                }
                while !group.isEmpty {
                    let fails = await group.next() ?? 0
                    errorCount += fails
                    processedLocal += 1
                    if processedLocal % 5 == 0 { processed = processedLocal }
                    if fails > 0, firstErrorMessage == nil {
                        firstErrorMessage = errors.get()
                    }
                    if errorCount >= 20, let msg = errors.get() {
                        faceAbort = "人脸补全连续失败已停止:\(msg)"
                        break
                    }
                    if Task.isCancelled { break }
                    if index < faceBackfill.count {
                        let asset = faceBackfill[index]; index += 1
                        group.addTask { await Self.faceOnlyWork(
                            asset: asset, photo: fphoto, store: fstore, face: faceEmbedder, errors: errors) }
                    }
                }
            }
            if let faceAbort { throw MSError(faceAbort) }
        }
        phase = .photos
        // 双路并发:解码/下载与 ANE 推理流水线重叠
        let useLabels = chineseLabels
        let clip = models.siglip
        let gemma = models.gemma
        let face = models.face
        let store = self.store
        let photo = self.photo
        var abortReason: String? = nil
        await withTaskGroup(of: Int.self) { group in
            var index = 0
            let maxConcurrent = 2
            while index < min(maxConcurrent, newPhotos.count) {
                let asset = newPhotos[index]; index += 1
                group.addTask { await Self.embedPhotoWork(
                    asset: asset, photo: photo, store: store,
                    clip: clip, gemma: gemma, face: face, useLabels: useLabels, errors: errors) }
            }
            while !group.isEmpty {
                let fails = await group.next() ?? 0
                errorCount += fails
                processedLocal += 1
                if processedLocal % 5 == 0 { processed = processedLocal }
                if fails > 0, firstErrorMessage == nil {
                    firstErrorMessage = errors.get()
                }
                // 连续失败 20 张 = 系统性故障,不再排新任务(在组外统一抛错停止)
                if errorCount >= 20, let msg = errors.get() {
                    abortReason = "连续处理失败已停止:\(msg)"
                    break
                }
                if Task.isCancelled { break }
                if index < newPhotos.count {
                    let asset = newPhotos[index]; index += 1
                    group.addTask { await Self.embedPhotoWork(
                        asset: asset, photo: photo, store: store,
                        clip: clip, gemma: gemma, face: face, useLabels: useLabels, errors: errors) }
                }
            }
        }
        if let abortReason { throw MSError(abortReason) }
        try Task.checkCancellation()   // 「停止」从这里立即生效
        processed = processedLocal
    }

    // MARK: - 视频语音转写(设置开关,默认关)

    /// 开关开启时,同步收尾把还没转写过的视频逐部抽音轨做端侧识别,
    /// 文本追加进 ocr_text(文字: 精确命中)。耗时≈音频时长,可中断可续;
    /// 无语音/转写为空也记账 speechDone(与 OCR "" 占位同理,防每轮重跑)。
    /// 失败连续 3 次进 speechFail 黑名单(与视频资产黑名单同范式),
    /// 「全量重建」清黑名单给重试机会;连续 10 部失败视为系统性故障,本轮放弃
    private func syncSpeechTranscripts(errors: FirstErrorBox) async throws {
        guard UserDefaults.standard.bool(forKey: "speechEnabled") else { return }

        let done = Set(UserDefaults.standard.stringArray(forKey: "speechDone") ?? [])
        let failed = Set(UserDefaults.standard.stringArray(forKey: "speechFail") ?? [])
        let rd = Self.recentlyDeletedIDs()
        let black = Set(UserDefaults.standard.stringArray(forKey: "vidAllBlack") ?? [])
        let assetFailed = Set(UserDefaults.standard.stringArray(forKey: "vidAssetFail") ?? [])
        var pending: [PHAsset] = []
        let fetch = PHAsset.fetchAssets(with: .video, options: nil)
        fetch.enumerateObjects { asset, _, _ in
            let id = asset.localIdentifier
            guard !rd.contains(id), !black.contains(id), !assetFailed.contains(id),
                  !done.contains(id), !failed.contains(id) else { return }
            pending.append(asset)
        }
        guard !pending.isEmpty else { return }
        // 有活干才碰权限:语音权限弹窗只在首轮转写前出现一次
        let authorized = await SpeechTranscriber.requestAuthorization()
        UserDefaults.standard.set(!authorized, forKey: "speechDenied")
        guard authorized else { return }
        let supported = SpeechTranscriber.onDeviceSupported()
        UserDefaults.standard.set(!supported, forKey: "speechUnsupported")
        guard supported else { return }

        total += pending.count
        phase = .speech
        let store = self.store
        let photo = self.photo
        var consecutiveFails = 0
        for asset in pending {
            try Task.checkCancellation()
            let ok = await Self.speechWork(asset: asset, photo: photo, store: store, errors: errors)
            processedLocal += 1
            processed = processedLocal
            if ok {
                consecutiveFails = 0
            } else {
                errorCount += 1
                if firstErrorMessage == nil { firstErrorMessage = errors.get() }
                consecutiveFails += 1
                if consecutiveFails >= 10 {
                    errors.set("语音转写连续失败已停止,其余视频下轮同步重试")
                    break
                }
            }
        }
    }

    /// 单部视频转写。成功(含无语音)返回 true 并记账;失败返回 false 记失败计数
    nonisolated private static func speechWork(
        asset: PHAsset, photo: PhotoLibraryService, store: VectorStore,
        errors: FirstErrorBox) async -> Bool {
        let id = asset.localIdentifier
        do {
            guard let avAsset = await photo.avAsset(for: asset) else {
                errors.set("视频读取失败(可能未下载到本机):\(id)")
                markSpeechFail(id)
                return false
            }
            try Task.checkCancellation()
            let text = try await SpeechTranscriber.transcribeVideo(avAsset: avAsset)
            try store.appendSpeechText(refKey: id, text: text)
            var done = Set(UserDefaults.standard.stringArray(forKey: "speechDone") ?? [])
            done.insert(id)
            UserDefaults.standard.set(Array(done), forKey: "speechDone")
            return true
        } catch is CancellationError {
            return true   // 用户停止:不记账不算错,下轮同步从未转写的继续
        } catch {
            errors.set("语音转写失败:\(error.localizedDescription)")
            markSpeechFail(id)
            return false
        }
    }

    /// 语音转写失败计数:同一视频连续失败 3 次进 speechFail 黑名单不再重试
    nonisolated private static func markSpeechFail(_ refKey: String) {
        var cnt = (UserDefaults.standard.dictionary(forKey: "speechFailCnt") as? [String: Int]) ?? [:]
        cnt[refKey, default: 0] += 1
        UserDefaults.standard.set(cnt, forKey: "speechFailCnt")
        if cnt[refKey] ?? 0 >= 3 {
            var set = Set(UserDefaults.standard.stringArray(forKey: "speechFail") ?? [])
            set.insert(refKey)
            UserDefaults.standard.set(Array(set), forKey: "speechFail")
        }
    }

    /// 「最近删除」智能相册的资产 ID。系统未公开对应枚举,
    /// 通行做法:按智能相册 subtype 原始值 1000000201 识别(中英文标题兜底);
    /// 若系统未暴露该相册,返回空集,行为退化为不过滤。
    private static func recentlyDeletedIDs() -> Set<String> {
        var ids = Set<String>()
        let cols = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .any, options: nil)
        cols.enumerateObjects { col, _, _ in
            let isRD = col.assetCollectionSubtype.rawValue == 1000000201
                || col.localizedTitle == "最近删除" || col.localizedTitle == "Recently Deleted"
            guard isRD else { return }
            PHAsset.fetchAssets(in: col, options: nil).enumerateObjects { asset, _, _ in
                ids.insert(asset.localIdentifier)
            }
        }
        return ids
    }

    nonisolated private static func embedPhotoWork(
        asset: PHAsset, photo: PhotoLibraryService, store: VectorStore,
        clip: SigLIPEmbedder?, gemma: GemmaTextEmbedder?, face: FaceEmbedder?,
        useLabels: Bool, errors: FirstErrorBox) async -> Int {
        guard let clip, let gemma else { return 0 }
        do {
            // 2048px 用于 OCR 文字识别(拍屏角度/小字也尽量认出);320px 用于嵌入向量
            guard let full = try await photo.image(for: asset, maxPixel: 2048) else {
                errors.set("取图失败(可能 iCloud 未下载或存储空间不足):\(asset.localIdentifier)")
                return 1
            }
            try Task.checkCancellation()   // 取消:取图后立即中断
            guard let small = Self.downscaled(full, to: 320) else {
                errors.set("图像降采样失败:\(asset.localIdentifier)")
                return 1
            }
            let colorBucket = PhotoColor.bucket(of: small)
            let vec = try clip.embedImage(small)
            try store.upsert(kind: .photo, refKey: asset.localIdentifier, space: clip.space,
                             vector: vec, title: nil, date: asset.creationDate, color: colorBucket)

            // OCR 文字层:屏幕文字/证件名/编号可被精确检索
            // 无文字也写 "" 占位行:增量同步靠「缺 OCR 行」识别待补照片,
            // 没有占位的话风景照等无字照片会被每轮反复重处理
            let ocr = Self.recognizeText(in: full)
            try store.saveOCR(refKey: asset.localIdentifier, text: ocr ?? "")

            // 人脸层:检测+对齐+身份向量(人物归组数据源);模型未装则跳过且不记账
            try Self.embedFaces(full: full, assetID: asset.localIdentifier,
                                date: asset.creationDate, face: face, store: store)

            guard useLabels, let labels = Self.visionLabels(cg: small), !labels.isEmpty else { return 0 }
            try Task.checkCancellation()
            let lvec = try gemma.embedDocument("photo tags: \(labels)")
            try store.upsert(kind: .photoLabel, refKey: asset.localIdentifier, space: gemma.space,
                             vector: lvec, title: labels, date: asset.creationDate)
            return 0
        } catch {
            if Task.isCancelled { return 0 }   // 用户停止不算错误
            errors.set("照片处理失败:\(error)")
            return 1
        }
    }

    /// 人脸补全(轻量管线):对已索引但缺人脸数据的照片只做检测+嵌入,
    /// 不重跑 SigLIP/OCR/标签——14k 张约 30-60 分钟,可中断可续
    nonisolated private static func faceOnlyWork(
        asset: PHAsset, photo: PhotoLibraryService, store: VectorStore,
        face: FaceEmbedder, errors: FirstErrorBox) async -> Int {
        do {
            guard let full = try await photo.image(for: asset, maxPixel: 2048) else {
                errors.set("取图失败(可能 iCloud 未下载):\(asset.localIdentifier)")
                return 1
            }
            try Task.checkCancellation()
            try Self.embedFaces(full: full, assetID: asset.localIdentifier,
                                date: asset.creationDate, face: face, store: store)
            return 0
        } catch {
            if Task.isCancelled { return 0 }
            errors.set("人脸处理失败:\(error)")
            return 1
        }
    }

    /// 检测人脸并对齐嵌入入库。先清旧行防残留(重处理时脸数可能变少);
    /// 无论检出几张(含 0 张)都写 face_done 记账,防无脸照片反复重处理
    nonisolated private static func embedFaces(full: CGImage, assetID: String,
                                               date: Date?, face: FaceEmbedder?,
                                               store: VectorStore) throws {
        guard let face else { return }
        try store.removeByRef(kind: .face, refKey: assetID)
        for (i, obs) in FaceKit.detect(cg: full).enumerated() {
            guard let aligned = FaceKit.alignedFace(source: full, observation: obs) else { continue }
            let vec = try face.embed(aligned: aligned)
            try store.upsert(kind: .face, refKey: assetID, frameIndex: i, space: face.space,
                             vector: vec, title: nil, date: date)
        }
        try store.markFaceDone(refKey: assetID)
    }

    private func embedVideo(_ asset: PHAsset, errors: FirstErrorBox) async {
        do {
            guard let clip = models.siglip else { return }
            let gemma = models.gemma
            let store = self.store
            let photo = self.photo
            let frameCount = videoFramesPerVideo
            let refKey = asset.localIdentifier
            guard let avAsset = await photo.avAsset(for: asset) else {
                // iCloud 未下载/损坏的视频读不出 AVAsset:原先这里无声 return,
                // 不报错不记账不写行 → 每轮同步都被当新视频重做(132 实测 7 个视频
                // 每轮空转)。改为可见失败 + 计数拉黑
                errors.set("视频读取失败(可能未下载到本机或已损坏)")
                errorCount += 1
                markVideoAssetFail(refKey)
                return
            }
            try await Task.detached(priority: .utility) {
                // 先清旧帧再采新帧:重索引(如抽帧数变更)后不残留旧帧行
                try? store.removeFrames(refKey: asset.localIdentifier)
                // 2048px 抽帧(帧 OCR 需要照片同级的分辨率),嵌入/标签/颜色用 320 降采样
                let frames = try await FrameSampler.sample(asset: avAsset, count: frameCount, maxPixel: 2048)
                var rows = 0
                var ocrParts: [String] = []
                for (i, frame) in frames.enumerated() {
                    // 近全黑帧不入索引:黑屏录音类视频没有视觉内容,入库后只会被
                    // 视频相对标定灌进任何搜索结果(2026-10-11 用户实测 water/常住
                    // 混入"只有声音"黑视频)
                    if PhotoColor.isTooDark(frame) { continue }
                    guard let small = Self.downscaled(frame, to: 320) else { continue }
                    let vec = try clip.embedImage(small)
                    try store.upsert(kind: .videoFrame, refKey: asset.localIdentifier, frameIndex: i,
                                     space: clip.space, vector: vec, title: nil, date: asset.creationDate,
                                     color: PhotoColor.bucket(of: small))
                    rows += 1
                    // 帧标签:视频侧唯一可靠的召回通道(与照片同款 Vision 分类)。
                    // SigLIP 图文分数在本库区分度不足(马尔代夫 vs 抖音噪声分不开),
                    // 标签零幻觉:海滩/水上屋帧得 beach/island/water,抖音室内帧得
                    // people/clothing,词典词不命中即不灌入
                    if let gemma, let labels = Self.visionLabels(cg: small) {
                        let lvec = try gemma.embedDocument("photo tags: \(labels)")
                        try store.upsert(kind: .videoFrameLabel, refKey: asset.localIdentifier,
                                         frameIndex: i, space: gemma.space, vector: lvec,
                                         title: labels, date: asset.creationDate)
                    }
                    // 帧OCR(与照片同管线):视频画面里的文字可被 文字: 命中
                    if let t = Self.recognizeText(in: frame) {
                        ocrParts.append("【帧\(i)】\(t)")
                    }
                }
                // OCR 原文与照片同表同键;无文字写 "" 占位,防「缺 OCR 行」自愈反复重处理
                try store.saveOCR(refKey: asset.localIdentifier,
                                  text: String(ocrParts.joined(separator: "\n").prefix(12_000)))
                // 记账:≥1 帧入库记下当时的抽帧数(「抽帧数变更」判断的唯一依据,
                // 直接比行数会因黑帧跳过而无限重跑);采到帧但全黑(0 行)拉黑名单,
                // 同步直接跳过;抽帧失败(0 帧)不记,下次同步重试
                if !frames.isEmpty {
                    if rows > 0 {
                        var cfg = (UserDefaults.standard.dictionary(forKey: "vidFrameCfg") as? [String: Int]) ?? [:]
                        cfg[asset.localIdentifier] = frameCount
                        UserDefaults.standard.set(cfg, forKey: "vidFrameCfg")
                        // 成功落库清失败计数(此前几次失败可能是网络/时机问题)
                        var cnt = (UserDefaults.standard.dictionary(forKey: "vidAssetFailCnt") as? [String: Int]) ?? [:]
                        if cnt.removeValue(forKey: asset.localIdentifier) != nil {
                            UserDefaults.standard.set(cnt, forKey: "vidAssetFailCnt")
                        }
                    } else {
                        var set = Set(UserDefaults.standard.stringArray(forKey: "vidAllBlack") ?? [])
                        set.insert(asset.localIdentifier)
                        UserDefaults.standard.set(Array(set), forKey: "vidAllBlack")
                    }
                }
            }.value
        } catch is CancellationError {
            return   // 用户停止不算错误
        } catch {
            errors.set("视频处理失败:\(error)")
            errorCount += 1
            markVideoAssetFail(asset.localIdentifier)
        }
    }

    /// 视频读取/处理失败计数:同一视频连续失败 3 次进黑名单 vidAssetFail,
    /// 增量同步不再重试(避免每轮空转);成功处理会在 embedVideo 落库时清零计数
    private func markVideoAssetFail(_ refKey: String) {
        var cnt = (UserDefaults.standard.dictionary(forKey: "vidAssetFailCnt") as? [String: Int]) ?? [:]
        cnt[refKey, default: 0] += 1
        UserDefaults.standard.set(cnt, forKey: "vidAssetFailCnt")
        if cnt[refKey] ?? 0 >= 3 {
            var set = Set(UserDefaults.standard.stringArray(forKey: "vidAssetFail") ?? [])
            set.insert(refKey)
            UserDefaults.standard.set(Array(set), forKey: "vidAssetFail")
        }
    }

    /// Vision 内置 1300+ 类分类器,取高置信度标签(如 cat / dog / beach)
    nonisolated private static func visionLabels(cg: CGImage) -> String? {
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            let labels = (request.results ?? [])
                .filter { $0.confidence >= 0.3 }
                .prefix(6)
                .map { $0.identifier }
            return labels.isEmpty ? nil : labels.joined(separator: ", ")
        } catch {
            return nil
        }
    }

    /// Vision 文字识别(OCR):简繁中文+英文,精确检索层的数据来源
    nonisolated private static func recognizeText(in cg: CGImage) -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
        request.usesLanguageCorrection = false
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            let lines = (request.results as? [VNRecognizedTextObservation])?
                .compactMap { $0.topCandidates(1).first?.string } ?? []
            let joined = lines.joined(separator: "\n")
            return joined.isEmpty ? nil : String(joined.prefix(2000))
        } catch {
            return nil
        }
    }

    /// 高分辨率图降采样到嵌入模型需要的尺寸(资源不足时返回 nil,不再强解包)
    nonisolated private static func downscaled(_ image: CGImage, to side: Int) -> CGImage? {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: side * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        return ctx.makeImage()
    }

    // MARK: - 导入文件

    private func syncImportedFiles() async throws {
        let files = imports.allFiles().filter { !$0.indexed }
        total += files.count
        phase = .files
        for file in files {
            try Task.checkCancellation()
            await embedImported(file)
            processed += 1
        }
    }

    func embedImported(_ record: ImportedFile) async {
        do {
            guard let gemma = models.gemma else { throw MSError("文本模型未就绪") }
            let clip = models.siglip
            let imports = self.imports
            let store = self.store
            let frameCount = videoFramesPerVideo
            try await Task.detached(priority: .utility) {
                let url = try imports.resolveURL(record)
                defer { url.stopAccessingSecurityScopedResource() }

                let type = UTType(filenameExtension: url.pathExtension) ?? .data
                let date = record.addedAt

                // 文件名总是嵌入(基础可检索性)
                let nameVec = try gemma.embedDocument("filename: \(record.name)")
                try store.upsert(kind: .file, refKey: record.id, space: gemma.space,
                                 vector: nameVec, title: record.name, date: date)

                if type.conforms(to: .image), let clip,
                   let cg = Self.downsampled(at: url, maxPixel: 320) {
                    let vec = try clip.embedImage(cg)
                    try store.upsert(kind: .file, refKey: record.id, space: clip.space,
                                     vector: vec, title: record.name, date: date)
                } else if type.conforms(to: .movie), let clip {
                    let frames = try await FrameSampler.sample(url: url, count: frameCount)
                    for (i, frame) in frames.enumerated() {
                        let vec = try clip.embedImage(frame)
                        try store.upsert(kind: .fileChunk, refKey: record.id, frameIndex: 1000 + i,
                                         space: clip.space, vector: vec, title: record.name, date: date)
                    }
                } else if type.conforms(to: .pdf) || url.pathExtension.lowercased() == "pdf" {
                    // 扫描件月结单等:PDF 无文字层,逐页 OCR(繁简)。原文入 ocr_text
                    // 供 文字: 精确搜;每页文本做 Gemma 嵌入供 语义: 搜
                    let pages = TextExtractor.pdfPageTexts(at: url)
                    if !pages.isEmpty {
                        let joined = pages.map { "【第\($0.page)页】\($0.text)" }.joined(separator: "\n")
                        try store.saveOCR(refKey: record.id, text: String(joined.prefix(40000)))
                    }
                    for (i, page) in pages.enumerated() where !page.text.isEmpty {
                        let vec = try gemma.embedDocument(page.text)
                        try store.upsert(kind: .fileChunk, refKey: record.id, frameIndex: 3000 + i,
                                         space: gemma.space, vector: vec,
                                         title: String(page.text.prefix(60)), date: date)
                    }
                } else if let text = TextExtractor.extractText(at: url) {
                    // 原文入 ocr_text(与扫描 PDF 同路径):内容可被 文字: 精确命中。
                    // 无文字也写 "" 占位:open() 自愈按「无 OCR 行」挑文件,占位防反复重处理
                    try store.saveOCR(refKey: record.id, text: String(text.prefix(120_000)))
                    for (i, chunk) in TextExtractor.chunks(of: text).enumerated() {
                        let vec = try gemma.embedDocument(chunk)
                        try store.upsert(kind: .fileChunk, refKey: record.id, frameIndex: i,
                                         space: gemma.space, vector: vec,
                                         title: String(chunk.prefix(60)), date: date)
                    }
                }
                try store.setFileIndexed(id: record.id)
            }.value
        } catch {
            errorCount += 1
        }
    }

    nonisolated private static func downsampled(at url: URL, maxPixel: CGFloat) -> CGImage? {
        let src = CGImageSourceCreateWithURL(url as CFURL, nil)
        guard let src else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: Int(maxPixel)
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }
}

// MARK: - 视频抽帧

enum FrameSampler {
    static func sample(asset: AVAsset, count: Int, maxPixel: CGFloat = 320) async throws -> [CGImage] {
        let duration = try await asset.load(.duration)
        guard duration.seconds > 0, count > 0 else { return [] }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: maxPixel, height: maxPixel)
        gen.requestedTimeToleranceBefore = CMTime(seconds: 1, preferredTimescale: 600)
        gen.requestedTimeToleranceAfter = CMTime(seconds: 1, preferredTimescale: 600)

        let step = duration.seconds / Double(max(count, 1))
        let times: [CMTime] = (0..<count).map { i in
            let sec = min(Double(i) * step + step / 2, max(duration.seconds - 0.1, 0))
            return CMTime(seconds: sec, preferredTimescale: 600)
        }

        var frames: [CGImage] = []
        for await result in gen.images(for: times) {
            if case .success(let frame) = result {
                frames.append(frame.image)
            }
        }
        return frames
    }

    static func sample(url: URL, count: Int) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        return try await sample(asset: asset, count: count)
    }
}

// MARK: - 文本抽取与分块

enum TextExtractor {
    static let textExtensions: Set<String> = [
        "txt", "md", "markdown", "csv", "tsv", "json", "xml", "yaml", "yml",
        "log", "rtf", "swift", "m", "h", "c", "cpp", "py", "js", "ts", "html", "css",
        "java", "kt", "go", "rs", "sql", "ini", "conf", "plist"
    ]

    static func extractText(at url: URL) -> String? {
        let ext = url.pathExtension.lowercased()
        if ext == "xlsx" || ext == "xlsm" {
            return xlsxText(at: url) ?? ""
        }
        // 大于 10MB 的纯文本不抽全文,避免病态文件(用户 3.5MB 交易 CSV 需覆盖)。
        // 支持的类型一律返回 String(空串而非 nil):调用方写 "" 占位 OCR 行,
        // open() 的自愈按「无 OCR 行」挑文件,占位防止每个同步反复重处理
        if textExtensions.contains(ext) {
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return "" }
            guard size < 10_000_000 else { return "" }
            guard let data = try? Data(contentsOf: url) else { return "" }
            if ext == "rtf" {
                guard let attr = try? NSAttributedString(
                    data: data,
                    options: [.documentType: NSAttributedString.DocumentType.rtf],
                    documentAttributes: nil) else { return "" }
                return attr.string
            }
            return String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1)
                ?? ""
        }
        if ext == "pdf" {
            guard let doc = PDFDocument(url: url) else { return nil }
            var all = ""
            for i in 0..<min(doc.pageCount, 200) {
                if let page = doc.page(at: i), let s = page.string {
                    all += s + "\n"
                }
            }
            return all.isEmpty ? nil : all
        }
        return nil
    }

    /// 扫描件 PDF 没有文字层:逐页渲染 2048px 位图后走 Vision OCR
    /// (繁体 zh-Hant/简体/英文)。返回 [(页码, 该页文本)],页数封顶防病态文件。
    static func pdfPageTexts(at url: URL, maxPages: Int = 24) -> [(page: Int, text: String)] {
        guard let doc = PDFDocument(url: url) else { return [] }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hant", "zh-Hans", "en-US"]
        request.usesLanguageCorrection = false
        var out: [(page: Int, text: String)] = []
        for i in 0..<min(doc.pageCount, maxPages) {
            guard let page = doc.page(at: i),
                  let cg = page.thumbnail(of: CGSize(width: 2048, height: 2048), for: .mediaBox).cgImage
            else { continue }
            let handler = VNImageRequestHandler(cgImage: cg, options: [:])
            guard (try? handler.perform([request])) != nil else { continue }
            let text = ((request.results as? [VNRecognizedTextObservation]) ?? [])
                .compactMap { $0.topCandidates(1).first?.string }
                .joined(separator: "\n")
            if !text.isEmpty { out.append((i + 1, text)) }
        }
        return out
    }

    /// xlsx/xlsm(Office Open XML)内容抽取:本质是 ZIP 容器。
    /// iOS 无公开 zip API,用自实现最小读取器解包,只取 sharedStrings 与
    /// 各 worksheet;单元格文本按行拼接(数字/日期序列号以原值入文,供编号检索)。
    static func xlsxText(at url: URL) -> String? {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size < 20_000_000,
              let data = try? Data(contentsOf: url),
              let entries = MiniZip.entries(of: data) else { return nil }
        var shared: [String] = []
        for entry in entries where entry.name == "xl/sharedStrings.xml" {
            guard let xml = MiniZip.entryData(entry, in: data) else { continue }
            let parser = XMLParser(data: xml)
            let delegate = XLSXTextParser(mode: .sharedStrings, shared: [])
            parser.delegate = delegate
            parser.parse()
            shared = delegate.shared
        }
        var rows: [String] = []
        for entry in entries where entry.name.hasPrefix("xl/worksheets/sheet") {
            guard let xml = MiniZip.entryData(entry, in: data) else { continue }
            let parser = XMLParser(data: xml)
            let delegate = XLSXTextParser(mode: .sheet, shared: shared)
            parser.delegate = delegate
            parser.parse()
            rows.append(contentsOf: delegate.rows)
        }
        return rows.isEmpty ? nil : rows.joined(separator: "\n")
    }

    /// 按段落聚合为 ~500 字块,相邻块重叠 ~80 字
    static func chunks(of text: String, maxChars: Int = 500, overlap: Int = 80) -> [String] {
        var chunks: [String] = []
        var current = ""
        for para in text.components(separatedBy: .newlines) {
            let p = para.trimmingCharacters(in: .whitespaces)
            if p.isEmpty { continue }
            if current.count + p.count + 1 > maxChars {
                if !current.isEmpty { chunks.append(current) }
                current = current.suffix(overlap) + " " + p
            } else {
                current += (current.isEmpty ? "" : " ") + p
            }
            if chunks.count >= 24 { break }   // 单文件块数上限
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}

// MARK: - 最小 ZIP 只读解析器(xlsx 解包用)

/// xlsx 即 ZIP 容器,iOS 又无公开 zip API——自实现只读路径:
/// 尾部找 EOCD → 中央目录逐条目 → 按本地头偏移取数据。
/// 只走中央目录(尺寸字段可靠,数据描述符场景天然兼容),zip64/加密不在范围。
enum MiniZip {
    struct Entry {
        let name: String
        let method: UInt16          // 0=stored 8=deflate
        let compressedSize: Int
        let uncompressedSize: Int
        let headerOffset: Int
    }

    static func entries(of data: Data) -> [Entry]? {
        guard data.count >= 22 else { return nil }
        var eocd = -1
        let floorBound = max(0, data.count - 65557)   // 注释区最长 65535 + 固定 22
        var i = data.count - 22
        while i >= floorBound {
            if data[i] == 0x50, data[i + 1] == 0x4B, data[i + 2] == 0x05, data[i + 3] == 0x06 {
                eocd = i
                break
            }
            i -= 1
        }
        guard eocd >= 0, eocd + 22 <= data.count else { return nil }
        let count = u16(data, eocd + 10)
        var p = u32(data, eocd + 16)
        var out: [Entry] = []
        for _ in 0..<count {
            guard p + 46 <= data.count, u32(data, p) == 0x02014B50 else { break }
            let nameLen = u16(data, p + 28), extraLen = u16(data, p + 30), commentLen = u16(data, p + 32)
            guard let nameData = sub(data, p + 46, nameLen),
                  let name = String(data: nameData, encoding: .utf8) else { break }
            out.append(Entry(name: name,
                             method: UInt16(u16(data, p + 10)),
                             compressedSize: u32(data, p + 20),
                             uncompressedSize: u32(data, p + 24),
                             headerOffset: u32(data, p + 42)))
            p += 46 + nameLen + extraLen + commentLen
        }
        return out.isEmpty ? nil : out
    }

    static func entryData(_ entry: Entry, in data: Data) -> Data? {
        let p = entry.headerOffset
        guard p + 30 <= data.count, u32(data, p) == 0x04034B50 else { return nil }
        let nameLen = u16(data, p + 26), extraLen = u16(data, p + 28)
        guard let raw = sub(data, p + 30 + nameLen + extraLen, entry.compressedSize) else { return nil }
        guard entry.method == 8 else { return entry.method == 0 ? raw : nil }
        return inflate(raw, size: entry.uncompressedSize)
    }

    private static func sub(_ data: Data, _ offset: Int, _ length: Int) -> Data? {
        guard length >= 0, offset >= 0, offset + length <= data.count else { return nil }
        return data.subdata(in: offset..<(offset + length))
    }

    private static func u16(_ d: Data, _ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
    private static func u32(_ d: Data, _ o: Int) -> Int { u16(d, o) | u16(d, o + 2) << 16 }

    /// zip 条目是 raw deflate;Compression 的 COMPRESSION_ZLIB 即 raw deflate,直接对接
    private static func inflate(_ src: Data, size: Int) -> Data? {
        guard size > 0 else { return Data() }
        var dst = Data(count: size)
        let n = dst.withUnsafeMutableBytes { dp -> Int in
            src.withUnsafeBytes { sp -> Int in
                guard let db = dp.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let sb = sp.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return -1 }
                return compression_decode_buffer(db, size, sb, src.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard n >= 0 else { return nil }
        dst.count = n
        return dst
    }
}

// MARK: - xlsx XML 文本抽取

/// sharedStrings.xml 收集全部 <t>(富文本多段自动拼接);
/// sheet 收集每行单元格文本。仅这两类 XML 有检索价值。
final class XLSXTextParser: NSObject, XMLParserDelegate {
    enum Mode { case sharedStrings, sheet }
    private let mode: Mode
    private(set) var shared: [String]
    private(set) var rows: [String] = []

    private var siBuffer = ""
    private var lineBuffer = ""
    private var cellType = ""
    private var cellValue = ""
    private var cellText = ""
    private var collecting: String?

    init(mode: Mode, shared: [String]) {
        self.mode = mode
        self.shared = shared
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
                qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]) {
        switch elementName {
        case "si": siBuffer = ""
        case "row": lineBuffer = ""
        case "c":
            cellType = attributeDict["t"] ?? ""
            cellValue = ""
            cellText = ""
        case "v", "t": collecting = elementName
        default: break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard let tag = collecting else { return }
        switch (mode, tag) {
        case (.sharedStrings, "t"): siBuffer += string
        case (.sheet, "v"): cellValue += string
        case (.sheet, "t"): cellText += string
        default: break
        }
    }

    func parser(_ parser: XMLParser, foundCDATA CDATA: Data) {
        guard let tag = collecting, let s = String(data: CDATA, encoding: .utf8) else { return }
        switch (mode, tag) {
        case (.sharedStrings, "t"): siBuffer += s
        case (.sheet, "v"): cellValue += s
        case (.sheet, "t"): cellText += s
        default: break
        }
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName qName: String?) {
        collecting = nil
        switch elementName {
        case "si":
            shared.append(siBuffer)
        case "c":
            let text: String
            switch cellType {
            case "s":
                let idx = Int(cellValue) ?? -1
                text = idx >= 0 && idx < shared.count ? shared[idx] : ""
            case "inlineStr": text = cellText
            case "b", "e": text = ""          // 布尔/错误值对检索是噪声
            default: text = cellValue         // 数字/日期序列号/公式结果按原值入文
            }
            if !text.isEmpty { lineBuffer += (lineBuffer.isEmpty ? "" : " ") + text }
        case "row":
            if !lineBuffer.isEmpty { rows.append(lineBuffer) }
        default: break
        }
    }
}
