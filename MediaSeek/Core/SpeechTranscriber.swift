import Foundation
import AVFoundation
import Speech

/// 视频语音转文字:AVAsset 抽音轨(m4a)→ SFSpeechRecognizer 端侧识别。
/// 强制 on-device:音频不出设备;识别文本追加进 ocr_text(〔语音〕段落),
/// 被 文字: 精确检索命中。耗时≈音频时长,故置于设置开关之后、同步收尾阶段,
/// 可中断可续(与 OCR/人脸补全同范式)。
enum SpeechTranscriber {

    /// 单视频音频时长上限:超长视频只转写前 30 分钟(导出时 timeRange 截断)
    static let maxAudioSeconds: Double = 1800

    /// 语音识别权限(Info.plist 必须声明 NSSpeechRecognitionUsageDescription)
    static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status == .authorized)
            }
        }
    }

    /// 本机是否支持中文端侧识别。不支持时整个语音阶段跳过(全局记账),
    /// 不逐视频试错——每部试一遍只会得到同一张错误罚单
    static func onDeviceSupported() -> Bool {
        guard let rec = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")) else { return false }
        return rec.supportsOnDeviceRecognition   // 属性,不是方法
    }

    /// 转写一部视频的音轨。无有效语音返回空串(调用方照常记账防重复处理)。
    /// 取消策略:进行中的识别自然收尾(用户视频多为短片,代价一段),
    /// 下一部视频开跑前 checkCancellation 生效——不把外部取消接进识别器,
    /// 换取 continuation 恢复路径只有「完成/出错」两条,无泄漏窗口
    static func transcribeVideo(avAsset: AVAsset) async throws -> String {
        let duration = try await avAsset.load(.duration)
        guard duration.seconds > 0.5 else { return "" }

        // 抽音轨 → 临时 m4a:SFSpeechURLRecognitionRequest 对视频容器不保证,
        // 走标准音频文件路径
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("speech-\(UUID().uuidString).m4a")
        guard let export = AVAssetExportSession(asset: avAsset, presetName: AVAssetExportPresetAppleM4A) else {
            throw MSError("无法建立音频导出会话")
        }
        export.outputURL = tmp
        export.outputFileType = .m4a
        if duration.seconds > maxAudioSeconds {
            export.timeRange = CMTimeRange(
                start: .zero,
                duration: CMTime(seconds: maxAudioSeconds, preferredTimescale: 600))
        }
        try await exportAudio(export)
        defer { try? FileManager.default.removeItem(at: tmp) }

        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")),
              recognizer.isAvailable else {
            throw MSError("语音识别服务不可用")
        }
        let request = SFSpeechURLRecognitionRequest(url: tmp)
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true   // 强制本机:音频不上传

        return try await withCheckedThrowingContinuation { cont in
            let gate = ResumeGate(cont)
            let task = recognizer.recognitionTask(with: request) { result, error in
                if let result {
                    gate.absorb(result.bestTranscription.formattedString)
                    if result.isFinal { gate.finish(error: error); return }
                }
                if let error { gate.finish(error: error); return }
            }
            gate.attach(task)
        }
    }

    /// exportAsynchronously 包成 async(部署目标 iOS 17,无 async export())
    private static func exportAudio(_ export: AVAssetExportSession) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            export.exportAsynchronously {
                switch export.status {
                case .completed: cont.resume()
                case .cancelled: cont.resume(throwing: CancellationError())
                default:
                    cont.resume(throwing: export.error ?? MSError("音轨导出失败"))
                }
            }
        }
    }
}

/// 识别回调 → continuation 的单次恢复闸门:final/error 两条路径都可能到,
/// 只恢复一次,重复回调(含 isFinal 后再来的残尾)静默丢弃
private final class ResumeGate: @unchecked Sendable {
    private let lock = NSLock()
    private var resumed = false
    private var task: SFSpeechRecognitionTask?
    private var text = ""
    private let cont: CheckedContinuation<String, Error>

    init(_ cont: CheckedContinuation<String, Error>) { self.cont = cont }

    /// 持有识别任务引用直到闸门析构(continuation 恢复后),
    /// 防任务对象被提前释放(样例代码均以属性持有,照做)
    func attach(_ t: SFSpeechRecognitionTask) {
        lock.lock(); task = t; lock.unlock()
    }

    func absorb(_ s: String) { lock.lock(); text = s; lock.unlock() }

    func finish(error: Error?) {
        lock.lock()
        if resumed { lock.unlock(); return }
        resumed = true
        let value = text
        lock.unlock()
        if let error { cont.resume(throwing: error) } else { cont.resume(returning: value) }
    }
}
