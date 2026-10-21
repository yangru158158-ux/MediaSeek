import Foundation
import AVFoundation
import Speech

/// 视频语音转文字:AVAsset 抽音轨(WAV PCM 16kHz)→ SFSpeechRecognizer 识别。
/// 第三十八批:m4a 导出在真机上可能产出坏文件(StackOverflow 实锤,模拟器
/// 正常真机无声),换成 AVAssetReader+Writer 直写 WAV PCM——Speech 框架
/// 最可靠的输入格式。识别文本追加进 ocr_text(〔语音〕段落),被 文字: 检索。
enum SpeechTranscriber {
    static let noSpeechMarker = "(未识别到语音)"
    static let maxAudioSeconds: Double = 1800

    static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status == .authorized)
            }
        }
    }

    static func diagnosticInfo() -> String {
        guard let rec = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")) else {
            return "zh-CN 识别器创建失败(系统不支持该语言)"
        }
        let onDevice = rec.supportsOnDeviceRecognition
        let available = rec.isAvailable
        return "zh-CN 识别器:可用=\(available), 本机模型=\(onDevice ? "已支持" : "未下载")"
    }

    static func onDeviceSupported() -> Bool {
        guard let rec = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")) else { return false }
        return rec.supportsOnDeviceRecognition
    }

    /// 转写一部视频的音轨:抽 WAV → SFSpeechRecognizer
    static func transcribeVideo(avAsset: AVAsset) async throws -> String {
        let duration = try await avAsset.load(.duration)
        guard duration.seconds > 0.5 else { return "" }

        // WAV PCM 16kHz 单声道(第三十八批):m4a 在真机上可能坏,PCM 直写
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("speech-\(UUID().uuidString).wav")
        let maxSec = min(duration.seconds, maxAudioSeconds)
        try await exportWAV(from: avAsset, to: tmp, maxSeconds: maxSec)
        defer { try? FileManager.default.removeItem(at: tmp) }

        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")),
              recognizer.isAvailable else {
            throw MSError("语音识别服务不可用")
        }
        let request = SFSpeechURLRecognitionRequest(url: tmp)
        request.shouldReportPartialResults = true
        let allowServer = UserDefaults.standard.object(forKey: "speechAllowServer") as? Bool ?? true
        request.requiresOnDeviceRecognition = !allowServer

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

    /// AVAssetReader+Writer 直写 WAV PCM 16kHz 单声道。
    /// 第三十八批:用后台同步循环,不用 requestMediaDataWhenReady 回调——
    /// 回调模式下 reader 出错/writer 卡死会导致 continuation 永不 resume,
    /// TaskGroup 排队等它 → 连续多个泄漏后崩溃(用户实测最后几个视频崩)
    private static func exportWAV(from avAsset: AVAsset, to url: URL, maxSeconds: Double) async throws {
        guard let audioTrack = try await avAsset.loadTracks(withMediaType: .audio).first else {
            throw MSError("无音轨")
        }
        let pcmSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let reader = try AVAssetReader(asset: avAsset)
        let readerOut = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: pcmSettings)
        reader.add(readerOut)
        reader.timeRange = CMTimeRange(
            start: .zero,
            duration: CMTime(seconds: maxSeconds, preferredTimescale: 600))

        let writer = try AVAssetWriter(outputURL: url, fileType: .wav)
        let writerIn = AVAssetWriterInput(mediaType: .audio, outputSettings: pcmSettings)
        writer.add(writerIn)

        guard reader.startReading() else {
            throw reader.error ?? MSError("音轨读取启动失败")
        }
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // 后台线程逐块搬运:同步循环,每块一定被处理,不依赖回调再触发
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .utility).async {
                var wroteAny = false
                while writerIn.isReadyForMoreMediaData {
                    if let buf = readerOut.copyNextSampleBuffer() {
                        writerIn.append(buf)
                        wroteAny = true
                    } else {
                        writerIn.markAsFinished()
                        // 等 writer 收尾
                        writer.finishWriting {
                            if writer.status == .completed {
                                cont.resume()
                            } else {
                                cont.resume(throwing: writer.error ?? MSError("WAV 写入失败(状态=\(writer.status.rawValue))"))
                            }
                        }
                        return
                    }
                }
                // isReadyForMoreMediaData 变 false 且 reader 已读完但没进 else 分支
                // (writer 内部缓冲满):等 writer 消化后重试,60 秒超时兜底
                if reader.status == .completed {
                    writerIn.markAsFinished()
                    writer.finishWriting {
                        if writer.status == .completed {
                            cont.resume()
                        } else {
                            cont.resume(throwing: writer.error ?? MSError("WAV 写入失败(尾部)"))
                        }
                    }
                } else if reader.status == .failed {
                    cont.resume(throwing: reader.error ?? MSError("音轨读取失败"))
                } else {
                    // writer buffer 满且 reader 还在读:极罕见,等 1 秒后强制收尾
                    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 1.0) {
                        writerIn.markAsFinished()
                        writer.finishWriting {
                            if writer.status == .completed {
                                cont.resume()
                            } else {
                                cont.resume(throwing: writer.error ?? MSError("WAV 写入超时"))
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 识别回调 → continuation 的单次恢复闸门
private final class ResumeGate: @unchecked Sendable {
    private let lock = NSLock()
    private var resumed = false
    private var task: SFSpeechRecognitionTask?
    private var text = ""
    private let cont: CheckedContinuation<String, Error>

    init(_ cont: CheckedContinuation<String, Error>) { self.cont = cont }

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
