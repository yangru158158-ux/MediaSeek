import Foundation
import Photos
import UIKit
import AVFoundation

/// 系统相册访问:全量枚举照片/视频 + 高效取图
final class PhotoLibraryService: ObservableObject {
    @Published var authStatus: PHAuthorizationStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)

    var isAuthorized: Bool { authStatus == .authorized }
    var needsRequest: Bool { authStatus == .notDetermined }

    func requestAccess() async {
        authStatus = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }

    private func fetchOptions() -> PHFetchOptions {
        let o = PHFetchOptions()
        o.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        return o
    }

    func fetchAllAssets() -> PHFetchResult<PHAsset> {
        PHAsset.fetchAssets(with: fetchOptions())
    }

    /// 相册视频总数(视频索引覆盖度展示用)
    func totalVideoCount() -> Int {
        PHAsset.fetchAssets(with: .video, options: nil).count
    }

    /// 指定尺寸取图(会自动处理 iCloud 下载)。maxPixel 建议 320(索引)或屏幕尺寸(查看)
    /// 批83:桥接盒——锁保 continuation 单次恢复;取消先于注册时记标志,
    /// 注册瞬间立即让步(后台化瞬间回调可能被系统丢弃或 iCloud 网络卡死
    /// 永不回来,裸 continuation 会把子任务悬死 → 任务组 next() 永等 →
    /// run() 卡死、isRunning 永不归零,UI 挂「正在索引 n/n」——与 232
    /// 词表悬死同族的另一条路径)。
    final class PHBridgeBox: @unchecked Sendable {
        let lock = NSLock()
        var cont: CheckedContinuation<CGImage?, Error>?
        var requestID: PHImageRequestID = PHInvalidImageRequestID
        var cancelled = false
    }

    func image(for asset: PHAsset, maxPixel: CGFloat) async throws -> CGImage? {
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        let box = PHBridgeBox()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                box.lock.lock()
                if box.cancelled {
                    box.lock.unlock()
                    cont.resume(throwing: CancellationError())
                    return
                }
                box.cont = cont
                box.lock.unlock()
                let reqID = PHImageManager.default().requestImage(
                    for: asset,
                    targetSize: CGSize(width: maxPixel, height: maxPixel),
                    // 分析取图必须 aspectFit(完整画面):aspectFill 会把照片裁成中央
                    // 正方形——横构图左右各裁 12.5%、竖构图上下各裁,站在画面边缘的
                    // 人被裁掉,人脸索引/合影补认/OCR 全都看不见她(166 根因之一;
                    // 缩略图显示走 Components 自己的路径,不受影响)
                    contentMode: .aspectFit,
                    options: options) { image, info in
                    // 降质中间帧不结束请求,继续等最终回调。判定必须放在取图之前:
                    // 曾因先判 image 非空,降质帧提前 resume,最终帧再 resume
                    // 触发 continuation 重复恢复 → EXC_BREAKPOINT 主线程崩溃
                    if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
                    box.lock.lock()
                    guard let c = box.cont else { box.lock.unlock(); return }
                    box.cont = nil
                    box.lock.unlock()
                    if let cg = image?.cgImage {
                        c.resume(returning: cg)
                    } else if let err = info?[PHImageErrorKey] as? Error {
                        c.resume(throwing: err)
                    } else {
                        c.resume(returning: nil)
                    }
                }
                // ID 回填过锁:onCancel 可能已并发读到失效占位(此时 cont 已被
                // onCancel 恢复,请求回调会空转,撤销漏发只浪费一个惰性请求)
                box.lock.lock()
                box.requestID = reqID
                box.lock.unlock()
            }
        } onCancel: {
            // 取消:撤销请求 + 抛中止(上层 catch 判 Task.isCancelled 按用户
            // 停止记账,不进失败黑名单/错误数);cont 为 nil 说明尚未注册,
            // 记 cancelled 标志由注册路径立即让步
            box.lock.lock()
            box.cancelled = true
            let c = box.cont
            box.cont = nil
            let id = box.requestID
            box.lock.unlock()
            c?.resume(throwing: CancellationError())
            if id != PHInvalidImageRequestID {
                PHImageManager.default().cancelImageRequest(id)
            }
        }
    }

    func playerItem(for asset: PHAsset) async -> AVPlayerItem? {
        let options = PHVideoRequestOptions()
        options.version = .current
        options.isNetworkAccessAllowed = true
        return await withCheckedContinuation { cont in
            PHImageManager.default().requestPlayerItem(
                forVideo: asset, options: options, resultHandler: { item, _ in
                    cont.resume(returning: item)
                })
        }
    }

    /// 视频资源对应的 AVAsset(用于抽帧)
    func avAsset(for asset: PHAsset) async -> AVAsset? {
        let options = PHVideoRequestOptions()
        options.version = .current
        options.isNetworkAccessAllowed = true
        return await withCheckedContinuation { cont in
            PHImageManager.default().requestAVAsset(
                forVideo: asset, options: options) { avAsset, _, _ in
                cont.resume(returning: avAsset)
            }
        }
    }
}
