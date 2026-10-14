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
    func image(for asset: PHAsset, maxPixel: CGFloat) async throws -> CGImage? {
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        return try await withCheckedThrowingContinuation { cont in
            var finished = false
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: CGSize(width: maxPixel, height: maxPixel),
                // 分析取图必须 aspectFit(完整画面):aspectFill 会把照片裁成中央
                // 正方形——横构图左右各裁 12.5%、竖构图上下各裁,站在画面边缘的
                // 人被裁掉,人脸索引/合影补认/OCR 全都看不见她(166 根因之一;
                // 缩略图显示走 Components 自己的路径,不受影响)
                contentMode: .aspectFit,
                options: options) { image, info in
                guard !finished else { return }
                // 降质中间帧不结束请求,继续等最终回调。判定必须放在取图之前:
                // 曾因先判 image 非空,降质帧提前 resume,最终帧再 resume
                // 触发 continuation 重复恢复 → EXC_BREAKPOINT 主线程崩溃
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
                finished = true
                if let cg = image?.cgImage {
                    cont.resume(returning: cg)
                } else if let err = info?[PHImageErrorKey] as? Error {
                    cont.resume(throwing: err)
                } else {
                    cont.resume(returning: nil)
                }
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
