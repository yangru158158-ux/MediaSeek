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

    /// 指定尺寸取图(会自动处理 iCloud 下载)。maxPixel 建议 320(索引)或屏幕尺寸(查看)
    func image(for asset: PHAsset, maxPixel: CGFloat) async throws -> CGImage? {
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        let side = max(asset.pixelWidth, asset.pixelHeight) > 0
            ? maxPixel * 2   // aspectFill 取方形裁剪,放大一点保清晰
            : maxPixel
        return try await withCheckedThrowingContinuation { cont in
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: CGSize(width: side, height: side),
                contentMode: .aspectFill,
                options: options) { image, info in
                if let cg = image?.cgImage {
                    cont.resume(returning: cg)
                } else if let err = info?[PHImageErrorKey] as? Error {
                    cont.resume(throwing: err)
                } else if info?[PHImageResultIsDegradedKey] as? Bool == true {
                    // 忽略降质中间帧,等待最终回调
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
