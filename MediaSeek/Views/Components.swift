import Foundation
import SwiftUI
import Photos
import AVKit
import QuickLook

/// 缩略图加载(相册资产):PHCachingImageManager + 内存缓存
final class PhotoThumbLoader {
    static let shared = PhotoThumbLoader()
    private let manager = PHCachingImageManager()
    private let cache = NSCache<NSString, UIImage>()

    func thumb(for asset: PHAsset, pixel: CGFloat) async -> UIImage? {
        let key = "\(asset.localIdentifier)-\(Int(pixel))" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        let image: UIImage? = await withCheckedContinuation { cont in
            manager.requestImage(for: asset,
                                 targetSize: CGSize(width: pixel, height: pixel),
                                 contentMode: .aspectFill,
                                 options: options) { img, info in
                if info?[PHImageResultIsDegradedKey] as? Bool == true {
                    // 等最终结果
                } else {
                    cont.resume(returning: img)
                }
            }
        }
        if let image { cache.setObject(image, forKey: key) }
        return image
    }
}

/// 相册缩略图
struct AssetThumbView: View {
    let asset: PHAsset
    var pixel: CGFloat = 220

    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(Color(.systemGray5))
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            }
        }
        .clipped()
        .task(id: asset.localIdentifier) {
            image = await PhotoThumbLoader.shared.thumb(for: asset, pixel: pixel)
        }
    }
}

/// 导入文件缩略图
struct FileThumbView: View {
    let record: ImportedFile
    @EnvironmentObject var app: AppModel
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(Color(.systemGray5))
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "doc").font(.largeTitle).foregroundStyle(.secondary)
            }
        }
        .clipped()
        .task(id: record.id) {
            image = await app.imports.thumbnail(for: record)
        }
    }
}

/// 搜索结果格
struct HitCell: View {
    let hit: DisplayHit
    @EnvironmentObject var app: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            thumbnail
                .frame(height: 110)
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(alignment: .topLeading) { badge }
            Text(hit.title)
                .font(.caption2)
                .lineLimit(1)
            Text("\(Int(hit.score * 100))%")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var thumbnail: some View {
        switch hit.target {
        case .asset(let asset):
            AssetThumbView(asset: asset)
                .overlay { hit.kind == .videoFrame ? playIcon : nil }
        case .file(let record):
            FileThumbView(record: record)
        }
    }

    private var playIcon: some View {
        ZStack {
            Circle().fill(.black.opacity(0.45)).frame(width: 30, height: 30)
            Image(systemName: "play.fill").foregroundStyle(.white).font(.caption)
        }
    }

    private var badge: some View {
        Text(badgeText)
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(.black.opacity(0.55), in: Capsule())
            .foregroundStyle(.white)
            .padding(4)
    }

    private var badgeText: String {
        switch hit.kind {
        case .photo, .photoLabel: return "照片"
        case .videoFrame: return "视频"
        case .file, .fileChunk: return "文件"
        }
    }
}

/// 详情页:照片查看 / 视频播放 / 文件 QuickLook
struct DetailSheet: View {
    let hit: DisplayHit
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            content
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("完成") { dismiss() }
                    }
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch hit.target {
        case .asset(let asset):
            if asset.mediaType == .video {
                VideoPlayerScreen(asset: asset)
            } else {
                FullImageViewer(asset: asset)
            }
        case .file(let record):
            FilePreviewScreen(record: record)
        }
    }
}

struct FullImageViewer: View {
    let asset: PHAsset
    @State private var image: UIImage?

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if let image {
                    Image(uiImage: image).resizable().scaledToFit()
                } else {
                    ProgressView().tint(.white)
                }
            }
            .task {
                let side = max(geo.size.width, geo.size.height) * UIScreen.main.scale
                image = await PhotoThumbLoader.shared.thumb(for: asset, pixel: side)
            }
        }
    }
}

struct VideoPlayerScreen: View {
    let asset: PHAsset
    @EnvironmentObject var app: AppModel
    @State private var player: AVPlayer?

    var body: some View {
        Group {
            if let player {
                VideoPlayer(player: player)
            } else {
                ZStack { Color.black; ProgressView().tint(.white) }
            }
        }
        .task {
            if player == nil, let item = await app.photoLib.playerItem(for: asset) {
                player = AVPlayer(playerItem: item)
                player?.play()
            }
        }
        .onDisappear { player?.pause() }
    }
}

struct FilePreviewScreen: View {
    let record: ImportedFile
    @EnvironmentObject var app: AppModel
    @State private var previewURL: URL?

    var body: some View {
        VStack(spacing: 8) {
            if let url = previewURL {
                Text(url.lastPathComponent)
                    .font(.footnote).foregroundStyle(.secondary)
                Text("点击下方「快速查看」预览文件")
                    .font(.caption).foregroundStyle(.tertiary)
            } else {
                ProgressView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .quickLookPreview($previewURL)
        .task {
            if previewURL == nil, let url = try? app.imports.resolveURL(record) {
                previewURL = url
            }
        }
    }
}
