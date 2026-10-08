# 免费侧载路线:Sideloadly 装上 iPhone(无需付费开发者账号)

> 原理:GitHub Actions 产出**未签名 IPA** → Windows 上用 Sideloadly 以你的**免费 Apple ID** 重新签名 → 数据线装进 iPhone。
> 限制:签名 **7 天有效**到期需重签;免费 Apple ID 同时最多 3 个侧载 App。功能与 TestFlight 版完全一致。

## 第 0 步:下载 IPA(电脑浏览器)

1. 打开 <https://github.com/yangru158158-ux/MediaSeek/actions> → 左侧 **Build Unsigned IPA** → 点最新一次绿色运行
2. 页面底部 **Artifacts** 区 → 下载 `MediaSeek-unsigned-ipa`
3. 解压 zip,得到 `MediaSeek-unsigned.ipa`,记住存放位置

## 第 1 步:装苹果驱动(让 Windows 认出 iPhone)

- Microsoft Store 搜索安装 **「Apple Devices」**(苹果官方);或到 <https://www.apple.com.cn/itunes/> 下载安装版 iTunes
- 用数据线连接 iPhone → 手机上弹窗点 **「信任」** 并输锁屏密码
- 验证:资源管理器里能看到 iPhone 相册即算成功

## 第 2 步:装 Sideloadly

- 打开 <https://sideloadly.io> → 下载 Windows 版 → 安装

## 第 3 步:签名并安装

1. 打开 Sideloadly,iPhone 保持连接
2. 把 `MediaSeek-unsigned.ipa` 拖进 Sideloadly 窗口
3. **Apple ID 邮箱**填:`44957799@qq.com`
4. **密码在 Sideloadly 窗口里自己输,不要发给任何人(包括我)**
   - 若提示需要「App 专用密码」:浏览器开 <https://account.apple.com> → 登录与安全 → **App 专用密码** → 生成一个,把它填进 Sideloadly 的密码框
5. 点 **Start**,等进度条走完 → iPhone 上出现「智搜」

## 第 4 步:首次打开的信任设置

- iPhone 上如果提示"不受信任的开发者":**设置 → 通用 → VPN与设备管理** → 点你的 Apple ID 那一项 → **信任**
- 之后正常打开「智搜」

## 第 5 步:导入模型(第一次需要,之后永久有效)

1. 电脑上:`pip install torch "transformers>=4.55" "coremltools>=8.0" sentencepiece`,然后
   `python scripts/export_models.py --zip models.zip`
2. 把 `models.zip` 传到 iPhone 的「文件」App(微信文件助手 / iCloud 云盘 / 百度网盘均可)
3. 打开「智搜」→ 设置 → **从「文件」导入模型包** → 选中 models.zip,等待编译加载
4. 回资料库页 → 同步照片 → 搜索"一只猫"

## 续命:第 7 天之前

- 插线打开 Sideloadly → 重拖同一个 IPA → Start,即可再续 7 天
- 嫌麻烦可装 **AltServer**(AltStore 官网)保持后台,同一 Wi-Fi 下自动续签
- 满意后随时切换 ¥688/年 TestFlight 路线:从此免维护(告诉我一声即可)

## 常见问题

- **Sideloadly 报 "Apple Drivers not found"**:装上面第 1 步的驱动后重启 Sideloadly
- **报 "Provisioning Profile" 错误**:免费证书生成偶尔抽风,重试一次;或换网络
- **手机上 App 图标变灰/打不开**:签名过期了,重签即可,数据不丢(索引库在 App 沙盒里,重签不影响)
