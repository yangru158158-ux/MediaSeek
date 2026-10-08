# TestFlight 上手机全流程指南(Windows 无 Mac 版)

> 目标:让「智搜」最终安装到你的 iPhone 17 Pro Max。
> 你的角色:按顺序做浏览器操作;我已把所有自动化部分(代码、CI、工作流)准备好。
> 💳 标记 = 需要付款的检查点,全文只有一处。

---

## 阶段 A:GitHub 免费部分(不花钱,先把编译跑通)

### A1. 注册/登录 GitHub
- 浏览器打开 <https://github.com/signup>,用邮箱注册(已有账号直接登录 <https://github.com/login>)

### A2. 创建仓库
- 打开 <https://github.com/new>
- Repository name 填 `MediaSeek`
- **可见性必须选 `Public`(公开)** ← 这一步决定成本:公开仓库的 macOS 构建机免费;私有仓库 macOS 分钟按 10 倍扣,很快烧完免费额度
- 其余保持默认,点 **Create repository**(不要勾选自动生成 README,本地已有全部文件)

### A3. 把代码推上去
在 Windows 上执行(Git Bash,把 `你的用户名` 换掉):

```bash
cd ~/.zcode/workspace/default/MediaSeek
git remote add origin https://github.com/你的用户名/MediaSeek.git
git push -u origin main
```

推的时候会弹 GitHub 登录授权(浏览器或 Token),照提示操作即可。

### A4. 验证自动编译
- 打开 <https://github.com/你的用户名/MediaSeek/actions>
- 应该看到 **Build Check** 工作流自动开始跑(绿色机器图标 = macOS)
- 约 5-10 分钟出结果,全绿 ✅ = 代码能在真机 iOS 上编译通过
- 此时**还没有装到手机的路径,继续阶段 B**

---

## 阶段 B:注册 Apple 开发者账号

### 💳 B1. 【全文唯一付费点:$99/年 ≈ ¥688】
- 用你 iPhone 上已有的 Apple ID,浏览器打开 <https://developer.apple.com/programs/enroll/>
- 选 **Individual(个人)** → 用 Apple ID 登录 → 填写真实姓名、地址、手机验证
- 付款:支持支付宝/银联(中国区)或信用卡,按页面提示支付
- ⚠️ 注册填的姓名会成为开发者署名,填真实中文姓名
- 等审核开通:一般几分钟到 48 小时,开通后 <https://appstoreconnect.apple.com> 能正常登录即为成功

### B2. 拿到 Team ID
- 登录 <https://developer.apple.com/account> → Membership details → 记下 **Team ID**(10 位字母数字,如 `A1B2C3D4E5`)

---

## 阶段 C:创建 App 与 API 密钥(浏览器,不花钱)

### C1. 注册 App ID
- 打开 <https://developer.apple.com/account/resources/identifiers/list/bundleId> → 左上 **+**
- 选 **App IDs → App** → 勾选能力**什么都不用加**,直接 Continue
- Description 填 `MediaSeek`,Bundle ID 选 **Explicit**,输入 `com.medseek.MediaSeek`
- → Register

### C2. 在 App Store Connect 创建 App 记录
- 打开 <https://appstoreconnect.apple.com/apps> → 左上 **+** → **新建 App**
- 名称:`智搜`(被占用就换,名字只影响商店显示)、主要语言:简体中文
- Bundle ID:选刚注册的 `com.medseek.MediaSeek`
- SKU 随便填:`medseek001` → 创建

### C3. 生成 App Store Connect API 密钥(CI 上传用)
- 打开 <https://appstoreconnect.apple.com/access/integrations/api> → **+ 生成 API 密钥**
- 名称:`github-ci`,权限选 **App Manager** → 生成
- 页面会显示三样东西,**全部记录下来**:
  1. **Issuer ID**(页面顶部,一长串 UUID)
  2. **Key ID**(密钥那行的 10 位 ID)
  3. **下载 .p8 私钥文件** ← ⚠️ **只能下载一次**,立即保存到电脑安全位置,关掉就再也拿不到了

### C4. 把密钥配置到 GitHub(浏览器)
- 打开 <https://github.com/你的用户名/MediaSeek/settings/secrets/actions> → **New repository secret**,逐条添加 4 个:

| Name | Secret 值 |
|---|---|
| `ASC_KEY_ID` | C3 记下的 Key ID |
| `ASC_ISSUER_ID` | C3 记下的 Issuer ID |
| `ASC_API_KEY_P8` | 打开命令行执行 `base64 -w0 你的下载路径/AuthKey_XXXX.p8`,粘贴输出的全部内容 |
| `DEVELOPMENT_TEAM_ID` | B2 的 Team ID |

> `.p8` 内容做 base64 是为了避开换行符问题,Windows 的 Git Bash 自带 `base64` 命令。

---

## 阶段 D:发布并安装到 iPhone(见证时刻)

### D1. 触发发布
- 打开 <https://github.com/你的用户名/MediaSeek/actions> → 左侧选 **Release to TestFlight** → 右侧 **Run workflow** → 确认运行
- 约 10-20 分钟:它会自动归档 App、签名(首次会自动创建签名证书和描述文件)、上传到苹果

### D2. 添加自己为测试员
- 等 5-15 分钟让苹果处理构建:打开 <https://appstoreconnect.apple.com/testflight>
- 点进 App → **内部测试** → **App Store Connect 用户** → **+** → 勾选你自己(注册开发者用的 Apple ID)→ 邀请

### D3. iPhone 安装 🎉
1. iPhone 上 App Store 搜索安装 **TestFlight**(苹果官方,免费)
2. Apple ID 登录 TestFlight → 你会看到「智搜」→ 点 **安装**
3. 首次启动授权相册 → 到「设置」页确认模型状态;模型包还没导的话先按 README 跑 `scripts/export_models.py`,把 `models.zip` 通过隔空投送不行的话用 iCloud/微信传到手机,存到「文件」里,在 App 设置里导入
4. 开始用:「一只猫」搜起

### D4. 以后的更新
改完代码 → `git push` → 手动跑一次 **Release to TestFlight** → 手机上 TestFlight 点更新。构建 90 天过期,重跑一次工作流即可。

---

## 常见问题

- **Build Check 挂了?** 点进失败步骤看红色日志发给我;常见是 swift-transformers 版本 API 变动,我远程调。
- **Release 报签名错误?** 确认 4 个 secret 拼写、Team ID 无多余空格;首次运行苹果偶尔要几分钟激活自动签名,重跑一次。
- **上传被拒"Invalid Bundle"?** Bundle ID 拼写要与 C1 注册完全一致(当前 `com.medseek.MediaSeek`)。
- **模型包太大传不进手机?** 模型 zip 也可以直接放进 GitHub 仓库的 Release 附件,手机 Safari 下载后存「文件」,App 内导入。
- **费用总览:** 付费只有 B1 一次 $99/年;GitHub 公开仓库构建 $0;TestFlight $0。
