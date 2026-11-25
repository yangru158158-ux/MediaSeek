# 查询 App Store Connect 构建处理状态(团队密钥 8H66F4QJCN,App 管理权限)
# 用法: python scripts/asc_builds.py   → 打印 com.medseek.MediaSeek 最近 6 个构建的
#       构建号/所属版本/处理状态/上传时间/测试组
# 批140 修(2026-11-25):接口域名原写 api.appstoreconnect.com(少了 .apple,该域名不存在,
#   脚本自创建起从未查通——此前两次被误判为「沙箱网络限制」「本机网络访问不了」);
#   正确域名 api.appstoreconnect.apple.com(与 asc_add_tester.py / asc_external_setup.py 一致)。
#   同时补出「所属版本」与「测试组」:发版后核对一个构建有没有自成新版本、有没有进
#   内测组,直接看这里,不必登录浏览器。
#   注意:App Store Connect API 用 fields[...] 限定字段后,关系(preReleaseVersion、
#   betaGroups)必须在 fields[builds] 里显式列出,否则 relationships 不返回
import time, json, urllib.request, urllib.error, collections
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import base64

API = "https://api.appstoreconnect.apple.com"
KEY_PATH = Path(r"C:\Users\44957\Downloads\MediaSeek装机\AuthKey_8H66F4QJCN.p8")
KEY_ID = "8H66F4QJCN"
ISSUER = "fb64c8fb-5335-4bf6-933b-a8b1a7837103"
BUNDLE = "com.medseek.MediaSeek"

def b64(d: bytes) -> str:
    return base64.urlsafe_b64encode(d).rstrip(b"=").decode()

header = b64(json.dumps({"alg": "ES256", "kid": KEY_ID}).encode())
payload = b64(json.dumps({
    "iss": ISSUER, "iat": int(time.time()), "exp": int(time.time()) + 1200,
    "aud": "appstoreconnect-v1"}).encode())
signing_input = f"{header}.{payload}".encode()

key = load_pem_private_key(KEY_PATH.read_bytes(), password=None)
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives import hashes
der = key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
# cryptography 的 ECDSA sign 返回 DER,JWT 要 raw r||s
r, s = decode_dss_signature(der)
jwt = f"{header}.{payload}.{b64(r.to_bytes(32,'big') + s.to_bytes(32,'big'))}"

def get(path):
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {jwt}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

try:
    apps = get(f"/v1/apps?filter[bundleId]={BUNDLE}")
except urllib.error.HTTPError as e:
    print("App Store Connect 接口返回", e.code, e.read().decode("utf-8", "replace")[:400]); raise SystemExit(1)
if not apps["data"]:
    print("未找到 App", BUNDLE); raise SystemExit(1)
app_id = apps["data"][0]["id"]

res = get(
    f"/v1/builds?filter[app]={app_id}&sort=-uploadedDate&limit=6"
    "&fields[builds]=version,uploadedDate,processingState,expired,preReleaseVersion,betaGroups"
    "&include=preReleaseVersion,betaGroups"
    "&fields[preReleaseVersions]=version&fields[betaGroups]=name,isInternalGroup")
by = collections.defaultdict(dict)
for i in res.get("included", []):
    by[i["type"]][i["id"]] = i.get("attributes", {})

# processingState: PROCESSING=处理中 / FAILED=处理失败 / VALID=可用
print(f"查询时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC | 最近 {len(res['data'])} 个构建")
print("构建号 | 所属版本 | 处理状态 | 已过期 | 上传时间(接口原值)      | 测试组")
for b in res["data"]:
    a = b["attributes"]
    rel = b.get("relationships", {})
    pv = by["preReleaseVersions"].get((rel.get("preReleaseVersion", {}).get("data") or {}).get("id"), {}).get("version", "?")
    groups = [by["betaGroups"].get(g.get("id"), {}).get("name", "?")
              for g in (rel.get("betaGroups", {}).get("data") or [])]
    print(f"  {a['version']:>4} | {pv:>8} | {a['processingState']:>9} | {str(a.get('expired')):>5} "
          f"| {a['uploadedDate'][:19].replace('T', ' ')} | {', '.join(groups) or '(无)'}")
