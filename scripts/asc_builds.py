# 查询 App Store Connect 构建处理状态(团队密钥 8H66F4QJCN,App 管理权限)
# 用法: python asc_builds.py   → 打印 com.medseek.MediaSeek 最近 5 个构建的版本/上传时间/处理状态
import time, json, urllib.request
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import base64

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

def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {jwt}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

apps = get(f"https://api.appstoreconnect.com/v1/apps?filter[bundleId]={BUNDLE}")
if not apps["data"]:
    print("未找到 App", BUNDLE); raise SystemExit(1)
app_id = apps["data"][0]["id"]

builds = get(
    "https://api.appstoreconnect.com/v1/builds"
    f"?filter[app]={app_id}&sort=-uploadedDate&limit=5"
    "&fields[builds]=version,uploadedDate,processingState")
for b in builds["data"]:
    a = b["attributes"]
    print(f"build {a['version']}  上传 {a['uploadedDate'][:16]}  状态 {a['processingState']}")
