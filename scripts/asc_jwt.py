# 只生成 App Store Connect API 的 ES256 JWT(打印到 stdout),HTTP 请求交给 curl
import time, json, base64, sys
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives import hashes

KEY_PATH = Path(r"C:\Users\44957\Downloads\MediaSeek装机\AuthKey_8H66F4QJCN.p8")
KEY_ID = "8H66F4QJCN"
ISSUER = "fb64c8fb-5335-4bf6-933b-a8b1a7837103"

def b64(d: bytes) -> str:
    return base64.urlsafe_b64encode(d).rstrip(b"=").decode()

header = b64(json.dumps({"alg": "ES256", "kid": KEY_ID}).encode())
payload = b64(json.dumps({
    "iss": ISSUER, "iat": int(time.time()), "exp": int(time.time()) + 1200,
    "aud": "appstoreconnect-v1"}).encode())
signing_input = f"{header}.{payload}".encode()

key = load_pem_private_key(KEY_PATH.read_bytes(), password=None)
der = key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
r, s = decode_dss_signature(der)
print(f"{header}.{payload}.{b64(r.to_bytes(32,'big') + s.to_bytes(32,'big'))}")
