# -*- coding: utf-8 -*-
"""邀请 TestFlight 测试员(2026-10-19):本机 DNS 不通苹果域名,此脚本只在
CI(macOS runner)上经 asc_tools.yml 运行。
用法: python asc_add_tester.py <email> <组名> [--internal]
流程: 签 ES256 JWT → 查/建 betaGroup(默认外部组;外部组首次要过 Beta 审核)
     → POST /v1/betaTesters 邀请(对方邮箱收 TestFlight 邀请函)。
内部组须对方已是 App Store Connect 团队成员(用户和访问里加),API 的
App 管理角色建不了团队成员——跨账号一律走外部组。
"""
import base64, json, os, sys, time, urllib.request
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives import hashes

APP_ID = "6820832214"
API = "https://api.appstoreconnect.apple.com"


def b64(d: bytes) -> str:
    return base64.urlsafe_b64encode(d).rstrip(b"=").decode()


def jwt() -> str:
    key_id, issuer = os.environ["ASC_KEY_ID"], os.environ["ASC_ISSUER_ID"]
    pem = base64.b64decode(os.environ["ASC_API_KEY_P8"])
    header = b64(json.dumps({"alg": "ES256", "kid": key_id}).encode())
    payload = b64(json.dumps({"iss": issuer, "iat": int(time.time()),
                              "exp": int(time.time()) + 1200,
                              "aud": "appstoreconnect-v1"}).encode())
    si = f"{header}.{payload}".encode()
    key = load_pem_private_key(pem, password=None)
    r, s = decode_dss_signature(key.sign(si, ec.ECDSA(hashes.SHA256())))
    return f"{header}.{payload}.{b64(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method,
                                 headers={"Authorization": f"Bearer {jwt()}",
                                          "Content-Type": "application/json"})
    data = json.dumps(body).encode() if body else None
    try:
        with urllib.request.urlopen(req, data=data) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {method} {path}\n{e.read().decode()[:800]}")


def main() -> None:
    email, group_name = sys.argv[1], sys.argv[2]
    internal = "--internal" in sys.argv
    groups = call("GET", f"/v1/betaGroups?filter[app]={APP_ID}&limit=200")
    gid = next((g["id"] for g in groups.get("data", [])
                if g["attributes"]["name"] == group_name), None)
    if not gid:
        made = call("POST", "/v1/betaGroups", {
            "data": {"type": "betaGroups",
                     "attributes": {"name": group_name, "isInternal": internal},
                     "relationships": {"app": {"data": {"id": APP_ID, "type": "apps"}}}}})
        gid = made["data"]["id"]
        print(f"已建组 {group_name} (内部={internal})")
    else:
        print(f"组已存在: {group_name}")
    call("POST", "/v1/betaTesters", {
        "data": {"type": "betaTesters",
                 "attributes": {"email": email},
                 "relationships": {
                     "apps": {"data": [{"id": APP_ID, "type": "apps"}]},
                     "betaGroups": {"data": [{"id": gid, "type": "betaGroups"}]}}}})
    print(f"已邀请 {email} → {group_name};对方邮箱将收到 TestFlight 邀请函")


if __name__ == "__main__":
    main()
