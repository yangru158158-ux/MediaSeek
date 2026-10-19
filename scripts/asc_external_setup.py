# -*- coding: utf-8 -*-
"""外部测试组配货(2026-10-19):邀请邮件不发出的头号原因=外部组没有已过
Beta 审核的构建。流程:查组→查 App 最新可用构建→把构建挂到组→提交
Beta 审核(betaAppReviewSubmissions)→打印测试员状态。
"""
import base64, json, os, sys, time, urllib.parse, urllib.request
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


def call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method,
                                 headers={"Authorization": f"Bearer {jwt()}",
                                          "Content-Type": "application/json"})
    data = json.dumps(body).encode() if body else None
    try:
        with urllib.request.urlopen(req, data=data) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def q(**params) -> str:
    return "?" + urllib.parse.urlencode(params)


def main() -> None:
    # ASC 团队用户名单(权威核验:邀请到底发没发)
    st, users = call("GET", "/v1/users?limit=200")
    print(f"  /v1/users HTTP {st}")
    if isinstance(users, dict):
        data = users.get("data", [])
        print(f"  团队用户数: {len(data)}")
        for u in data:
            a = u.get("attributes", {})
            roles = [d.get("id", "?") for d in
                     (u.get("relationships", {}).get("roles", {}).get("data") or [])]
            print(f"  ASC用户: {a.get('username')} 名={a.get('firstName','')}{a.get('lastName','')} roleIds={roles}")
        if not data:
            print(f"  原始返回片段: {str(users)[:400]}")
    else:
        print(f"  非dict返回: {str(users)[:400]}")
    group_name = sys.argv[1] if len(sys.argv) > 1 else "亲友外测"
    st, groups = call("GET", "/v1/betaGroups" + q(**{"filter[app]": APP_ID, "limit": 200}))
    gid = next((g["id"] for g in groups.get("data", [])
                if g["attributes"]["name"] == group_name), None)
    print(f"组 {group_name}: {gid}")
    if not gid:
        return
    # 组的测试员状态
    st, testers = call("GET", f"/v1/betaGroups/{gid}/betaTesters")
    for t in testers.get("data", []):
        print(f"  测试员: {t['attributes'].get('email')} 状态={t['attributes'].get('status')}")
    # 组的构建
    st, builds = call("GET", f"/v1/betaGroups/{gid}/builds")
    print(f"  组内构建数: {len(builds.get('data', []))}")
    # App 最新已处理构建
    # filter[processed] 不是合法参数(400 实测)——拉回本地筛 processed
    st, latest = call("GET", "/v1/builds" + q(**{
        "filter[app]": APP_ID,
        "sort": "-uploadedDate", "limit": 10}))
    if isinstance(latest, str) or st >= 300:
        print(f"  构建查询失败 HTTP {st}: {latest}")
        return
    for b in latest.get("data", [])[:10]:
        a = b.get("attributes", {})
        print(f"    build v{a.get('version')} processed={a.get('processed')} "
              f"state={a.get('processingState')} uploaded={a.get('uploadedDate')}")
    # processed 属性已空壳(None),真判据=processingState=="VALID"
    items = [b for b in latest.get("data", [])
             if b.get("attributes", {}).get("processingState") == "VALID"]
    if not items:
        print("  没有已处理的构建!")
        return
    b = items[0]
    bid, ver = b["id"], b["attributes"]["version"]
    print(f"  最新可用构建: v{ver} (id={bid})")
    # 挂到组
    st, resp = call("POST", f"/v1/betaGroups/{gid}/relationships/builds",
                    {"data": [{"id": bid, "type": "builds"}]})
    print(f"  挂构建到组: HTTP {st} {resp if st >= 300 else 'OK'}")
    # 前置一:Beta App 描述(外部测试硬性要求,422 实测)
    st, locs = call("GET", "/v1/betaAppLocalizations" + q(**{"filter[app]": APP_ID}))
    loc = next((x for x in locs.get("data", [])
                if x["attributes"].get("locale") == "zh-Hans"),
               (locs.get("data") or [None])[0])
    desc = "个人相册端侧语义搜索测试版:人物归组、文字/语义/组合检索、查重。仅供内部测试,不会公开发布。"
    if loc:
        st, resp = call("PATCH", f"/v1/betaAppLocalizations/{loc['id']}", {
            "data": {"type": "betaAppLocalizations", "id": loc["id"],
                     "attributes": {"description": desc,
                                    "feedbackEmail": "44957799@qq.com"}}})
        print(f"  补描述(PATCH): HTTP {st}")
    else:
        # 新 App 可能一条 Beta 本地化都没有——POST 新建
        st, resp = call("POST", "/v1/betaAppLocalizations", {
            "data": {"type": "betaAppLocalizations",
                     "attributes": {"locale": "zh-Hans", "description": desc,
                                    "feedbackEmail": "44957799@qq.com"},
                     "relationships": {"app": {"data": {"id": APP_ID, "type": "apps"}}}}})
        print(f"  补描述(POST 新建): HTTP {st} {resp if st >= 300 else ''}")
    # 前置二:出口合规——端侧推理,不含非豁免加密
    # 合规字段在 build 本体(不在 buildBetaDetails,409 实测):端侧推理,不含非豁免加密
    st, resp = call("PATCH", f"/v1/builds/{bid}", {
        "data": {"type": "builds", "id": bid,
                 "attributes": {"usesNonExemptEncryption": False}}})
    print(f"  出口合规(false): HTTP {st} {resp if st >= 300 else ''}")
    # 前置三:Beta 审核联系人(外部测试必需;argv: 组名 姓名 邮箱 电话)
    if len(sys.argv) >= 5:
        # betaReviewDetails 是 App 的子资源:路径必须挂在 App 下(独立集合 404 实测)
        # 资源真名 betaAppReviewDetail(Beta App 审核详情,App 的一对一子资源)
        st, dets = call("GET", f"/v1/apps/{APP_ID}/betaAppReviewDetail")
        d = dets.get("data") if isinstance(dets, dict) else None
        body = {"data": {"type": "betaAppReviewDetails",
                         "attributes": {"contactFirstName": sys.argv[2],
                                        "contactLastName": sys.argv[3],
                                        "contactEmail": sys.argv[4],
                                        "contactPhone": sys.argv[5]}}}
        if d:
            body["data"]["id"] = d["id"]
            st, resp = call("PATCH", f"/v1/betaAppReviewDetails/{d['id']}", body)
        else:
            body["data"]["relationships"] = {"app": {"data": {"id": APP_ID, "type": "apps"}}}
            st, resp = call("POST", "/v1/betaAppReviewDetails", body)
        print(f"  审核联系人: HTTP {st} {resp if st >= 300 else ''}")
    # 提交 Beta 审核(外部组首次分发必需)
    st, resp = call("POST", "/v1/betaAppReviewSubmissions", {
        "data": {"type": "betaAppReviewSubmissions",
                 "relationships": {"build": {"data": {"id": bid, "type": "builds"}}}}})
    print(f"  Beta 审核提交: HTTP {st} {resp if st >= 300 else 'OK'}")
    print("  → 审核通过后(几小时~一天)邀请邮件自动发出")


if __name__ == "__main__":
    main()
