#!/usr/bin/env python3
"""远程探针:不下载 2.1GB 的 models-so400m.zip,用 HTTP Range 只取
zip 尾部中央目录,验证包结构符合 App 导入器(importModelZip)的预期。

验证点(第四十六批 so400m 切换的前置条件):
  1. zip 根下有 ImageEmbedder/ 目录(导入器按目录名认槽位)
  2. 含 SiglipImage.mlpackage / SiglipText.mlpackage / tokenizer.json / meta.json
  3. meta.json: type=clip、dim=1152(so400m)、image_size=384
     —— dim>=1024 正是导入器自动路由 .imagePro 槽位的启发式条件
用法:
  python scripts/probe_so400m_zip.py <github_token>
"""
import json
import struct
import subprocess
import sys
import urllib.request


def api(token, url):
    req = urllib.request.Request(url, headers={"Authorization": f"token {token}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def fetch_range_via_curl(token, asset_id, rng):
    """assets API 层不吃 Range(501):先 HEAD 拿 302 的签名 URL,
    再对签名 URL(objects.githubusercontent.com)发 Range 取片段"""
    api_url = f"https://api.github.com/repos/yangru158158-ux/MediaSeek/releases/assets/{asset_id}"
    loc = subprocess.run(
        ["curl", "-sI", "-H", f"Authorization: token {token}",
         "-H", "Accept: application/octet-stream", api_url],
        capture_output=True, text=True).stdout
    signed = next(l.split(":", 1)[1].strip() for l in loc.splitlines()
                  if l.lower().startswith("location:"))
    out = subprocess.run(
        ["curl", "-sL", "-H", f"Range: bytes={rng}", signed,
         "-o", "zip_tail.bin", "-w", "%{http_code}"],
        capture_output=True, text=True).stdout.strip()
    if out not in ("206", "200"):
        raise RuntimeError(f"range fetch HTTP {out}")
    with open("zip_tail.bin", "rb") as f:
        return f.read()


def parse_eocd(tail):
    """从尾部字节找 EOCD,返回(目录条数, 目录偏移, 目录大小)。
    反向搜索并校验边界,跳过数据区里截断的签名误报"""
    idx = tail.rfind(b"PK\x05\x06")
    while idx >= 0 and idx + 22 > len(tail):
        idx = tail.rfind(b"PK\x05\x06", 0, idx)
    if idx < 0:
        raise RuntimeError("EOCD 未找到(尾部片段太短?)")
    # EOCD: 签名4 + disk2 + cd盘2 + 本盘条数2 + 总条数2 + 目录大小4 + 目录偏移4
    _, _, n, _, cd_size, cd_off = struct.unpack("<HHHHII", tail[idx + 4:idx + 20])
    return n, cd_off, cd_size


def parse_central_dir(tail, file_size, cd_off, cd_size):
    """zip64 无需处理(单资产<4GB);把尾部片段内的 CD 条目解出来"""
    base = len(tail) - (file_size - cd_off)   # cd_off 在片段内的起点
    entries, p = [], base
    end = base + cd_size
    while p < end and tail[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", tail[p + 10:p + 12])
        csize, usize = struct.unpack("<II", tail[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", tail[p + 28:p + 34])
        off, = struct.unpack("<I", tail[p + 42:p + 46])
        name = tail[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        entries.append(dict(name=name, method=method, csize=csize, usize=usize, offset=off))
        p += 46 + nlen + elen + clen
    return entries


def fetch_file_bytes(token, asset_id, entry, file_size):
    """局部头里文件名变长不定长——多取一点再按魔数切"""
    start = entry["offset"]
    span = min(4096 + entry["csize"], 1 << 20)
    data = fetch_range_via_curl(token, asset_id, f"{start}-{start + span - 1}")
    sig = data.find(b"PK\x03\x04")
    nlen, elen = struct.unpack("<HH", data[sig + 26:sig + 30])
    body = data[sig + 30 + nlen + elen: sig + 30 + nlen + elen + entry["csize"]]
    if entry["method"] == 0:
        return body
    import zlib
    return zlib.decompress(body, -15)


def main():
    token = sys.argv[1]
    rel = api(token, "https://api.github.com/repos/yangru158158-ux/MediaSeek/releases/tags/models")
    asset = next(a for a in rel["assets"] if a["name"] == "models-so400m.zip")
    size = asset["size"]
    print(f"asset: {asset['name']}  {size/1e6:.0f}MB  id={asset['id']}")

    tail_len = min(262144, size)
    # 该 CDN 边缘(Varnish)不吃后缀区间 bytes=-N(501 Unsupported client
    # range),必须发显式区间 bytes=start-end
    tail = fetch_range_via_curl(token, asset["id"], f"{size - tail_len}-{size - 1}")
    n, cd_off, cd_size = parse_eocd(tail)
    entries = parse_central_dir(tail, size, cd_off, cd_size)
    print(f"EOCD: {n} 个条目,解析出 {len(entries)} 个")

    roots = sorted({e["name"].split("/")[0] for e in entries})
    print("根目录:", roots)
    assert roots == ["ImageEmbedder"], "zip 根必须是 ImageEmbedder/(导入器按目录名认槽位)"

    need = ["SiglipImage.mlpackage", "SiglipText.mlpackage", "tokenizer.json", "meta.json"]
    top = {e["name"].split("/", 1)[1] for e in entries if "/" in e["name"]}
    for item in need:
        hit = any(t == item or t.startswith(item + "/") for t in top)
        assert hit, f"缺少 {item}"
        print(f"  ✓ {item}")

    meta_entry = next(e for e in entries if e["name"] == "ImageEmbedder/meta.json")
    meta = json.loads(fetch_file_bytes(token, asset["id"], meta_entry, size))
    print("meta.json:", json.dumps(meta, ensure_ascii=False))
    assert meta["type"] == "clip"
    assert meta["dim"] == 1152, f"dim 应为 1152(so400m),实际 {meta['dim']}"
    assert meta["image_size"] == 384, f"image_size 应为 384,实际 {meta['image_size']}"
    print("\nPASS: so400m 包结构与 meta 维度全部符合导入器/切换设计预期")


if __name__ == "__main__":
    main()
