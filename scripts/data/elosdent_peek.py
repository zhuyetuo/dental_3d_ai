"""用系统的 libarchive（zip / rar / 7z 都能读）从各分包里各取一个样本，看文件头能不能读。只读，不解压到磁盘。"""
import ctypes as c
import ctypes.util
import fnmatch
import sys
from pathlib import Path

INV = Path.home() / "zyt/dental_data/.elosdent_inventory"
a = c.CDLL(ctypes.util.find_library("archive") or "libarchive.so.13")
P = c.c_void_p


def api(name, res, args):
    f = getattr(a, name)
    f.restype, f.argtypes = res, args
    return f


new = api("archive_read_new", P, [])
sf = api("archive_read_support_filter_all", c.c_int, [P])
sfm = api("archive_read_support_format_all", c.c_int, [P])
openf = api("archive_read_open_filename", c.c_int, [P, c.c_char_p, c.c_size_t])
nxt = api("archive_read_next_header", c.c_int, [P, c.POINTER(P)])
pname = api("archive_entry_pathname", c.c_char_p, [P])
psize = api("archive_entry_size", c.c_longlong, [P])
rd = api("archive_read_data", c.c_ssize_t, [P, c.c_void_p, c.c_size_t])
free = api("archive_read_free", c.c_int, [P])


def fix(s):
    try:
        return s.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def first(bundle, pattern, nbytes=4096):
    """返回 (包内路径, 文件大小, 前 nbytes 字节)：第一个匹配 pattern（按还原后的中文名匹配）的文件。"""
    h = new(); sf(h); sfm(h)
    if openf(h, str(INV / bundle).encode(), 1 << 16) < 0:
        return None
    e = P()
    try:
        while nxt(h, c.byref(e)) in (0, -20):
            raw = pname(e)
            if not raw:
                continue
            n = fix(raw.decode("utf-8", "replace"))
            if fnmatch.fnmatch(n, pattern):
                buf = c.create_string_buffer(nbytes)
                got = rd(h, buf, nbytes)
                return n, psize(e), buf.raw[: max(got, 0)]
    finally:
        free(h)


def show(bundle, pattern):
    r = first(bundle, pattern)
    if r is None:
        print(f"== {bundle} {pattern}: 没找到"); return
    n, size, head = r
    txt = head[:400]
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in txt) / max(1, len(txt))
    print(f"== {bundle}: {n}（{size / 1e3:.0f} KB）  可读文本比例 {printable:.0%}")
    print("   hex:", head[:32].hex(" "))
    if printable > 0.85:
        print("   文本:", head[:600].decode("utf-8", "replace").replace("\r", "")[:600])
    else:
        print("   开头字符:", "".join(chr(b) if 32 <= b < 127 else "." for b in head[:96]))


for args in [("bundle_04.rar", "*.eoff"), ("bundle_02.7z", "*.drmsdfa"), ("bundle_02.7z", "*.eoff"), ("bundle_02.7z", "*.off"),
             ("bundle_04.rar", "*upperjaw.xml"), ("bundle_06.zip", "*CAD/* 0.dcm"), ("bundle_06.zip", "*Raw Antagonist scan.dcm"),
             ("bundle_06.zip", "*/2311110229-5036.xml"), ("bundle_08.rar", "*.dme"), ("bundle_08.rar", "*.obj"),
             ("bundle_01.zip", "*.constructionInfo"), ("bundle_10.zip", "{*}"), ("bundle_10.zip", "*.3ml")]:
    try:
        show(*args)
    except Exception as exc:  # noqa: BLE001
        print("==", args, "失败", type(exc).__name__, exc)
    sys.stdout.flush()
