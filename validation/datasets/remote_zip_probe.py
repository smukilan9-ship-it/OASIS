#!/usr/bin/env python
"""
List — and selectively fetch from — a remote ZIP without downloading it.

WHY. HyReCo.zip is 233 GB, and almost none of that is what OASIS needs. The expert landmarks
are CSVs measured in kilobytes; the bulk is 45 BigTIFF whole-slide images at ~5 GB each, and
a certification run thumbnails every one of them to 1920 px anyway. Pulling a quarter of a
terabyte to use a few hundred megabytes of it is worth one probe first.

HOW. A ZIP's central directory sits at the END of the file and names every member with its
offset and size. Python's `zipfile` only requires an object with read/seek/tell, so a file
object backed by HTTP range requests lets `zipfile` read that directory over the network and
then pull individual members — no full download, no temporary 233 GB file.

WHETHER IT WORKS depends on the server honouring `Range`. Signed S3 and CloudFront URLs
normally do; this prints a clear answer either way rather than silently downloading everything.

    # what is in there, and how big
    python validation/datasets/remote_zip_probe.py 'SIGNED_URL' --list

    # just the landmarks (kilobytes)
    python validation/datasets/remote_zip_probe.py 'SIGNED_URL' --get '*.csv' --out /Volumes/Expansion/HyReCo

    # one case's CD8 and H&E slides
    python validation/datasets/remote_zip_probe.py 'SIGNED_URL' \\
        --get '*29*CD8*' '*29*HE*' --out /Volumes/Expansion/HyReCo

The URL is time-limited and carries its own authorisation, so nothing secret is stored: pass
it on the command line, and it expires on its own. If it expires mid-transfer, re-copy a fresh
one from the browser and re-run — completed members are skipped.
"""
import argparse
import fnmatch
import io
import os
import sys
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP byte ranges, for zipfile to read a remote archive."""

    def __init__(self, url, timeout=60):
        self.url = url
        self.timeout = timeout
        self._pos = 0
        self.size = self._probe()

    def _probe(self):
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            n = r.headers.get("Content-Length")
            accepts = (r.headers.get("Accept-Ranges") or "").lower()
        if n is None:
            raise SystemExit("server did not report Content-Length — cannot range-read")
        if "bytes" not in accepts:
            # Not fatal: many servers omit the header but still honour Range. Verified below.
            print("  note: server did not advertise Accept-Ranges; testing a range anyway")
        return int(n)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=io.SEEK_SET):
        self._pos = (off if whence == io.SEEK_SET
                     else self._pos + off if whence == io.SEEK_CUR
                     else self.size + off)
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(self.url,
                                     headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise SystemExit(
                    f"server returned {r.status}, not 206 Partial Content — this URL does "
                    f"not support range requests, so the whole archive must be downloaded")
            data = r.read()
        self._pos += len(data)
        return data


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    ap.add_argument("--get", nargs="+", default=[], help="glob(s) of members to fetch")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    print("probing remote archive…")
    f = HTTPRangeFile(a.url)
    print(f"  archive size: {human(f.size)}")
    z = zipfile.ZipFile(f)
    infos = z.infolist()
    print(f"  members: {len(infos)}\n")

    if a.list or not a.get:
        by_ext = {}
        for i in infos:
            e = os.path.splitext(i.filename)[1].lower() or "(none)"
            b = by_ext.setdefault(e, [0, 0])
            b[0] += 1
            b[1] += i.file_size
        print(f"  {'ext':<10}{'count':>7}{'total':>14}")
        for e, (c, s) in sorted(by_ext.items(), key=lambda kv: -kv[1][1]):
            print(f"  {e:<10}{c:>7}{human(s):>14}")
        print("\n  largest members:")
        for i in sorted(infos, key=lambda x: -x.file_size)[:15]:
            print(f"    {human(i.file_size):>10}  {i.filename}")
        print("\n  all CSV members (the landmarks — this is what matters most):")
        for i in infos:
            if i.filename.lower().endswith(".csv"):
                print(f"    {human(i.file_size):>10}  {i.filename}")
        if not a.get:
            return

    want = [i for i in infos if any(fnmatch.fnmatch(i.filename, g) for g in a.get)]
    total = sum(i.file_size for i in want)
    print(f"\nfetching {len(want)} member(s), {human(total)} of {human(f.size)} "
          f"({100.0 * total / max(f.size, 1):.2f} %)")
    os.makedirs(a.out, exist_ok=True)
    for i in want:
        dst = os.path.join(a.out, i.filename)
        if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
            print(f"  skip (have)  {i.filename}")
            continue
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        print(f"  {human(i.file_size):>10}  {i.filename}")
        with z.open(i) as src, open(dst, "wb") as out:
            while True:
                chunk = src.read(8 << 20)
                if not chunk:
                    break
                out.write(chunk)
    print(f"\ndone -> {a.out}")


if __name__ == "__main__":
    main()
