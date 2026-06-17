#!/usr/bin/env python3
"""
Recursively download a password-protected share from rec.ustc.edu.cn (USTC 睿客网).


The web UI is a JS SPA backed by a public JSON API at https://recapi.ustc.edu.cn/api/v2.
Password-protected shares need no login token: the password is sent in
`share_constraint.password` on every call.

Default settings point at the DiffusionForensics share:
  https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070  (password: dire)

Usage:
  python3 download_rec_share.py                      # download everything to ./DiffusionForensics_download
  python3 download_rec_share.py -o /data/df          # custom output dir
  python3 download_rec_share.py --subpath images/test/lsun_bedroom   # only a subtree
  python3 download_rec_share.py --list-only          # print the tree, download nothing
  python3 download_rec_share.py --share <uuid> --password <pw>

Re-running is safe: files whose local size already matches the remote size are skipped,
so an interrupted run resumes by just running it again.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

API = "https://recapi.ustc.edu.cn/api/v2"
LIST_URL = f"{API}/share/target/resource/list"
DOWNLOAD_URL = f"{API}/share/download"
PAGE = 500          # items per listing page
RETRIES = 5         # retries per network operation

# stdlib only — no `requests`/`pip` needed, runs on any Python 3.
NETERR = (urllib.error.URLError, OSError, ValueError)


def _post(url, payload):
    """POST returning parsed JSON entity, retrying on transient errors."""
    last = None
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8-sig"))  # responses carry a UTF-8 BOM
            if data.get("status_code") == 200:
                return data["entity"]
            # non-200 app status: surface message (e.g. wrong password)
            raise RuntimeError(f"API error {data.get('status_code')}: {data.get('message')}")
        except NETERR as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {RETRIES} tries: {last}")


def list_folder(share, password, resource_number):
    """Yield all entries inside a folder (None = share root), handling pagination."""
    offset = 0
    while True:
        entity = _post(LIST_URL, {
            "share_number": share,
            "share_resource_number": resource_number,
            "is_rec": "false",
            "share_constraint": {"password": password},
            "offset": offset,
            "limit": PAGE,
        })
        if not entity:
            break
        for item in entity:
            yield item
        if len(entity) < PAGE:
            break
        offset += PAGE


def get_download_url(share, password, number):
    entity = _post(DOWNLOAD_URL, {
        "share_number": share,
        "share_constraint": {"password": password},
        "share_resources_list": [number],
    })
    url = entity[number]
    return url + "&download=download"


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _progress(got, size, elapsed, final=False):
    """Render an in-place progress line: percent, downloaded, speed, ETA."""
    speed = got / elapsed if elapsed > 0 else 0
    if size:
        pct = got / size * 100
        eta = (size - got) / speed if speed > 0 else 0
        line = f"    {pct:5.1f}%  {human(got)}/{human(size)}  {human(speed)}/s  ETA {int(eta)}s"
    else:
        line = f"    {human(got)}  {human(speed)}/s"
    end = "\n" if final else ""
    print(f"\r{line:<70}{end}", end="", flush=True)


def download_file(share, password, item, dest, live):
    """Download one file. Returns ('skip'|'done', bytes). `live` shows an in-place
    progress line (only sensible with a single worker)."""
    size = int(item["bytes"]) if str(item["bytes"]).strip() else 0
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        return "skip", size
    tmp = dest + ".part"
    url = get_download_url(share, password, item["number"])
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as r:
                got = 0
                start = last_print = time.time()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        now = time.time()
                        if live and now - last_print >= 0.5:
                            _progress(got, size, now - start)
                            last_print = now
                if live:
                    _progress(got, size, time.time() - start, final=True)
            if size and got != size:
                raise IOError(f"size mismatch: got {got}, expected {size}")
            os.replace(tmp, dest)
            return "done", got
        except NETERR as e:
            print(f"  retry {attempt+1}/{RETRIES} {dest}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
            url = get_download_url(share, password, item["number"])  # URL may expire
    raise RuntimeError(f"failed to download {dest}")


def walk_list(share, password, resource_number, local_dir):
    """Print the tree without downloading (for --list-only)."""
    for item in list_folder(share, password, resource_number):
        local = os.path.join(local_dir, item["name"])
        if item["type"] == "folder":
            print(f"[dir]  {local}")
            walk_list(share, password, item["number"], local)
        else:
            size = int(item["bytes"]) if str(item["bytes"]).strip() else 0
            print(f"[file] {local}  ({human(size)})")


def collect_files(share, password, resource_number, local_dir):
    """Recursively gather (item, dest) for every file, creating directories."""
    os.makedirs(local_dir, exist_ok=True)
    files = []
    for item in list_folder(share, password, resource_number):
        local = os.path.join(local_dir, item["name"])
        if item["type"] == "folder":
            files.extend(collect_files(share, password, item["number"], local))
        else:
            files.append((item, local))
    return files


def download_all(share, password, files, jobs):
    """Download all (item, dest) pairs using `jobs` parallel workers."""
    total = len(files)
    counter = {"n": 0}
    lock = threading.Lock()
    live = jobs == 1

    def work(idx_item_dest):
        item, dest = idx_item_dest
        t0 = time.time()
        status, got = download_file(share, password, item, dest, live)
        with lock:
            counter["n"] += 1
            n = counter["n"]
            if status == "skip":
                print(f"[{n}/{total}] skip  {dest}  ({human(got)})")
            else:
                dt = max(time.time() - t0, 1e-6)
                print(f"[{n}/{total}] done  {dest}  ({human(got)} @ {human(got/dt)}/s)")

    if jobs == 1:
        for f in files:
            work(f)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(work, files))


def resolve_subpath(share, password, subpath):
    """Walk a slash-separated subpath from the root, returning the folder number."""
    number = None
    for part in [p for p in subpath.split("/") if p]:
        match = next((i for i in list_folder(share, password, number)
                      if i["name"] == part and i["type"] == "folder"), None)
        if match is None:
            sys.exit(f"subpath component not found: {part!r}")
        number = match["number"]
    return number


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", default="ec980150-4615-11ee-be0a-eb822f25e070",
                    help="share UUID (from the /share/<uuid> URL)")
    ap.add_argument("--password", default="dire", help="share access password")
    ap.add_argument("-o", "--output", default="DiffusionForensics_download",
                    help="local output directory")
    ap.add_argument("--subpath", default="",
                    help="only download this subtree, e.g. 'images/test/lsun_bedroom'")
    ap.add_argument("--list-only", action="store_true",
                    help="print the file tree without downloading")
    ap.add_argument("-j", "--jobs", type=int, default=4,
                    help="parallel downloads (the server throttles per-connection, "
                         "so >1 is faster; default 4)")
    args = ap.parse_args()

    root_number = resolve_subpath(args.share, args.password, args.subpath) if args.subpath else None
    print(f"Share {args.share}  ->  {args.output}" + (f"  (subpath: {args.subpath})" if args.subpath else ""))

    if args.list_only:
        walk_list(args.share, args.password, root_number, args.output)
        print("Listing complete.")
        return

    print("Scanning tree...")
    files = collect_files(args.share, args.password, root_number, args.output)
    total_bytes = sum(int(i["bytes"]) for i, _ in files if str(i["bytes"]).strip())
    print(f"{len(files)} files, {human(total_bytes)} total. Downloading with {args.jobs} job(s)...")
    download_all(args.share, args.password, files, max(1, args.jobs))
    print("All done.")


if __name__ == "__main__":
    main()
