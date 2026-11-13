#!/usr/bin/env python3
"""Resumable parallel HTTP range downloader for large public atlas files."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import ssl
import time
import urllib.request
from pathlib import Path

import certifi


SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
        return int(response.headers["Content-Length"])


def fetch_part(url: str, part: Path, start: int, end: int) -> tuple[int, int]:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return start, expected
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as response:
                payload = response.read()
            if len(payload) != expected:
                raise IOError(f"range {start}-{end}: expected {expected}, got {len(payload)}")
            temporary = part.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(part)
            return start, expected
        except Exception:
            if attempt == 11:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(args.url)
    current = args.output.stat().st_size if args.output.exists() else 0
    if current > size:
        raise RuntimeError(f"local file is larger than remote: {current} > {size}")
    if current == size:
        print(f"complete {args.output} ({size:,} bytes)", flush=True)
        return

    chunk = args.chunk_mib * 1024 * 1024
    parts_dir = args.output.with_suffix(args.output.suffix + ".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = [(start, min(size - 1, start + chunk - 1))
              for start in range(current, size, chunk)]
    print(f"resume {current:,}/{size:,}; {len(ranges)} ranges; {args.workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for start, end in ranges:
            part = parts_dir / f"{start:020d}-{end:020d}.part"
            futures.append(pool.submit(fetch_part, args.url, part, start, end))
        done = 0
        for future in concurrent.futures.as_completed(futures):
            _, amount = future.result()
            done += amount
            print(f"downloaded {done:,}/{size-current:,} remaining bytes", flush=True)

    with args.output.open("ab") as output:
        for start, end in ranges:
            part = parts_dir / f"{start:020d}-{end:020d}.part"
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
            part.unlink()
    parts_dir.rmdir()
    if args.output.stat().st_size != size:
        raise RuntimeError("assembled file size mismatch")
    print(f"complete {args.output} ({size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
