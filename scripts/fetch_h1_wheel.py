"""Fetch pinned H1 binary artifacts with bounded, resumable range requests."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--size", type=int)
    parser.add_argument("--sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    config = json.loads((ROOT / "config/serving/h1_nanovllm.json").read_text())["environment"]
    if any((args.url, args.size, args.sha256, args.output)) and not all((args.url, args.size, args.sha256, args.output)):
        parser.error("Custom downloads require --url, --size, --sha256 and --output together")
    url = args.url or config["flash_attn_wheel"]
    size = args.size or config["flash_attn_wheel_bytes"]
    expected = args.sha256 or config["flash_attn_wheel_sha256"]
    directory = ROOT / "runs/downloads" / (expected + ".parts")
    directory.mkdir(parents=True, exist_ok=True)
    output = args.output or ROOT / "runs/downloads" / url.rsplit("/", 1)[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == size:
        with output.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == expected:
                print(f"Already verified {expected}: {output}")
                return
    chunk_size = 8 * 1024 * 1024

    def fetch(index):
        start = index * chunk_size
        end = min(start + chunk_size, size) - 1
        path = directory / f"{index:04d}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(args.retries):
            try:
                request = Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=args.timeout) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Server did not honor the requested byte range")
                    with path.open("wb") as target:
                        while data := response.read(1024 * 1024):
                            target.write(data)
                if path.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete range")
                print(f"range {index + 1} complete", flush=True)
                return path
            except Exception:
                if attempt == args.retries - 1:
                    raise
                time.sleep(attempt + 1)

    count = (size + chunk_size - 1) // chunk_size
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(fetch, range(count)))
    checksum = hashlib.sha256()
    temporary = output.with_suffix(".whl.tmp")
    with temporary.open("wb") as target:
        for path in paths:
            with path.open("rb") as source:
                while data := source.read(1024 * 1024):
                    checksum.update(data)
                    target.write(data)
    if checksum.hexdigest() != expected:
        raise RuntimeError("Wheel SHA256 differs from the official pinned release")
    temporary.replace(output)
    print(f"Verified {expected}: {output}")


if __name__ == "__main__":
    main()
