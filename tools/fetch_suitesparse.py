#!/usr/bin/env python3
"""Fetch curated SuiteSparse .mtx graphs for edgecc's test set.

Reads data/suitesparse/manifest.json, downloads each requested graph's Matrix
Market tarball from sparse.tamu.edu, extracts the inner <name>.mtx, verifies
its sha256 against the manifest, and writes data/suitesparse/<name>.mtx.
Stdlib only (no new runtime dependency).

Usage:
  fetch_suitesparse.py <name> [<name> ...]   # fetch specific graphs
  fetch_suitesparse.py --all                 # fetch every graph in the manifest
  fetch_suitesparse.py --all --update-hashes # (re)compute sha256 into manifest
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "suitesparse")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
BASE_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def extract_mtx(tar_bytes, name):
    """Return the bytes of <name>.mtx inside a SuiteSparse tar.gz blob."""
    leaf = "%s.mtx" % name
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            # Match the member at <name>.mtx, anchored to a path separator so
            # e.g. name="karate" does not match "notkarate.mtx".
            if m.name == leaf or m.name.endswith("/" + leaf):
                f = tf.extractfile(m)
                if f is not None:
                    return f.read()
        raise ValueError("no %s found in tarball" % leaf)


def download(group, name):
    url = BASE_URL.format(group=group, name=name)
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise SystemExit("download failed for %s (%s): %s" % (name, url, e))


def fetch_one(entry, update_hashes=False, force=False):
    name = entry["name"]
    dest = os.path.join(DATA_DIR, "%s.mtx" % name)

    if os.path.exists(dest) and not force and not update_hashes:
        with open(dest, "rb") as f:
            have = sha256_bytes(f.read())
        want = entry.get("sha256")
        if want and have != want:
            raise SystemExit("cached checksum mismatch for %s: have %s want %s" % (name, have, want))
        print("ok (cached): %s" % name)
        return

    blob = download(entry["group"], name)
    mtx = extract_mtx(blob, name)
    digest = sha256_bytes(mtx)

    if update_hashes:
        entry["sha256"] = digest
        print("hash %s: %s" % (name, digest))
    else:
        want = entry.get("sha256")
        if not want:
            raise SystemExit("manifest has no sha256 for %s; run with --update-hashes first" % name)
        if digest != want:
            raise SystemExit("checksum mismatch for %s: got %s want %s" % (name, digest, want))

    with open(dest, "wb") as f:
        f.write(mtx)
    print("wrote %s (%d bytes)" % (dest, len(mtx)))


def main():
    ap = argparse.ArgumentParser(description="Fetch SuiteSparse graphs for edgecc tests.")
    ap.add_argument("names", nargs="*", help="manifest keys to fetch")
    ap.add_argument("--all", action="store_true", help="fetch every graph in the manifest")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--update-hashes", action="store_true",
                    help="download and write computed sha256 back into the manifest")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.all or args.update_hashes:
        keys = list(manifest)
    elif args.names:
        keys = args.names
    else:
        ap.error("specify graph names, or --all")

    os.makedirs(DATA_DIR, exist_ok=True)
    for k in keys:
        if k not in manifest:
            raise SystemExit("unknown graph %r (not in manifest)" % k)
        fetch_one(manifest[k], update_hashes=args.update_hashes, force=args.force)

    if args.update_hashes:
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print("manifest updated")


if __name__ == "__main__":
    main()
