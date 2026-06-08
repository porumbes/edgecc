#!/usr/bin/env python3
"""Offline unit tests for tools/fetch_suitesparse.py helpers (no network)."""
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, TOOLS)

import fetch_suitesparse as fs

results = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}")
    results.append(bool(cond))

def _make_tarball(member_name, payload):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()

def test_extract_mtx_nested():
    payload = b"%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n1 2\n"
    blob = _make_tarball("karate/karate.mtx", payload)
    got = fs.extract_mtx(blob, "karate")
    check("extract_mtx finds nested member", got == payload)

def test_extract_mtx_missing():
    blob = _make_tarball("karate/karate_other.txt", b"x")
    try:
        fs.extract_mtx(blob, "karate")
        check("extract_mtx raises on missing member", False)
    except ValueError:
        check("extract_mtx raises on missing member", True)

def test_sha256_bytes():
    # sha256("") is the well-known empty-string digest
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    check("sha256_bytes empty", fs.sha256_bytes(b"") == empty)

test_extract_mtx_nested()
test_extract_mtx_missing()
test_sha256_bytes()

print(f"\n{sum(results)}/{len(results)} fetch-helper tests passed")
sys.exit(0 if all(results) else 1)
