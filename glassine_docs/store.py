# SPDX-License-Identifier: GPL-2.0-or-later
"""The binary container of the vector and model files (see docs/format.md).

    magic   8 bytes  b"GLDSRCH1"
    hlen    u32 LE   length of the JSON header in bytes
    header  hlen bytes UTF-8 JSON: {"meta": {...}, "arrays": {name: {dtype, shape, offset}}}
    data    arrays, each starting at a multiple of 8 from the start of the file

dtypes: "i8", "u8", "f32" (little endian).
"""
from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"GLDSRCH1"
_DT = {"i8": np.int8, "u8": np.uint8, "f32": np.dtype("<f4")}


def _dtname(a: np.ndarray) -> str:
    for k, v in _DT.items():
        if a.dtype == v:
            return k
    raise TypeError(f"unsupported dtype {a.dtype}")


def pack(meta: dict, arrays: dict[str, np.ndarray]) -> bytes:
    names = list(arrays)
    # Two passes: offsets depend on the header length, which depends on the offsets.
    offsets = {n: 0 for n in names}
    for _ in range(3):
        header = {"meta": meta, "arrays": {
            n: {"dtype": _dtname(arrays[n]), "shape": list(arrays[n].shape), "offset": offsets[n]}
            for n in names}}
        hb = json.dumps(header, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        pos = len(MAGIC) + 4 + len(hb)
        new = {}
        for n in names:
            pos = (pos + 7) // 8 * 8
            new[n] = pos
            pos += arrays[n].nbytes
        if new == offsets:
            break
        offsets = new
    out = bytearray(MAGIC + struct.pack("<I", len(hb)) + hb)
    for n in names:
        out.extend(b"\0" * (offsets[n] - len(out)))
        out.extend(np.ascontiguousarray(arrays[n]).astype(_DT[_dtname(arrays[n])], copy=False).tobytes())
    return bytes(out)


def unpack(data: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    if data[:8] != MAGIC:
        raise ValueError("not a glassine search file")
    (hlen,) = struct.unpack_from("<I", data, 8)
    header = json.loads(data[12:12 + hlen].decode())
    arrays = {}
    for n, a in header["arrays"].items():
        dt = np.dtype(_DT[a["dtype"]])
        count = int(np.prod(a["shape"])) if a["shape"] else 1
        arrays[n] = np.frombuffer(data, dtype=dt, count=count, offset=a["offset"]).reshape(a["shape"])
    return header["meta"], arrays
