"""Reads one NVFP4 expert's nine weight and scale tensors off the SSD into caller-supplied
byte buffers, from either on-disk layout."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import struct
import sys
import threading
from dataclasses import dataclass

import numpy as np

from freetoken.utils import init_logger, set_nocache_fd

logger = init_logger(__name__)

# Alignment the repack pads its records to, and the alignment direct I/O needs.
_PAGE = 4096

# F_RDADVISE (Darwin <sys/fcntl.h>): an async read of a byte range with no copy to user, which
# Python's fcntl lacks. Argument: struct radvisory {off_t ra_offset; int ra_count}.
_F_RDADVISE = 44

_KEY_RE = re.compile(
    r"model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)

_PROJ_ROLE = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
_KIND_FIELD = {"weight": "weight", "weight_scale": "scale", "weight_scale_2": "global"}

# The nine raw checkpoint tensors per expert, in the order the repack writes them and the
# disk cache reads them back.
PIECE_ORDER = (
    "gate_weight", "gate_scale", "gate_global", "up_weight", "up_scale", "up_global", "down_weight",
    "down_scale", "down_global",
)


def config_sha(model_path: str) -> str:
    """Fingerprint of a checkpoint's ``config.json``, so a stale repack is detectable."""
    with open(os.path.join(model_path, "config.json"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class TensorLoc:
    """Where one checkpoint tensor's bytes are: shard, absolute offset, size, dtype, shape."""

    shard_path: str
    offset: int
    nbytes: int
    dtype: str  # safetensors dtype string, e.g. "U8", "F8_E4M3", "F16"
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ExpertLocation:
    """One expert's nine tensor locations, named ``<projection>_<field>``."""

    layer: int
    expert: int
    gate_weight: TensorLoc
    gate_scale: TensorLoc
    gate_global: TensorLoc
    up_weight: TensorLoc
    up_scale: TensorLoc
    up_global: TensorLoc
    down_weight: TensorLoc
    down_scale: TensorLoc
    down_global: TensorLoc

    def pieces(self) -> dict[str, list[TensorLoc]]:
        """The nine locations grouped into the six banks the cache holds; gate and up are
        separate tensors on disk, so three of the six take two reads each."""
        return {
            "gate_up_packed": [self.gate_weight, self.up_weight],
            "gate_up_scale": [self.gate_scale, self.up_scale],
            "gate_up_global": [self.gate_global, self.up_global],
            "down_packed": [self.down_weight],
            "down_scale": [self.down_scale],
            "down_global": [self.down_global],
        }

    def all_locs(self) -> list[TensorLoc]:
        return [getattr(self, name) for name in PIECE_ORDER]


class Nvfp4DiskIndex:
    """Maps (layer, expert) -> :class:`ExpertLocation` for one NVFP4 checkpoint directory,
    from ``model.safetensors.index.json`` plus each shard's safetensors header (an 8-byte
    little-endian length followed by JSON). No tensor payload is ever read."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        shard_headers: dict[str, dict] = {}
        raw: dict[tuple[int, int], dict[str, TensorLoc]] = {}
        for name, shard in weight_map.items():
            match = _KEY_RE.match(name)
            if match is None:
                continue
            shard_path = os.path.join(model_path, shard)
            header = shard_headers.get(shard_path)
            if header is None:
                with open(shard_path, "rb") as f:
                    header_len = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(header_len))
                header.pop("__metadata__", None)
                header["__payload_base__"] = 8 + header_len
                shard_headers[shard_path] = header
            entry = header[name]
            start, end = entry["data_offsets"]
            loc = TensorLoc(
                shard_path=shard_path, offset=header["__payload_base__"] + start,
                nbytes=end - start, dtype=entry["dtype"], shape=tuple(entry["shape"]),
            )
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            role = _PROJ_ROLE[match.group("proj")]
            field = _KIND_FIELD[match.group("kind")]
            raw.setdefault((layer, expert), {})[f"{role}_{field}"] = loc

        self._experts = {
            key: ExpertLocation(layer=key[0], expert=key[1], **fields)
            for key, fields in raw.items()
        }

    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self._experts})

    def experts_per_layer(self, layer: int) -> list[int]:
        return sorted(expert for (layer_, expert) in self._experts if layer_ == layer)

    def get(self, layer: int, expert: int) -> ExpertLocation:
        return self._experts[(layer, expert)]


class ExpertReader:
    """Reads one expert's nine pieces into nine caller-supplied writable buffers, from the
    expert-major repack at ``repacked_dir`` when there is one (one read per expert), else
    the raw checkpoint through :class:`Nvfp4DiskIndex` (nine)."""

    def __init__(self, model_path: str, repacked_dir: str | None = None):
        model_path = os.path.expanduser(model_path)
        repacked_dir = os.path.expanduser(repacked_dir) if repacked_dir else None
        self._local = threading.local()
        self._fds: dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._entries: dict[tuple[int, int], tuple[int, int]] | None = None
        self._index: Nvfp4DiskIndex | None = None
        self._hint = sys.platform == "darwin"
        if repacked_dir and os.path.isdir(repacked_dir):
            with open(os.path.join(repacked_dir, "experts.index.json"), encoding="utf-8") as f:
                meta = json.load(f)
            actual = config_sha(model_path)
            if actual != meta["source_config_sha256"]:
                raise ValueError(
                    f"repack at {repacked_dir} was built from a different checkpoint than "
                    f"{model_path} (config sha256 {actual} != {meta['source_config_sha256']})"
                )
            self._entries = {
                (int(k.split(":")[0]), int(k.split(":")[1])): (e["offset"], e["size"])
                for k, e in meta["experts"].items()
            }
            self._max_record = max(size for _off, size in self._entries.values())
            self._bin_fd = self._fd(os.path.join(repacked_dir, "experts.bin"))
            self.reads_per_expert = 1
            logger.info_rank0(
                f"MoE disk tier: expert-major repack at {repacked_dir}, 1 read/expert"
            )
        else:
            self._index = Nvfp4DiskIndex(model_path)
            self.reads_per_expert = 9
            logger.info_rank0(
                f"MoE disk tier: offset index over {model_path}, 9 reads/expert "
                f"(build a repack with freetoken.checkpoint.repack_nvfp4_experts for 1)"
            )

    def _fd(self, path: str) -> int:
        """This reader's ``F_NOCACHE`` descriptor for ``path``, opened once; ``preadv`` is
        positional, so one serves every thread."""
        fd = self._fds.get(path)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    # Paired with the F_RDADVISE hints: drop this call to measure them against
                    # a warm page cache instead.
                    set_nocache_fd(fd)
                    self._fds[path] = fd
        return fd

    def hint(self, layer: int, expert: int) -> None:
        """Ask the drive to start fetching this expert's bytes now. Best-effort: the first
        ``OSError`` (EINVAL on some descriptors) turns the hint off for this reader."""
        if not self._hint:
            return
        try:
            if self._entries is None:
                for piece in self._index.get(layer, expert).all_locs():
                    fcntl.fcntl(self._fd(piece.shard_path), _F_RDADVISE,
                                struct.pack("qi", piece.offset, piece.nbytes))
            else:
                offset, size = self._entries[(layer, expert)]
                fcntl.fcntl(self._bin_fd, _F_RDADVISE, struct.pack("qi", offset, size))
        except OSError as err:
            self._hint = False
            logger.warning(f"MoE disk tier: readahead hint refused ({err}); hints off")

    def read_into(self, layer: int, expert: int, dests: list[memoryview]) -> None:
        """Fill ``dests``, nine writable byte views in :data:`PIECE_ORDER`."""
        self.hint(layer, expert)
        if self._entries is None:
            loc = self._index.get(layer, expert)
            for name, dest in zip(PIECE_ORDER, dests):
                piece = getattr(loc, name)
                got = os.preadv(self._fd(piece.shard_path), [dest], piece.offset)
                if got != piece.nbytes:
                    raise OSError(
                        f"short read for {name} of expert ({layer}, {expert}): "
                        f"{got} of {piece.nbytes} bytes"
                    )
            return
        # One aligned read of the whole record, then a copy per piece.
        offset, size = self._entries[(layer, expert)]
        scratch = getattr(self._local, "scratch", None)
        if scratch is None:
            raw = np.empty(self._max_record + _PAGE, dtype=np.uint8)
            pad = (-raw.ctypes.data) % _PAGE
            scratch = self._local.scratch = raw[pad : pad + self._max_record]
            self._local.raw = raw  # keep the base alive; the view does not own it
        got = os.preadv(self._bin_fd, [memoryview(scratch[:size])], offset)
        if got != size:
            raise OSError(f"short read for expert ({layer}, {expert}): {got} of {size} bytes")
        off = 0
        for dest in dests:
            n = len(dest)
            dest[:] = scratch[off : off + n]
            off += n

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
