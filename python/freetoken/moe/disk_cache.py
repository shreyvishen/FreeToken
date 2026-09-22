"""MoE expert cache for unified memory: slot banks on the accelerator, source on the SSD."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from freetoken.kernel import backend
from freetoken.moe.expert_reader import ExpertReader
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.utils import init_logger

logger = init_logger(__name__)

MAX_THREADS = 8  # preads in flight at most
DEFAULT_STAGE_ROWS = 32  # staging rows; prefill loops in chunks, decode needs top_k
_NEVER = np.iinfo(np.int64).max

_PF_MAX_RANK = 2  # leading predictor ranks worth a read; a wasted one costs its full time
# Demand-read bytes per layer per decode token above which prefetch pays: past it the step is
# stalled on the drive, below it the reads fit inside the layer that issues them.
_PF_AUTO_BYTES = 4 * 2**20

PIN_FRAC = 0.10  # share of each layer's experts pinned; 0.05 and 0.15 both lose to it
# Decode tokens counted before the pin set is ranked, then re-ranked at every doubling.
PIN_RANK_TOKENS = 64


class DiskMoeCache(OffloadMoeCache):
    """``OffloadMoeCache`` whose source tier is the SSD instead of pinned host banks:
    construct it like the base cache, then call :meth:`set_disk_source`."""

    def set_disk_source(
        self, reader: ExpertReader, hidden_size: int, intermediate_size: int
    ) -> None:
        if self.quant_format != "nvfp4":
            raise NotImplementedError(
                f"the disk tier reads native NVFP4 rows; quant_format={self.quant_format!r} "
                f"has no expert-major on-disk layout"
            )
        from freetoken.moe.legacy_format import nvfp4_bank_specs

        # alphas_for_slots reads the base id_of_slot, which this subclass never writes.
        assert self.gate_up_alpha is None, (
            "DiskMoeCache does not support gate_up_alpha (marlin/b12x); its id_of_slot is "
            "never kept in sync with the numpy LRU"
        )
        self.reader = reader
        self._pool = ThreadPoolExecutor(max_workers=MAX_THREADS, thread_name_prefix="ft-moe-disk")
        self._hinter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ft-moe-hint")
        h, i = hidden_size, intermediate_size
        specs = nvfp4_bank_specs(self.num_experts, h, i)

        self.bank_caches = {
            name: torch.empty((self.cache_size, *shape[1:]), dtype=dtype, device=self.device)
            for name, (shape, dtype) in specs.items()
        }
        self.banks = [([], self.bank_caches[name]) for name in self.bank_schema]

        # One arena of ``stage_rows`` records, each holding the six banks' rows back to back,
        # so a fetched expert crosses in one copy rather than six.
        self.stage_rows = DEFAULT_STAGE_ROWS
        self.intermediate_size = i
        self._layout: list[tuple[str, int, int, tuple[int, ...], torch.dtype]] = []
        offset = 0
        for name in self.bank_schema:
            shape, dtype = specs[name]
            row_bytes = int(np.prod(shape[1:])) * torch.empty((), dtype=dtype).element_size()
            self._layout.append((name, offset, row_bytes, tuple(shape[1:]), dtype))
            offset += row_bytes
        self.bytes_per_expert = offset
        # The Metal scatter moves 16-byte words, so every bank row and record must be whole words.
        self._scatter = self.device.type == "mps" and all(
            off % 16 == 0 and rb % 16 == 0 for _n, off, rb, _s, _d in self._layout
        ) and offset % 16 == 0
        (self._host_arena, self._dev_arena, self._arena_np, self._row_view, self._glob,
         self._dests, self._idx) = self._make_arena(self.stage_rows)
        # Two prefetch arenas, alternating so neither is rewritten before the routing D2H two
        # layers later drains the copy that read it. Allocated on first use.
        self._pf_banks: list[tuple] = []
        self._pf_bank = 0
        self._pf_in_flight: tuple | None = None
        self._pf: dict[int, tuple[int, object]] = {}
        self._pf_layer = -1
        self._pred_host: torch.Tensor | None = None
        self._pred_np: np.ndarray | None = None
        self._pred_src: torch.Tensor | None = None
        self._pred_layer = -1
        self.prefetch_pays = False  # false until the tier has measured itself
        self._auto_tokens = 0
        self._auto_next = PIN_RANK_TOKENS
        self._auto_at = (0, 0)  # (tokens, misses) at the last decision

        self._slot_of = np.full((self.num_layers, self.num_experts), -1, dtype=np.int32)
        self._id_of_slot = np.full((self.cache_size,), -1, dtype=np.int32)
        self._usage = np.zeros((self.cache_size,), dtype=np.int64)
        self._step = 0
        self._window = self.num_experts  # <= cache_size: the base class validates that
        self._bias = np.zeros((self.cache_size,), dtype=np.int64)
        self._bias[: self._window] = 1
        self.prefill_on_demand = True  # only the routed experts, not the whole layer
        self._slot_buf: torch.Tensor | None = None  # host source for the slot-id H2D
        self._slot_buf_np: np.ndarray | None = None
        self._pending: list[tuple[int, int]] = []  # (expert, slot)
        self._pending_layer: int | None = None
        self.n_misses = 0
        # The pin set: each layer's top PIN_FRAC of experts by routing count, held against
        # eviction while the LRU fights over the rest.
        self._pinned = np.zeros((self.cache_size,), dtype=bool)
        self._pin_expert = None  # bool[num_layers, num_experts], None until first ranked
        self._route_counts = np.zeros((self.num_layers, self.num_experts), dtype=np.int64)
        self._pin_tokens = 0
        self._pin_next = PIN_RANK_TOKENS  # rank here, then at every doubling

    def _rank_pins(self) -> None:
        """Freeze the pin set from the decode routing counted so far."""
        pinned = self._pinned
        k = min(max(int(PIN_FRAC * self.num_experts), 1), self.num_experts)
        # Half the cache at most: PIN_FRAC of a 512-expert layer over 48 layers can name more
        # slots than the tier holds, and an all-pinned cache re-ranks unbiased on every miss.
        k = min(k, max(1, self.cache_size // (2 * self.num_layers)))
        head = np.argpartition(-self._route_counts, k - 1, axis=1)[:, :k]
        self._pin_expert = np.zeros((self.num_layers, self.num_experts), dtype=bool)
        np.put_along_axis(self._pin_expert, head, True, axis=1)
        live = np.flatnonzero(self._id_of_slot >= 0)
        pinned[:] = False
        if live.size:
            pinned[live] = self._pin_expert.reshape(-1)[self._id_of_slot[live].astype(np.int64)]
        logger.info_rank0(
            f"MoE disk tier: pinned the top {k} of {self.num_experts} experts per layer "
            f"({k * self.num_layers} of {self.cache_size} slots), "
            f"ranked over {self._pin_tokens} decode tokens"
        )

    def _make_arena(self, rows: int) -> tuple:
        """A staging arena of ``rows`` expert records: host buffer (``None`` on Metal, where it is
        read in place), device tensor, numpy alias, the fp16 global banks, the f32 globals, the
        ``preadv`` destinations and, on Metal, the scatter's indices."""
        nbytes = rows * self.bytes_per_expert
        if self.device.type == "mps":
            # pinned-on-the-shared-heap, so a pread lands where the GPU reads it (kernel/backend.py)
            host = None
            dev, arena_np = backend.shared_host_arena(nbytes)
        else:
            host = torch.empty(nbytes, dtype=torch.uint8)
            dev = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            arena_np = host.numpy()
        # For the two writes preadv cannot do itself: every record's fp16 global bank as one
        # strided array over the arena, so a batch of rows is three numpy writes.
        row_view = tuple(
            np.ndarray((rows, *shape), np.float16, arena_np, off, (self.bytes_per_expert, 2))
            for name, off, _rb, shape, _dtype in self._layout
            if name in ("gate_up_global", "down_global")
        )
        glob = np.empty((rows, 3), dtype=np.float32)
        span = {name: (off, rb) for name, off, rb, _shape, _dtype in self._layout}

        def mv(row: int, name: str, lo: float = 0.0, hi: float = 1.0) -> memoryview:
            off, size = span[name]
            base = row * self.bytes_per_expert + off
            return memoryview(arena_np[base + int(size * lo) : base + int(size * hi)])

        # One expert's nine preadv destinations: six ranges of this row's record, and the three
        # f32 globals into ``glob`` (the bank keeps them per row, fp16).
        dests = [
            [mv(r, "gate_up_packed", 0, 0.5), mv(r, "gate_up_scale", 0, 0.5),
             memoryview(glob[r, 0:1]).cast("B"),
             mv(r, "gate_up_packed", 0.5, 1), mv(r, "gate_up_scale", 0.5, 1),
             memoryview(glob[r, 1:2]).cast("B"),
             mv(r, "down_packed"), mv(r, "down_scale"), memoryview(glob[r, 2:3]).cast("B")]
            for r in range(rows)
        ]
        # Written by numpy, read by the scatter in place: an H2D of them would wait on the GPU.
        idx = None
        if self._scatter:
            idx_dev, idx_np = backend.shared_host_arena(2 * rows * 4)
            idx = (idx_dev.view(torch.int32), idx_np.view(np.int32))
        return host, dev, arena_np, row_view, glob, dests, idx

    def _read_rows(self, layer_id: int, experts: list[int]) -> None:
        """Fill staging rows ``0..len(experts)`` from disk, ``MAX_THREADS`` reads in flight."""
        jobs = [(e, self._dests[r]) for r, e in enumerate(experts)]
        if len(jobs) == 1:
            self.reader.read_into(layer_id, jobs[0][0], jobs[0][1])
        else:
            list(self._pool.map(lambda j: self.reader.read_into(layer_id, j[0], j[1]), jobs))
        self._apply_globals(self._row_view, self._glob, range(len(experts)))

    def _apply_globals(self, row_view: tuple, glob: np.ndarray, rows) -> None:
        """Broadcast the f32 scales a read landed in ``glob`` into ``row_view``'s rows."""
        rows = list(rows)
        if not rows:
            return
        i = self.intermediate_size
        gate_up, down = row_view
        g = glob[rows].astype(np.float16)
        gate_up[rows, :i] = g[:, 0:1]
        gate_up[rows, i:] = g[:, 1:2]
        down[rows] = g[:, 2:3]

    # ``non_blocking=True`` is safe despite ``backend.NON_BLOCKING``: the arena is rewritten only
    # after the next layer's routing D2H drains the queue.
    def _flush_rows(
        self, slots: list[int], n: int, *, non_blocking: bool = False, arena: tuple | None = None,
        rows: list[int] | None = None,
    ) -> None:
        """Write staging rows ``0..n`` (or ``rows``) into ``slots``: on Metal one scatter launch
        per bank, else one plain row copy per (bank, row) -- not ``index_put_``, which on MPS
        costs several times the rows it moves."""
        host, dev, idx = ((self._host_arena, self._dev_arena, self._idx) if arena is None
                          else (arena[0], arena[1], arena[6]))
        nbytes = n * self.bytes_per_expert
        if host is not None:
            dev[:nbytes].copy_(host[:nbytes], non_blocking=non_blocking or backend.NON_BLOCKING)
        staged = list(range(n)) if rows is None else rows
        if self._scatter:
            from freetoken.kernel.metal.elementwise import scatter_rows

            k = len(slots)
            idx[1][:k] = slots
            idx[1][k : 2 * k] = staged
            for name, off, _row_bytes, _shape, _dtype in self._layout:
                scatter_rows(self.bank_caches[name], dev, idx[0], k, self.bytes_per_expert,
                             off)
        else:
            records = dev[:nbytes].view(n, self.bytes_per_expert)
            for name, off, row_bytes, shape, dtype in self._layout:
                bank = self.bank_caches[name]
                for i, r in enumerate(staged):
                    row = records[r, off : off + row_bytes].view(dtype).reshape(shape)
                    bank[slots[i]].copy_(row)
        if host is None and not non_blocking:
            # The copies read the pinned arena lazily; the caller made no promise.
            backend.synchronize()

    @backend.host_step
    def ensure_experts(
        self, layer_id: int, expert_ids: torch.Tensor, *, prefill: bool = False,
    ) -> None:
        """Host-side LRU: rewrite ``expert_ids`` to slot ids and stage this step's misses."""
        ids = expert_ids.detach().reshape(-1).to("cpu").numpy().astype(np.int64)
        self._step += 1
        slots = self._slot_of[layer_id]
        wanted = np.unique(ids)
        cur = slots[wanted]
        lo, hi = 0, self.cache_size
        if prefill and wanted.size * self.num_layers > hi - lo:
            lo, hi = 0, self._window
        if wanted.size > hi - lo:
            # More experts than evictable slots: stream the layer into the window, where
            # position == expert id and the given ids are already slot ids.
            self.materialize_layer(layer_id)
            return
        resident = wanted[cur >= 0]
        self._usage[slots[resident]] = self._step  # pin this step's hits

        if not prefill:
            np.add.at(self._route_counts[layer_id], ids, 1)
            if layer_id == self.num_layers - 1:
                self._pin_tokens += 1
                if self._pin_tokens >= self._pin_next:
                    self._rank_pins()
                    self._pin_next *= 2
        if not prefill and layer_id == self.num_layers - 1:
            self._auto_tokens += 1
            if self._auto_tokens >= self._auto_next:
                tokens0, misses0 = self._auto_at
                tokens = self._auto_tokens - tokens0
                per_layer = (
                    (self.n_misses - misses0)
                    * self.bytes_per_expert / tokens / self.num_layers
                )
                self.prefetch_pays = per_layer > _PF_AUTO_BYTES
                self._auto_at = (self._auto_tokens, self.n_misses)
                self._auto_next *= 2
                logger.info_rank0(
                    f"MoE disk tier: prefetch {'ON' if self.prefetch_pays else 'off'} over "
                    f"the last {tokens} decode tokens ({per_layer / 2**20:.2f} MiB of demand "
                    f"reads per layer per token, threshold {_PF_AUTO_BYTES / 2**20:.2f})"
                )

        pending = []
        misses = wanted[cur < 0]
        # Every rank is a copy (the bias add makes it one), so a victim is retired by hand.
        bias = self._bias[lo:hi]
        if not misses.size:
            rank = None
        else:
            rank = np.where(self._pinned[lo:hi], _NEVER, self._usage[lo:hi] + bias)
        for expert in misses:
            j = int(np.argmin(rank))
            if rank[j] == _NEVER:
                # All pinned or filled this step; the tier must still make progress.
                rank = self._usage[lo:hi] + bias
                j = int(np.argmin(rank))
            victim = lo + j
            evicted = self._id_of_slot[victim]
            if evicted >= 0:
                self._slot_of.reshape(-1)[evicted] = -1
            self._id_of_slot[victim] = layer_id * self.num_experts + int(expert)
            self._usage[victim] = self._step
            if self._pin_expert is not None:
                self._pinned[victim] = not prefill and bool(self._pin_expert[layer_id, int(expert)])
            rank[j] = _NEVER
            slots[expert] = victim
            pending.append((int(expert), victim))
            if not prefill:
                # Start the drive now: a decode step waits on the predicted reads first.
                self.reader.hint(layer_id, int(expert))
        if prefill and pending:
            # Off this thread: a hint that must queue I/O blocks, and the read pool starts now.
            experts = [e for e, _ in pending]
            self._hinter.submit(lambda: [self.reader.hint(layer_id, e) for e in experts])
        self.n_misses += len(pending)

        self._pending = pending
        self._pending_layer = layer_id
        # The slot ids in a buffer this object owns and keeps: the copy reads it lazily.
        values = slots[ids]
        buf = self._slot_buf
        if buf is None or buf.numel() < values.size or buf.dtype != expert_ids.dtype:
            buf = self._slot_buf = torch.empty(values.size, dtype=expert_ids.dtype)
            self._slot_buf_np = buf.numpy()
        self._slot_buf_np[: values.size] = values
        expert_ids.copy_(buf[: values.size].view(expert_ids.shape), non_blocking=True)

    def predict_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Take the ids layer L+1 will route to, predicted on layer L's pre-mixer stream."""
        ids = expert_ids.detach().reshape(-1)
        host = self._pred_host
        if host is None or host.numel() != ids.numel() or host.dtype != ids.dtype:
            host = self._pred_host = torch.empty(ids.numel(), dtype=ids.dtype)
            self._pred_np = host.numpy()
        host.copy_(ids, non_blocking=True)
        self._pred_src = ids  # the copy is lazy: do not let the source be freed
        self._pred_layer = layer_id

    def issue_prediction(self, layer_id: int) -> None:
        """Issue the reads predicted a layer ago; the previous layer's drain landed the ids."""
        if self._pred_layer != layer_id:
            return
        self._pred_layer = -1
        self._pred_src = None
        ids = self._pred_np.astype(np.int64)
        self._drop_prefetch()
        # Predictor order: topk sorts descending, so position is confidence rank.
        slots = self._slot_of[layer_id]
        want: list[int] = []
        seen: set[int] = set()
        for rank, expert in enumerate(ids):
            expert = int(expert)
            if expert in seen:
                continue
            seen.add(expert)
            if rank >= _PF_MAX_RANK:
                continue
            if slots[expert] < 0 and len(want) < self.stage_rows:
                want.append(expert)
        if not want:
            return
        if not self._pf_banks:
            self._pf_banks = [self._make_arena(self.stage_rows) for _ in range(2)]
        bank = self._pf_banks[self._pf_bank]
        self._pf_bank ^= 1
        self._pf_in_flight = bank
        self._pf_layer = layer_id
        dests = bank[5]
        self._pf = {
            e: (r, self._pool.submit(self.reader.read_into, layer_id, e, dests[r]))
            for r, e in enumerate(want)
        }

    def _drop_prefetch(self) -> None:
        """Retire what the last prediction left unclaimed, waiting on its in-flight reads."""
        for _row, fut in self._pf.values():
            fut.result()
        self._pf = {}
        self._pf_layer = -1

    def kick_queue(self) -> None:
        """Commit what the host encoded so the GPU starts now: MPS runs no queued kernel until
        something commits the buffer, and the only commit here is the next layer's routing D2H."""
        backend.flush()

    @backend.host_step
    def copy_missing(self) -> None:
        pending, self._pending = self._pending, []
        if not pending:
            self._drop_prefetch()
            return
        layer_id = self._pending_layer
        if self._pf and self._pf_layer == layer_id:
            bank = self._pf_in_flight
            rows: list[int] = []
            slots: list[int] = []
            rest: list[tuple[int, int]] = []
            for expert, slot in pending:
                hit = self._pf.pop(expert, None)
                if hit is None:
                    rest.append((expert, slot))
                    continue
                hit[1].result()
                rows.append(hit[0])
                slots.append(slot)
            self._drop_prefetch()
            if rows:
                self._apply_globals(bank[3], bank[4], rows)
                # Async: the banks alternate, so this one is untouched for two layers.
                self._flush_rows(slots, max(rows) + 1, arena=bank, rows=rows, non_blocking=True)
            pending = rest
            if not pending:
                return
        else:
            self._drop_prefetch()
        # One chunk means no rewrite before the next layer's routing D2H drains the queue.
        single = len(pending) <= self.stage_rows
        for start in range(0, len(pending), self.stage_rows):
            chunk = pending[start : start + self.stage_rows]
            self._read_rows(layer_id, [e for e, _ in chunk])
            self._flush_rows([s for _, s in chunk], len(chunk), non_blocking=single)
        self.kick_queue()

    def materialize_layer(self, layer_id: int) -> None:
        """Whole-layer prefill fetch into slots ``[0, num_experts)``, position == expert id, in
        expert order, which is file order in the repack."""
        self._pending = []
        self._pending_layer = layer_id
        for start in range(0, self.num_experts, self.stage_rows):
            chunk = list(range(start, min(start + self.stage_rows, self.num_experts)))
            self._read_rows(layer_id, chunk)
            self._flush_rows(chunk, len(chunk))
        window = self._id_of_slot[: self.num_experts]
        self._slot_of.reshape(-1)[window[window >= 0].astype(np.int64)] = -1
        base_id = layer_id * self.num_experts
        # An expert already in a decode slot keeps that mapping: the window holds a copy anyway.
        cur = self._slot_of[layer_id]
        pos = np.arange(self.num_experts, dtype=np.int32)
        fresh = cur < 0
        window[:] = np.where(fresh, base_id + pos, -1)
        cur[fresh] = pos[fresh]
        self._usage[: self.num_experts] = self._step
        self._pinned[: self.num_experts] = False  # whatever decode had pinned there is gone

    def validate_rebuild(self, cache_size: int) -> None:
        super().validate_rebuild(cache_size)
        if getattr(self, "bank_caches", None):
            # The numpy LRU and the arenas are sized once; the engine rejects before teardown.
            raise ValueError(
                f"the SSD tier cannot resize in place; restart with --moe-cache-size {cache_size}"
            )

    def reset(self) -> None:
        self._slot_of.fill(-1)
        self._id_of_slot.fill(-1)
        self._usage.fill(0)
        self._step = 0
        self._pending = []
        self._drop_prefetch()
        self._pinned.fill(False)
        # The routing counters too: a decode-plan capture runs dummy tokens through the tier.
        self._pin_expert = None
        self._route_counts.fill(0)
        self._pin_tokens, self._pin_next = 0, PIN_RANK_TOKENS
        self._auto_tokens, self._auto_next, self._auto_at = 0, PIN_RANK_TOKENS, (0, 0)
        self.prefetch_pays = False
        self.n_misses = 0

    def close(self) -> None:
        self._drop_prefetch()
        self._pool.shutdown(wait=True)
        self._hinter.shutdown(wait=True)
        self.reader.close()
