from __future__ import annotations

import ctypes
import mmap
import os
import sys
from typing import Any

"""
page_cache.py — macOS unified page cache management via madvise().

On Apple Silicon macOS the CPU and GPU share the same physical memory (unified
memory architecture).  When model weights are memory-mapped from disk the OS
keeps recently accessed pages in this shared pool.  Crucially:

  * Metal can directly read from mmap'd pages — no extra VRAM copy.
  * madvise(MADV_WILLNEED) prefetches pages from SSD into RAM proactively.
  * madvise(MADV_FREE) releases pages the OS may reclaim without a syscall
    from any reader — the fastest eviction path on macOS ≥14.
  * madvise(MADV_DONTNEED) is advisory only on macOS (unlike Linux where it
    clears pages immediately), but still signals LRU priority.

We obtain the virtual address of a Python mmap region via ctypes, which is
safe as long as the mmap object is kept alive.

This module is macOS-only.  On other platforms all functions are no-ops so
that the rest of the code runs in CI (Linux / Windows).
"""

_IS_MACOS = sys.platform == "darwin"

# ── macOS madvise constants (from <sys/mman.h>) ────────────────────────────
MADV_NORMAL     = 0
MADV_RANDOM     = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED   = 3
MADV_DONTNEED   = 4
MADV_FREE       = 5   # macOS ≥ 10.9 (most efficient release)

# ── load libSystem ─────────────────────────────────────────────────────────
_libc: ctypes.CDLL | None = None

def get_libc() -> ctypes.CDLL | None:
    global _libc
    if _libc is None and _IS_MACOS:
        try:
            _libc = ctypes.CDLL("libSystem.B.dylib", use_errno=True)
            _libc.madvise.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _libc.madvise.restype = ctypes.c_int
        except OSError:
            _libc = None
    return _libc


def _mmap_base_addr(mm: mmap.mmap) -> int:
    """
    Return the virtual address of the first byte of a Python mmap region.

    We use ctypes.from_buffer to obtain a c_char that aliases the mmap's
    memory; ctypes.addressof then gives us the raw pointer.
    """
    c_buf = (ctypes.c_char * 1).from_buffer(mm)
    return ctypes.addressof(c_buf)


def madvise_range(mm: mmap.mmap, offset: int, length: int, advice: int) -> bool:
    """
    Call madvise on [base+offset, base+offset+length) of *mm*.

    Returns True on success, False if madvise is unavailable or fails.
    The call is always page-aligned (rounds down offset, rounds up length).
    """
    libc = get_libc()
    if libc is None:
        return False

    page_size = os.sysconf("SC_PAGE_SIZE") if _IS_MACOS else 4096
    # Align down offset to page boundary
    aligned_offset = (offset // page_size) * page_size
    extra = offset - aligned_offset
    aligned_length = ((length + extra + page_size - 1) // page_size) * page_size

    try:
        base = _mmap_base_addr(mm)
        addr = base + aligned_offset
        ret = libc.madvise(ctypes.c_void_p(addr),
                           ctypes.c_size_t(aligned_length),
                           ctypes.c_int(advice))
        return ret == 0
    except Exception:
        return False


def prefetch(mm: mmap.mmap, offset: int, length: int) -> bool:
    """Hint the OS to pull pages into the page cache (MADV_WILLNEED)."""
    return madvise_range(mm, offset, length, MADV_WILLNEED)


def release(mm: mmap.mmap, offset: int, length: int,
            strategy: str = "free") -> bool:
    """
    Hint the OS that these pages are no longer needed.

    strategy:
        "free"     → MADV_FREE (macOS preferred; pages reused on pressure)
        "dontneed" → MADV_DONTNEED (advisory LRU deprioritisation)
        "none"     → no-op
    """
    if strategy == "none":
        return True
    advice = MADV_FREE if strategy == "free" else MADV_DONTNEED
    return madvise_range(mm, offset, length, advice)


def _get_free_pages() -> int:
    """Returns the number of free pages via vm_stat (macOS)."""
    if not _IS_MACOS:
        return 0
    try:
        import subprocess
        out = subprocess.check_output(["vm_stat"], encoding="utf-8")
        for line in out.splitlines():
            if "Pages free:" in line:
                return int(line.split(":")[1].strip().strip("."))
    except Exception:
        pass
    return 0


def release_and_verify(mm: mmap.mmap, offset: int, length: int, strategy: str = "free") -> int:
    """Release pages and return how many bytes were freed (best-effort).
    Uses vm_stat parsing on macOS to check page-free events."""
    before = _get_free_pages()
    success = release(mm, offset, length, strategy)
    if not success:
        return 0
    after = _get_free_pages()
    page_size = os.sysconf("SC_PAGE_SIZE") if _IS_MACOS else 4096
    return max(0, (after - before) * page_size)


def set_sequential(mm: mmap.mmap, offset: int, length: int) -> bool:
    """Mark a region as sequentially accessed so the OS can read-ahead."""
    return madvise_range(mm, offset, length, MADV_SEQUENTIAL)


def _madvise_array(arr: Any, advice: int) -> bool:
    """Internal helper to call madvise on the memory backing an MLX array."""
    libc = get_libc()
    if libc is None:
        return False
    
    try:
        # We use the numpy interface to get the pointer without a copy.
        # This works for MLX arrays as they implement the buffer protocol.
        import numpy as np
        view = np.array(arr, copy=False)
        ptr, _ = view.__array_interface__['data']
        nbytes = view.nbytes
        
        page_size = os.sysconf("SC_PAGE_SIZE") if _IS_MACOS else 4096
        # Align down ptr to page boundary
        aligned_ptr = (ptr // page_size) * page_size
        extra = ptr - aligned_ptr
        aligned_length = ((nbytes + extra + page_size - 1) // page_size) * page_size
        
        ret = libc.madvise(ctypes.c_void_p(aligned_ptr),
                         ctypes.c_size_t(aligned_length),
                         ctypes.c_int(advice))
        return ret == 0
    except Exception:
        return False


def prefetch_array(arr: Any) -> bool:
    """Issue MADV_WILLNEED on the mmap pages backing an MLX array."""
    return _madvise_array(arr, MADV_WILLNEED)


def release_array(arr: Any, strategy: str = "free") -> bool:
    """Issue MADV_FREE or MADV_DONTNEED on the memory backing an MLX array."""
    if strategy == "none":
        return True
    advice = MADV_FREE if strategy == "free" else MADV_DONTNEED
    return _madvise_array(arr, advice)


class PageCacheRegion:
    """
    Context manager that prefetches a mmap region on entry and
    optionally releases it on exit.

    Usage::

        with PageCacheRegion(mm, offset, size, evict_on_exit=True):
            data = mm[offset:offset+size]
            ...  # GPU computation on this data
        # pages released here
    """

    def __init__(
        self,
        mm: mmap.mmap,
        offset: int,
        size: int,
        evict_on_exit: bool = True,
        strategy: str = "free",
    ) -> None:
        self.mm = mm
        self.offset = offset
        self.size = size
        self.evict_on_exit = evict_on_exit
        self.strategy = strategy

    def __enter__(self) -> PageCacheRegion:
        prefetch(self.mm, self.offset, self.size)
        return self

    def __exit__(self, *_) -> None:
        if self.evict_on_exit:
            release(self.mm, self.offset, self.size, self.strategy)
