"""
mlx-flash monitor — Live RAM and Layer Progress Dashboard.

This tool provides a real-time terminal UI for monitoring MLX memory usage
and Flash Mode execution progress.
"""

import curses
import json
import os
import sys
import time
from typing import Any

import psutil

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

import contextlib
import queue
import threading

STATE_FILE = "/tmp/mlx_flash_monitor.json"

class TelemetryBridge(threading.Thread):
    """Bridges the internal monitor_queue to the external STATE_FILE."""
    def __init__(self, monitor_queue: queue.Queue):
        super().__init__(daemon=True)
        self.queue = monitor_queue
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Wait for an event with timeout to check stop_event
                event = self.queue.get(timeout=1.0)
                try:
                    with open(STATE_FILE, "w") as f:
                         json.dump(event, f)
                except Exception:
                    pass
                # self.queue.task_done() -- not strictly needed for daemon thread
            except queue.Empty:
                continue
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()

def start_telemetry(config: Any):
    """Initialize the telemetry bridge for external monitors."""
    if config.monitor_queue is None:
        config.monitor_queue = queue.Queue()
    bridge = TelemetryBridge(config.monitor_queue)
    bridge.start()
    return bridge

def get_model_process():
    """Find the most likely MLX model process."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            cmd = " ".join(proc.info['cmdline'] or [])
            if "mlx_lm" in cmd or "mlx_engine" in cmd or "flash" in cmd:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def draw_bar(val: float, max_val: float, width: int) -> str:
    """Render a text progress bar."""
    if max_val <= 0:
        return "░" * width
    filled = int((val / max_val) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)

def monitor_loop(stdscr, model_name: str = "MLX Model"):
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
    
    stdscr.nodelay(True)
    
    last_layer_state = {}
    start_time = time.time()
    
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # 1. Gather Stats
        metal_active = 0.0
        metal_peak = 0.0
        if _HAS_MLX:
            try:
                metal_active = mx.get_active_memory() / 1e6
                metal_peak = mx.get_peak_memory() / 1e6
            except Exception:
                pass
            
        process = get_model_process()
        rss = 0.0
        if process:
            with contextlib.suppress(Exception):
                rss = process.memory_info().rss / 1e6
            
        # Try to read per-layer state
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE) as f:
                    last_layer_state = json.load(f)
        except Exception:
            pass
        
        # 2. Render Header
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(1, 2, "⚡ mlx-flash monitor")
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(1, 22, f"│  {model_name[:20]:20}  │  macOS ARM")
        stdscr.addstr(2, 2, "─" * (w - 4))
        
        # 3. Memory Grid
        col_w = (w - 8) // 4
        stdscr.addstr(4, 4, "Metal Active")
        stdscr.addstr(4, 4 + col_w, "Metal Peak")
        stdscr.addstr(4, 4 + col_w*2, "RSS (Python)")
        stdscr.addstr(4, 4 + col_w*3, "Page Cache")
        
        # Active
        stdscr.addstr(5, 4, f"{metal_active:6.0f} MB ")
        stdscr.addstr(5, 4 + 10, draw_bar(metal_active, 16000, 10))
        
        # Peak
        stdscr.addstr(5, 4 + col_w, f"{metal_peak:6.0f} MB ")
        stdscr.addstr(5, 4 + col_w + 10, draw_bar(metal_peak, 16000, 10))
        
        # RSS
        stdscr.addstr(5, 4 + col_w*2, f"{rss:6.0f} MB ")
        stdscr.addstr(5, 4 + col_w*2 + 10, draw_bar(rss, 16000, 10))
        
        # 4. Progress Bar
        layer = last_layer_state.get("layer", 0)
        n_layers = last_layer_state.get("n_layers", 1)
        progress_w = w - 30
        stdscr.addstr(8, 4, "Progress:")
        stdscr.addstr(8, 14, draw_bar(layer, n_layers, progress_w))
        stdscr.addstr(8, 16 + progress_w, f"layer {layer}/{n_layers}")
        
        # 5. Footer / Stats
        elapsed = time.time() - start_time
        stdscr.addstr(10, 4, f"Uptime: {elapsed:6.1f}s  │  PID: {process.pid if process else 'N/A'}")
        
        if last_layer_state:
            t_layer = last_layer_state.get("timestamp", 0)
            if time.monotonic() - t_layer < 5:
                 stdscr.attron(curses.color_pair(2))
                 stdscr.addstr(10, 40, "● LIVE")
                 stdscr.attroff(curses.color_pair(2))
            else:
                 stdscr.attron(curses.color_pair(4))
                 stdscr.addstr(10, 40, "○ STALE")
                 stdscr.attroff(curses.color_pair(4))

        stdscr.addstr(12, 2, "─" * (w - 4))
        stdscr.addstr(13, 4, f"Budget: 16.0 GB  │  Actual peak: {metal_peak / 1000:.1f} GB  │  Saving: {(16000 - metal_peak) / 1000:.1f} GB")
        
        stdscr.refresh()
        
        # Input check
        try:
            key = stdscr.getch()
            if key in (ord('q'), 27): # q or ESC
                break
        except Exception:
            pass
        
        time.sleep(0.2)

def run_monitor():
    """CLI Entry Point."""
    model_name = "Dynamic Monitor"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    curses.wrapper(monitor_loop, model_name)

if __name__ == "__main__":
    run_monitor()
