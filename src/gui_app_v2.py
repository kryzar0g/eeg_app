"""Enhanced GUI with hierarchical navigation and configuration panel."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from .config import load_config

logger = logging.getLogger(__name__)


@dataclass
class GuiSelection:
    """User's mode and configuration selection."""
    mode: str
    offline_file: Optional[str] = None


class _QueueWriter:
    """Redirect stdout/stderr to queue (thread-safe GUI logging)."""
    
    def __init__(self, log_queue: queue.Queue[str]) -> None:
        self._queue = log_queue
    
    def write(self, text: str) -> None:
        self._queue.put(text)
    
    def flush(self) -> None:
        pass


class EegAppGui:
    """Enhanced EEG BCI Application GUI with hierarchical navigation."""
    
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("EEG BCI Application")
        self._root.resizable(True, True)
        self._root.minsize(900, 700)
        
        # State
        self._mode_var = tk.StringVar(value="info")  # info, record, offline, online
        self._file_var = tk.StringVar()
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._task_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Load config for info display
        try:
            self._config = load_config()
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to load config: {e}")
            self._config = None
        
        self._build_ui()
        self._on_page_change()
    
    def _build_ui(self) -> None:
        """Build the UI with hierarchical navigation."""
        style = ttk.Style(self._root)
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Section.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("Sub.TLabel", font=("Helvetica", 10), foreground="#666666")
        style.configure("Info.TLabel", font=("Courier", 9), foreground="#333333")
        
        # Main container
        main_frame = ttk.Frame(self._root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # left nav
        main_frame.columnconfigure(1, weight=1)  # content
        main_frame.rowconfigure(0, weight=1)
        
        # ── LEFT NAVIGATION ────────────────────────────────────────────────
        nav_frame = ttk.Frame(main_frame, width=200)
        nav_frame.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        nav_frame.grid_propagate(False)
        
        ttk.Label(nav_frame, text="🧠 EEG BCI", style="Title.TLabel").pack(
            side="top", padx=10, pady=10, anchor="w"
        )
        
        ttk.Separator(nav_frame, orient="horizontal").pack(
            side="top", fill="x", padx=5, pady=5
        )
        
        # Navigation buttons
        self._nav_buttons = {}
        nav_items = [
            ("info", "ℹ️  Overview"),
            ("record", "▶️  Record"),
            ("offline", "📊 Train Model"),
            ("online", "🔴 Online BCI"),
        ]
        
        for page_id, label in nav_items:
            btn = ttk.Button(
                nav_frame,
                text=label,
                command=lambda p=page_id: self._switch_page(p),
            )
            btn.pack(side="top", fill="x", padx=5, pady=2)
            self._nav_buttons[page_id] = btn
        
        ttk.Separator(nav_frame, orient="horizontal").pack(
            side="top", fill="x", padx=5, pady=10
        )
        
        # Config info
        ttk.Label(nav_frame, text="Configuration", style="Section.TLabel").pack(
            side="top", padx=10, pady=(10, 5), anchor="w"
        )
        
        if self._config:
            config_text = (
                f"Trials/class: {self._config.experiment.trials_per_class}\n"
                f"Freq bands: {self._config.features.bands}\n"
                f"Sampling: {self._config.preprocessing.sfreq} Hz\n"
                f"Classes: {len(self._config.paradigm.get('classes', {}))}"
            )
            ttk.Label(nav_frame, text=config_text, style="Info.TLabel", justify="left").pack(
                side="top", padx=10, anchor="w"
            )
        
        # ── RIGHT CONTENT AREA ─────────────────────────────────────────────
        self._content_frame = ttk.Frame(main_frame)
        self._content_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self._content_frame.columnconfigure(0, weight=1)
        self._content_frame.rowconfigure(0, weight=1)
        
        # Create pages
        self._pages = {
            "info": self._build_info_page,
            "record": self._build_record_page,
            "offline": self._build_offline_page,
            "online": self._build_online_page,
        }
        
        # Container for pages
        self._page_container = ttk.Frame(self._content_frame)
        self._page_container.grid(row=0, column=0, sticky="nsew")
        self._page_container.columnconfigure(0, weight=1)
        self._page_container.rowconfigure(0, weight=1)
        
        self._page_widgets = {}
    
    def _switch_page(self, page_id: str) -> None:
        """Switch to a different page."""
        # Remove old page
        if page_id in self._page_widgets:
            self._page_widgets[page_id].grid_remove()
        
        # Create new page if needed
        if page_id not in self._page_widgets:
            builder = self._pages[page_id]
            page = ttk.Frame(self._page_container)
            page.grid(row=0, column=0, sticky="nsew")
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
            
            builder(page)
            self._page_widgets[page_id] = page
        else:
            self._page_widgets[page_id].grid()
        
        # Highlight button
        for pid, btn in self._nav_buttons.items():
            state = "pressed" if pid == page_id else "normal"
            # Use a visual indicator
            if pid == page_id:
                btn.configure(text="→ " + self._nav_buttons[pid].cget("text").split("→ ")[-1])
            else:
                btn.configure(text=self._nav_buttons[pid].cget("text").split("→ ")[-1])
        
        self._mode_var.set(page_id)
    
    def _on_page_change(self) -> None:
        """Called when navigating pages."""
        page = self._mode_var.get()
        self._switch_page(page)
    
    # ── PAGE BUILDERS ──────────────────────────────────────────────────────
    
    def _build_info_page(self, parent: ttk.Frame) -> None:
        """Build the overview/info page."""
        ttk.Label(parent, text="EEG Motor Imagery BCI System", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )
        
        info_text = """
🧠 Overview
This is a flexible Brain-Computer Interface system for motor imagery classification.

📊 System Features:
• Supports any number of EEG channels (8, 32, 64, 128, ...)
• Flexible paradigm with N-class motor imagery (2-class, 4-class, etc.)
• Online real-time classification with trained models
• Offline analysis and model training
• Structured logging to file and console

🔧 Current Configuration:
"""
        ttk.Label(parent, text=info_text, style="Sub.TLabel", justify="left").pack(
            anchor="w", pady=10
        )
        
        if self._config:
            config_frame = ttk.LabelFrame(parent, text="Active Configuration", padding=10)
            config_frame.pack(fill="x", pady=10)
            
            config_info = f"""
Experiment:
  • Trials per class: {self._config.experiment.trials_per_class}
  • Baseline: {self._config.experiment.baseline_duration}s
  • Cue: {self._config.experiment.cue_duration}s
  • Imagery: {self._config.experiment.imagery_duration}s
  • ITI: {self._config.experiment.iti_duration}s

EEG Setup:
  • Sampling rate: {self._config.preprocessing.sfreq} Hz
  • Band-pass: {self._config.preprocessing.l_freq}-{self._config.preprocessing.h_freq} Hz
  • Notch: {self._config.preprocessing.notch_freq} Hz

Classification:
  • Algorithm: {self._config.classifier.algorithm.upper()}
  • Test split: {self._config.classifier.test_size*100:.0f}%
  • Feature bands: {self._config.features.bands}

Paradigm:
  • Classes: {len(self._config.paradigm.get('classes', {}))}
  • Available: {', '.join(self._config.paradigm.get('classes', {}).keys())}
"""
            
            ttk.Label(config_frame, text=config_info, style="Info.TLabel", justify="left").pack(
                anchor="w"
            )
    
    def _build_record_page(self, parent: ttk.Frame) -> None:
        """Build the record paradigm page."""
        ttk.Label(parent, text="▶️  Record Motor Imagery Session", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )
        
        info = ttk.Frame(parent)
        info.pack(fill="x", pady=10)
        
        ttk.Label(
            info,
            text="This mode presents visual stimuli for motor imagery tasks.\n"
                 "EEG data should be recorded using LabRecorder or similar LSL-compatible software.\n"
                 "Markers will be sent via LSL to synchronize with the visual paradigm.",
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w")
        
        # Requirements
        req_frame = ttk.LabelFrame(parent, text="Requirements", padding=10)
        req_frame.pack(fill="x", pady=10)
        
        ttk.Label(
            req_frame,
            text="✓ LabRecorder running and connected to EEG stream\n"
                 "✓ LSL marker stream listener configured\n"
                 "✓ Display with sufficient resolution for paradigm\n"
                 "✓ PsychoPy library installed",
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w")
        
        # Start button
        ttk.Button(parent, text="▶  Start Recording Session", command=self._on_start_record).pack(
            fill="x", pady=20
        )
    
    def _build_offline_page(self, parent: ttk.Frame) -> None:
        """Build the offline training page."""
        ttk.Label(parent, text="📊 Train Classification Model", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )
        
        info = ttk.Frame(parent)
        info.pack(fill="x", pady=10)
        
        ttk.Label(
            info,
            text="Load an EEG recording (EDF/BDF format) and train a classifier model.\n"
                 "Features are extracted using power spectral density in configured frequency bands.\n"
                 "The model is automatically saved for later use in online classification.",
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w")
        
        # File selection
        file_frame = ttk.LabelFrame(parent, text="EEG File Selection", padding=10)
        file_frame.pack(fill="x", pady=10)
        
        ttk.Label(file_frame, text="Select EEG file (EDF/BDF):", style="Section.TLabel").pack(
            anchor="w", pady=(0, 5)
        )
        
        file_entry_frame = ttk.Frame(file_frame)
        file_entry_frame.pack(fill="x", pady=5)
        
        ttk.Entry(file_entry_frame, textvariable=self._file_var).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(file_entry_frame, text="Browse...", command=self._browse_file).pack(
            side="left"
        )
        
        # Start button
        ttk.Button(parent, text="📊 Train Model", command=self._on_start_offline).pack(
            fill="x", pady=20
        )
    
    def _build_online_page(self, parent: ttk.Frame) -> None:
        """Build the online BCI page."""
        ttk.Label(parent, text="🔴 Real-Time Classification", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )
        
        info = ttk.Frame(parent)
        info.pack(fill="x", pady=10)
        
        ttk.Label(
            info,
            text="Connect to EEG stream and perform real-time classification using a trained model.\n"
                 "Predictions are made continuously on sliding windows of EEG data.\n"
                 "Press Ctrl+C to stop.",
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w")
        
        # Requirements
        req_frame = ttk.LabelFrame(parent, text="Requirements", padding=10)
        req_frame.pack(fill="x", pady=10)
        
        ttk.Label(
            req_frame,
            text="✓ Trained model exists (run 'Train Model' first)\n"
                 "✓ EEG stream available via LSL (LabRecorder or similar)\n"
                 "✓ Same number of channels and band configuration as training\n"
                 "✓ EEG_MODEL_PATH environment variable (optional)",
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w")
        
        # Start button
        ttk.Button(parent, text="🔴 Start Online BCI", command=self._on_start_online).pack(
            fill="x", pady=20
        )
    
    # ── EVENT HANDLERS ─────────────────────────────────────────────────────
    
    def _browse_file(self) -> None:
        """Browse for EEG file."""
        path = filedialog.askopenfilename(
            title="Select EEG file",
            filetypes=[("EEG files", "*.edf *.bdf"), ("All files", "*.*")],
        )
        if path:
            self._file_var.set(path)
    
    def _on_start_record(self) -> None:
        """Start recording mode."""
        selection = GuiSelection(mode="record")
        self._start_task(selection)
    
    def _on_start_offline(self) -> None:
        """Start offline training mode."""
        if not self._file_var.get():
            messagebox.showerror("Missing File", "Please select an EEG file first")
            return
        selection = GuiSelection(mode="offline", offline_file=self._file_var.get())
        self._start_task(selection)
    
    def _on_start_online(self) -> None:
        """Start online BCI mode."""
        selection = GuiSelection(mode="online")
        self._start_task(selection)
    
    def _start_task(self, selection: GuiSelection) -> None:
        """Start background task."""
        self._running = True
        
        # Disable nav buttons
        for btn in self._nav_buttons.values():
            btn.configure(state="disabled")
        
        # Clear queue
        while not self._log_queue.empty():
            self._log_queue.get_nowait()
        
        # Show log window
        self._show_log_window()
        
        def task() -> None:
            import traceback
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            writer = _QueueWriter(self._log_queue)
            sys.stdout = writer  # type: ignore[assignment]
            sys.stderr = writer  # type: ignore[assignment]
            try:
                _run_selection(selection)
                self._log_queue.put("\n✓ Task completed successfully\n")
            except KeyboardInterrupt:
                self._log_queue.put("\n⊘ Interrupted by user\n")
            except Exception as exc:
                self._log_queue.put(f"\n✗ Error: {type(exc).__name__}: {exc}\n")
                self._log_queue.put(traceback.format_exc())
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                self._root.after(0, self._on_task_done)
        
        self._task_thread = threading.Thread(target=task, daemon=True)
        self._task_thread.start()
    
    def _show_log_window(self) -> None:
        """Show log output window."""
        log_win = tk.Toplevel(self._root)
        log_win.title("Task Output")
        log_win.geometry("800x400")
        
        log_text = scrolledtext.ScrolledText(
            log_win,
            height=20,
            wrap="word",
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Courier", 9),
        )
        log_text.pack(fill="both", expand=True, padx=5, pady=5)
        log_text.configure(state="disabled")
        
        def poll_log() -> None:
            while True:
                try:
                    msg = self._log_queue.get_nowait()
                    log_text.configure(state="normal")
                    log_text.insert(tk.END, msg)
                    log_text.see(tk.END)
                    log_text.configure(state="disabled")
                except queue.Empty:
                    break
            
            if self._running:
                self._root.after(100, poll_log)
        
        poll_log()
    
    def _on_task_done(self) -> None:
        """Called when background task completes."""
        self._running = False
        
        # Re-enable nav buttons
        for btn in self._nav_buttons.values():
            btn.configure(state="normal")


def _run_selection(selection: GuiSelection) -> None:
    """Execute the selected mode (runs in background thread)."""
    if selection.mode == "record":
        from .lsl_acquisition import create_streams
        from .stimuli.paradigm_base import MotorImageryParadigm
        from .config import load_config
        
        config = load_config()
        streams = create_streams(config)
        paradigm = MotorImageryParadigm(config, streams.marker_outlet)
        paradigm.run()
    
    elif selection.mode == "offline":
        from .offline_analysis import run_offline_from_file
        
        if not selection.offline_file:
            raise ValueError("No EEG file selected")
        
        acc, n_epochs = run_offline_from_file(selection.offline_file)
        print(f"\n✓ Training complete: {n_epochs} epochs, accuracy: {acc:.4f}")
    
    elif selection.mode == "online":
        from .config import load_config
        from .online_bci import run_online_bci
        
        config = load_config()
        run_online_bci(config)


def run_gui() -> None:
    """Launch the enhanced GUI."""
    root = tk.Tk()
    EegAppGui(root)
    root.mainloop()
