from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional


@dataclass
class GuiSelection:
  mode: str  # "record", "offline" nebo "online"
  offline_file: Optional[str] = None


class _QueueWriter:
  """Přesměruje stdout/stderr do fronty (thread-safe výpis do GUI)."""

  def __init__(self, log_queue: "queue.Queue[str]") -> None:
    self._queue = log_queue

  def write(self, text: str) -> None:
    self._queue.put(text)

  def flush(self) -> None:
    pass


_MODE_INFO = [
  (
    "record",
    "▶  Record – zaznamenat tréninkové sezení",
    "Zobrazí MI paradigma se 4 směry a odesílá LSL markery. Data nahrávejte LabRecorderem.",
  ),
  (
    "offline",
    "📊  Offline analýza – natrénovat model",
    "Načte uložený EDF/BDF soubor, extrahuje příznaky a natrénuje klasifikátor.",
  ),
  (
    "online",
    "🔴  Online BCI – klasifikace v reálném čase",
    "Připojí se k EEG streamu a klasifikuje motorickou imaginaci natrénovaným modelem.",
  ),
]


class EegAppGui:
  """Hlavní okno EEG BCI aplikace."""

  def __init__(self, root: tk.Tk) -> None:
    self._root = root
    self._root.title("EEG BCI Aplikace")
    self._root.resizable(True, True)
    self._root.minsize(640, 600)

    self._mode_var = tk.StringVar(value="record")
    self._file_var = tk.StringVar()
    self._log_queue: "queue.Queue[str]" = queue.Queue()
    self._task_thread: Optional[threading.Thread] = None
    self._running = False

    self._build_ui()
    self._on_mode_change()

  # ── UI construction ───────────────────────────────────────────────────────

  def _build_ui(self) -> None:
    style = ttk.Style(self._root)
    style.configure("Section.TLabel", font=("Helvetica", 10, "bold"))
    style.configure("Title.TLabel", font=("Helvetica", 14, "bold"))
    style.configure("Sub.TLabel", font=("Helvetica", 9), foreground="#555555")

    outer = ttk.Frame(self._root, padding=15)
    outer.grid(row=0, column=0, sticky="nsew")
    self._root.columnconfigure(0, weight=1)
    self._root.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)

    row = 0

    # ── Nadpis ────────────────────────────────────────────────────────────
    ttk.Label(outer, text="🧠  EEG BCI Aplikace", style="Title.TLabel").grid(
      row=row, column=0, sticky="w"
    )
    row += 1
    ttk.Label(
      outer,
      text="Systém pro rozpoznávání motorické imaginace z EEG signálů",
      style="Sub.TLabel",
    ).grid(row=row, column=0, sticky="w", pady=(0, 10))
    row += 1

    ttk.Separator(outer, orient="horizontal").grid(
      row=row, column=0, sticky="ew", pady=(0, 10)
    )
    row += 1

    # ── Volba režimu ─────────────────────────────────────────────────────
    ttk.Label(outer, text="Vyberte režim:", style="Section.TLabel").grid(
      row=row, column=0, sticky="w"
    )
    row += 1

    mode_frame = ttk.Frame(outer)
    mode_frame.grid(row=row, column=0, sticky="ew", pady=(4, 8))
    mode_frame.columnconfigure(0, weight=1)
    row += 1

    for i, (val, label, desc) in enumerate(_MODE_INFO):
      ttk.Radiobutton(
        mode_frame,
        text=label,
        variable=self._mode_var,
        value=val,
        command=self._on_mode_change,
      ).grid(row=i * 2, column=0, sticky="w")
      ttk.Label(mode_frame, text=desc, style="Sub.TLabel").grid(
        row=i * 2 + 1, column=0, sticky="w", padx=(24, 0), pady=(0, 6)
      )

    ttk.Separator(outer, orient="horizontal").grid(
      row=row, column=0, sticky="ew", pady=(0, 10)
    )
    row += 1

    # ── EEG soubor (offline) ──────────────────────────────────────────────
    self._file_frame = ttk.Frame(outer)
    self._file_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
    self._file_frame.columnconfigure(1, weight=1)
    row += 1

    ttk.Label(self._file_frame, text="EEG soubor (pro offline analýzu):", style="Section.TLabel").grid(
      row=0, column=0, columnspan=3, sticky="w", pady=(0, 4)
    )
    ttk.Entry(self._file_frame, textvariable=self._file_var).grid(
      row=1, column=0, columnspan=2, sticky="ew", padx=(0, 4)
    )
    ttk.Button(self._file_frame, text="Procházet…", command=self._browse_file).grid(
      row=1, column=2, sticky="e"
    )

    ttk.Separator(outer, orient="horizontal").grid(
      row=row, column=0, sticky="ew", pady=(0, 10)
    )
    row += 1

    # ── Log ───────────────────────────────────────────────────────────────
    ttk.Label(outer, text="Výstup / log:", style="Section.TLabel").grid(
      row=row, column=0, sticky="w"
    )
    row += 1

    self._log_text = scrolledtext.ScrolledText(
      outer,
      height=10,
      state="disabled",
      wrap="word",
      bg="#1e1e1e",
      fg="#d4d4d4",
      font=("Courier", 9),
      insertbackground="white",
    )
    self._log_text.grid(row=row, column=0, sticky="nsew", pady=(4, 6))
    outer.rowconfigure(row, weight=1)
    row += 1

    # ── Progress bar ──────────────────────────────────────────────────────
    self._progress = ttk.Progressbar(outer, mode="indeterminate")
    self._progress.grid(row=row, column=0, sticky="ew", pady=(0, 10))
    row += 1

    ttk.Separator(outer, orient="horizontal").grid(
      row=row, column=0, sticky="ew", pady=(0, 10)
    )
    row += 1

    # ── Tlačítka ──────────────────────────────────────────────────────────
    btn_frame = ttk.Frame(outer)
    btn_frame.grid(row=row, column=0, sticky="e")

    self._start_btn = ttk.Button(
      btn_frame, text="▶  Spustit", command=self._on_start, width=14
    )
    self._start_btn.grid(row=0, column=0)

  # ── Interakce ─────────────────────────────────────────────────────────────

  def _on_mode_change(self) -> None:
    if self._mode_var.get() == "offline":
      self._file_frame.grid()
    else:
      self._file_frame.grid_remove()

  def _browse_file(self) -> None:
    path = filedialog.askopenfilename(
      title="Vyberte EDF/BDF soubor",
      filetypes=[("EEG soubory", "*.bdf *.edf"), ("Všechny soubory", "*.*")],
    )
    if path:
      self._file_var.set(path)

  def _log(self, text: str) -> None:
    self._log_text.configure(state="normal")
    self._log_text.insert(tk.END, text)
    self._log_text.see(tk.END)
    self._log_text.configure(state="disabled")

  def _poll_log(self) -> None:
    while True:
      try:
        msg = self._log_queue.get_nowait()
        self._log(msg)
      except queue.Empty:
        break
    if self._running:
      self._root.after(100, self._poll_log)

  def _on_start(self) -> None:
    mode = self._mode_var.get()
    if mode == "offline" and not self._file_var.get():
      messagebox.showerror("Chyba", "V režimu 'offline' musíte vybrat EEG soubor.")
      return

    selection = GuiSelection(mode=mode, offline_file=self._file_var.get() or None)
    self._start_task(selection)

  def _start_task(self, selection: GuiSelection) -> None:
    self._running = True
    self._start_btn.configure(state="disabled")
    self._progress.start(10)
    # Vyprázdnit frontu z předchozího běhu
    while not self._log_queue.empty():
      self._log_queue.get_nowait()

    self._log(f"--- Spouštím režim: {selection.mode} ---\n")
    self._root.after(100, self._poll_log)

    def task() -> None:
      import traceback
      old_stdout = sys.stdout
      old_stderr = sys.stderr
      writer = _QueueWriter(self._log_queue)
      sys.stdout = writer  # type: ignore[assignment]
      sys.stderr = writer  # type: ignore[assignment]
      try:
        _run_selection(selection)
        self._log_queue.put("\n--- Completed ✓ ---\n")
      except KeyboardInterrupt:
        self._log_queue.put("\n--- Interrupted by user ---\n")
      except Exception as exc:
        self._log_queue.put(f"\n[ERROR] {type(exc).__name__}: {exc}\n")
        self._log_queue.put(traceback.format_exc())
      finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        self._root.after(0, self._on_task_done)

    self._task_thread = threading.Thread(target=task, daemon=True)
    self._task_thread.start()

  def _on_task_done(self) -> None:
    self._running = False
    self._progress.stop()
    self._start_btn.configure(state="normal")
    # Vyprázdnit zbývající zprávy z fronty
    self._poll_log()


# ── Spuštění úlohy (voláno z vlákna na pozadí) ───────────────────────────────


def _run_selection(selection: GuiSelection) -> None:
  """Spustí zvolenou akci (voláno z vlákna na pozadí)."""

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
      raise ValueError("Není vybrán EEG soubor pro offline analýzu.")

    acc, n_epochs = run_offline_from_file(selection.offline_file)
    print(f"Počet epoch: {n_epochs}, přesnost na testovacích datech: {acc:.3f}")

  elif selection.mode == "online":
    from .config import load_config
    from .online_bci import run_online_bci

    config = load_config()
    run_online_bci(config)


# ── Veřejné API ───────────────────────────────────────────────────────────────


def run_gui() -> Optional[GuiSelection]:
  """Spustí grafické rozhraní aplikace.

  Úlohy jsou spouštěny přímo uvnitř GUI ve vlákně na pozadí.
  Funkce blokuje, dokud uživatel nezavře okno, a poté vrátí None.
  """

  root = tk.Tk()
  EegAppGui(root)
  root.mainloop()
  return None
