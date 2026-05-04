from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, ttk
from typing import Optional


@dataclass
class GuiSelection:
  mode: str  # "record" nebo "offline"
  offline_file: Optional[str] = None


def _run_gui_dialog() -> Optional[GuiSelection]:
  root = tk.Tk()
  root.title("eeg_app – BCI launcher")

  # Hlavní frame
  frame = ttk.Frame(root, padding=10)
  frame.grid(row=0, column=0, sticky="nsew")

  root.columnconfigure(0, weight=1)
  root.rowconfigure(0, weight=1)

  # Volba režimu
  mode_var = tk.StringVar(value="record")

  ttk.Label(frame, text="Režim:").grid(row=0, column=0, sticky="w")
  modes = [
    ("Online/record (LSL paradigma)", "record"),
    ("Offline analýza z EDF/BDF", "offline"),
  ]

  for i, (label, value) in enumerate(modes):
    ttk.Radiobutton(frame, text=label, variable=mode_var, value=value).grid(
      row=1 + i, column=0, columnspan=3, sticky="w"
    )

  # Offline: výběr souboru
  offline_file_var = tk.StringVar()

  def browse_offline_file() -> None:
    path = filedialog.askopenfilename(
      title="Vyberte EDF/BDF soubor",
      filetypes=[("EEG files", "*.bdf *.edf"), ("All files", "*.*")],
    )
    if path:
      offline_file_var.set(path)

  ttk.Label(frame, text="EEG soubor pro offline analýzu:").grid(
    row=3, column=0, columnspan=3, sticky="w", pady=(10, 0)
  )

  entry = ttk.Entry(frame, textvariable=offline_file_var, width=60)
  entry.grid(row=4, column=0, columnspan=2, sticky="we")

  ttk.Button(frame, text="Procházet…", command=browse_offline_file).grid(
    row=4, column=2, sticky="e"
  )

  frame.columnconfigure(0, weight=1)
  frame.columnconfigure(1, weight=1)

  # Tlačítka Start/Zrušit
  buttons = ttk.Frame(frame)
  buttons.grid(row=5, column=0, columnspan=3, pady=(15, 0), sticky="e")

  selection: dict[str, Optional[str]] = {"mode": None, "offline_file": None}

  def on_start() -> None:
    m = mode_var.get()
    if m == "offline" and not offline_file_var.get():
      # vyžadujeme soubor pro offline mód
      tk.messagebox.showerror(
        "Chyba",
        "V režimu 'offline' musíte vybrat EEG soubor.",
      )
      return

    selection["mode"] = m
    selection["offline_file"] = offline_file_var.get() or None
    root.destroy()

  def on_cancel() -> None:
    root.destroy()

  ttk.Button(buttons, text="Zrušit", command=on_cancel).grid(row=0, column=0)
  ttk.Button(buttons, text="Start", command=on_start).grid(row=0, column=1, padx=(5, 0))

  root.mainloop()

  if selection["mode"] is None:
    return None

  return GuiSelection(mode=selection["mode"] or "record", offline_file=selection["offline_file"])


def run_gui() -> Optional[GuiSelection]:
  """Spustí jednoduchý launcher GUI a vrátí zvolený režim a parametry.

  Funkce je oddělená od logiky samotných módů – pouze sbírá vstupy.
  """

  return _run_gui_dialog()
