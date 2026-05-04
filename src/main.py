from __future__ import annotations

import argparse
from enum import Enum

from .config import load_config
from .gui_app import run_gui
from .lsl_acquisition import create_streams
from .offline_analysis import run_offline_from_file
from .stimuli.four_dots_paradigm import FourDotsParadigm


class Mode(str, Enum):
  RECORD = "record"   # stimulace + LSL markery
  OFFLINE = "offline"  # offline analýza z EDF/BDF
  GUI = "gui"        # grafický launcher


def run_record_mode() -> None:
  """Spustí MI paradigma se 4 body a odesílá LSL markery.

  V této základní verzi se EEG data typicky nahrávají pomocí LabRecorderu
  z EEG a marker streamů.
  """

  config = load_config()
  streams = create_streams(config)

  paradigm = FourDotsParadigm(config, streams.marker_outlet)
  paradigm.run()


def run_offline_mode(file_path: str) -> None:
  """Spustí offline pipeline nad zadaným EDF/BDF souborem."""

  acc, n_epochs = run_offline_from_file(file_path)
  print(f"Offline analýza dokončena, počet epoch: {n_epochs}, test accuracy: {acc:.3f}")


def run_gui_mode() -> None:
  """Spustí jednoduché GUI pro výběr režimu a souboru."""

  selection = run_gui()
  if selection is None:
    return

  if selection.mode == Mode.RECORD.value:
    run_record_mode()
  elif selection.mode == Mode.OFFLINE.value:
    if not selection.offline_file:
      raise SystemExit("V režimu 'offline' musíte vybrat EEG soubor.")
    run_offline_mode(selection.offline_file)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="eeg_app – MI BCI aplikace")
  parser.add_argument(
    "mode",
    type=str,
    choices=[m.value for m in Mode],
    help="Režim běhu aplikace",
  )
  parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="Cesta k EDF/BDF souboru pro offline analýzu (režim 'offline')",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  mode = Mode(args.mode)

  if mode is Mode.RECORD:
    run_record_mode()
  elif mode is Mode.OFFLINE:
    if not args.file:
      raise SystemExit("V režimu 'offline' musíte zadat cestu k souboru pomocí --file.")
    run_offline_mode(args.file)
  elif mode is Mode.GUI:
    run_gui_mode()
  else:
    raise NotImplementedError(f"Režim '{mode}' zatím není implementován.")


if __name__ == "__main__":  # pragma: no cover
  main()
