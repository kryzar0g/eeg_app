from __future__ import annotations

import argparse
from enum import Enum

from .config import load_config
from .gui_app import run_gui
from .offline_analysis import run_offline_from_file


class Mode(str, Enum):
  RECORD = "record"   # stimulace + LSL markery
  OFFLINE = "offline"  # offline analýza z EDF/BDF
  ONLINE = "online"   # online BCI smyčka s natrénovaným modelem
  GUI = "gui"        # grafický launcher


def run_record_mode() -> None:
  """Spustí MI paradigma se 4 body a odesílá LSL markery.

  V této základní verzi se EEG data typicky nahrávají pomocí LabRecorderu
  z EEG a marker streamů.
  """

  # Lazy import – psychopy není potřeba pro ostatní režimy
  from .lsl_acquisition import create_streams
  from .stimuli.four_dots_paradigm import FourDotsParadigm

  config = load_config()
  streams = create_streams(config)

  paradigm = FourDotsParadigm(config, streams.marker_outlet)
  paradigm.run()


def run_offline_mode(file_path: str) -> None:
  """Spustí offline pipeline nad zadaným EDF/BDF souborem."""

  acc, n_epochs = run_offline_from_file(file_path)
  print(f"Offline analýza dokončena, počet epoch: {n_epochs}, test accuracy: {acc:.3f}")


def run_online_mode() -> None:
  """Spustí online BCI smyčku s natrénovaným modelem."""

  from .online_bci import run_online_bci

  config = load_config()
  run_online_bci(config)


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
  elif selection.mode == Mode.ONLINE.value:
    run_online_mode()


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
  elif mode is Mode.ONLINE:
    run_online_mode()
  elif mode is Mode.GUI:
    run_gui_mode()
  else:
    raise NotImplementedError(f"Režim '{mode}' zatím není implementován.")


if __name__ == "__main__":  # pragma: no cover
  main()
