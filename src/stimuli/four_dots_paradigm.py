from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from psychopy import core, event, visual

from ..config import AppConfig
from ..lsl_acquisition import push_marker


@dataclass
class TrialDefinition:
  label: str
  code: int


class FourDotsParadigm:
  """Jednoduché MI paradigma se 4 body (nahoře, dole, vlevo, vpravo).

  Zobrazuje fixační bod a zvýrazňuje jeden ze 4 bodů dle definovaných tříd.
  Při onsetu cue odesílá LSL marker s kódem třídy.
  """

  def __init__(self, config: AppConfig, marker_outlet) -> None:
    self.config = config
    self.marker_outlet = marker_outlet

    self.win = visual.Window(fullscr=True, color=(0, 0, 0), units="norm")

    self.fixation = visual.TextStim(self.win, text="+", color=(1, 1, 1), pos=(0, 0))

    # Pozice čtyř bodů v normovaných souřadnicích
    self.dots: Dict[str, visual.Circle] = {
      "UP": visual.Circle(self.win, radius=0.03, pos=(0, 0.6), fillColor="white"),
      "DOWN": visual.Circle(self.win, radius=0.03, pos=(0, -0.6), fillColor="white"),
      "LEFT": visual.Circle(self.win, radius=0.03, pos=(-0.6, 0), fillColor="white"),
      "RIGHT": visual.Circle(self.win, radius=0.03, pos=(0.6, 0), fillColor="white"),
    }

    self.trials: List[TrialDefinition] = self._build_trials()

  def _build_trials(self) -> List[TrialDefinition]:
    mapping = self.config.paradigm.get("classes", {})
    trials: List[TrialDefinition] = []
    n_per_class = int(self.config.experiment.get("trials_per_class", 40))

    for label, code in mapping.items():
      for _ in range(n_per_class):
        trials.append(TrialDefinition(label=str(label), code=int(code)))

    random.shuffle(trials)
    return trials

  def _draw_all_dots(self) -> None:
    for dot in self.dots.values():
      dot.draw()

  def run(self) -> None:
    exp_cfg = self.config.experiment

    baseline = float(exp_cfg.get("baseline_duration", 2.0))
    cue_dur = float(exp_cfg.get("cue_duration", 1.0))
    imagery_dur = float(exp_cfg.get("imagery_duration", 4.0))
    iti = float(exp_cfg.get("iti_duration", 2.0))

    clock = core.Clock()

    for trial in self.trials:
      # Umožnit ukončení klávesou ESC
      if "escape" in event.getKeys(keyList=["escape"]):
        break

      # Baseline s fixačním bodem
      clock.reset()
      while clock.getTime() < baseline:
        self.win.flip(clearBuffer=True)
        self.fixation.draw()
        self._draw_all_dots()

      # Cue: zvýraznění bodu odpovídajícího třídy
      clock.reset()
      push_marker(self.marker_outlet, str(trial.code))
      while clock.getTime() < cue_dur:
        self.win.flip(clearBuffer=True)
        self.fixation.draw()
        self._draw_all_dots()
        # zvýrazníme příslušný bod (větší radius / jiná barva)
        dot = self.dots[trial.label]
        dot.radius = 0.05
        dot.fillColor = "yellow"
        dot.draw()

      # Fáze imaginace – bod vrátíme do standardní velikosti/barvy
      dot.radius = 0.03
      dot.fillColor = "white"
      clock.reset()
      while clock.getTime() < imagery_dur:
        self.win.flip(clearBuffer=True)
        self.fixation.draw()
        self._draw_all_dots()

      # Inter-trial interval (odpočinek)
      clock.reset()
      while clock.getTime() < iti:
        self.win.flip(clearBuffer=True)
        self.fixation.draw()
        self._draw_all_dots()

    self.win.close()
