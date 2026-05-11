from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List

try:
    from psychopy import core, event, visual
    _PSYCHOPY_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    core = event = visual = None  # type: ignore[assignment]
    _PSYCHOPY_IMPORT_ERROR = exc

from ..config import AppConfig
from ..lsl_acquisition import push_marker

logger = logging.getLogger(__name__)


@dataclass
class TrialDefinition:
    """Single trial specification."""
    label: str
    code: int


class FourDotsParadigm:
    """Motor imagery paradigm with 4 directional targets (up, down, left, right).
    
    Shows fixation point and highlights target dots. Sends LSL markers at cue onset.
    """
    
    # Default visual parameters
    DOT_RADIUS = 0.03
    DOT_RADIUS_CUE = 0.05
    DOT_COLOR_DEFAULT = (1, 1, 1)  # white
    DOT_COLOR_CUE = (1, 1, 0)  # yellow
    FIXATION_COLOR = (1, 1, 1)  # white
    BACKGROUND_COLOR = (0, 0, 0)  # black

    def __init__(self, config: AppConfig, marker_outlet) -> None:
        if _PSYCHOPY_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Record mode requires psychopy. It may not be compatible with your Python version. "
                "Use Python 3.10 or 3.11 and install dependencies from requirements.txt."
            ) from _PSYCHOPY_IMPORT_ERROR
        
        self.config = config
        self.marker_outlet = marker_outlet
        self.win = self._create_window()
        
        self.fixation = visual.TextStim(
            self.win, text="+", color=self.FIXATION_COLOR, pos=(0, 0)
        )
        
        self.dots = {
            "UP": visual.Circle(
                self.win, radius=self.DOT_RADIUS, pos=(0, 0.6),
                fillColor=self.DOT_COLOR_DEFAULT
            ),
            "DOWN": visual.Circle(
                self.win, radius=self.DOT_RADIUS, pos=(0, -0.6),
                fillColor=self.DOT_COLOR_DEFAULT
            ),
            "LEFT": visual.Circle(
                self.win, radius=self.DOT_RADIUS, pos=(-0.6, 0),
                fillColor=self.DOT_COLOR_DEFAULT
            ),
            "RIGHT": visual.Circle(
                self.win, radius=self.DOT_RADIUS, pos=(0.6, 0),
                fillColor=self.DOT_COLOR_DEFAULT
            ),
        }
        
        self.trials: List[TrialDefinition] = self._build_trials()

    def _create_window(self):
        """Create PsychoPy window with fallback for weak graphics drivers."""
        window_kwargs = dict(
            color=self.BACKGROUND_COLOR,
            units="norm",
            allowGUI=False,
            checkTiming=False,
            waitBlanking=False,
            useFBO=False,
        )
        
        try:
            return visual.Window(fullscr=True, **window_kwargs)
        except Exception as first_error:
            try:
                return visual.Window(fullscr=False, size=(1280, 720), **window_kwargs)
            except Exception as second_error:
                raise RuntimeError(
                    "Failed to create PsychoPy window. Check graphics driver, "
                    "GPU support, or try running on a local machine."
                ) from second_error

    def _build_trials(self) -> List[TrialDefinition]:
        """Build randomized trial list from configuration."""
        class_map = self.config.paradigm.get("classes", {})
        
        n_per_class = self.config.experiment.trials_per_class
        
        trials: List[TrialDefinition] = []
        for label, code in class_map.items():
            for _ in range(n_per_class):
                trials.append(TrialDefinition(label=str(label), code=int(code)))
        
        random.shuffle(trials)
        logger.info(f"Built {len(trials)} trials ({n_per_class} per class)")
        return trials

    def _draw_all_dots(self) -> None:
        """Draw all dots in current state."""
        for dot in self.dots.values():
            dot.draw()

    def run(self) -> None:
        """Run the paradigm loop with exception safety."""
        exp_cfg = self.config.experiment
        
        baseline = exp_cfg.baseline_duration
        cue_dur = exp_cfg.cue_duration
        imagery_dur = exp_cfg.imagery_duration
        iti = exp_cfg.iti_duration
        
        clock = core.Clock()
        trial_num = 0
        
        try:
            for trial in self.trials:
                # Check for ESC to stop early
                if "escape" in event.getKeys(keyList=["escape"]):
                    logger.info("Paradigm stopped by user")
                    break
                
                trial_num += 1
                logger.debug(f"Trial {trial_num}/{len(self.trials)}: {trial.label} (code {trial.code})")
                
                # Baseline phase
                clock.reset()
                while clock.getTime() < baseline:
                    self.fixation.draw()
                    self._draw_all_dots()
                    self.win.flip()
                
                # Cue phase: highlight target dot and send marker
                dot = self.dots[trial.label]
                dot.radius = self.DOT_RADIUS_CUE
                dot.fillColor = self.DOT_COLOR_CUE
                
                clock.reset()
                push_marker(self.marker_outlet, str(trial.code))
                
                while clock.getTime() < cue_dur:
                    self.fixation.draw()
                    self._draw_all_dots()
                    dot.draw()
                    self.win.flip()
                
                # Imagery phase: dim the dot back to normal
                dot.radius = self.DOT_RADIUS
                dot.fillColor = self.DOT_COLOR_DEFAULT
                
                clock.reset()
                while clock.getTime() < imagery_dur:
                    self.fixation.draw()
                    self._draw_all_dots()
                    self.win.flip()
                
                # Inter-trial interval
                clock.reset()
                while clock.getTime() < iti:
                    self.fixation.draw()
                    self._draw_all_dots()
                    self.win.flip()
            
            logger.info(f"Paradigm completed: {trial_num} trials")
        
        except Exception as e:
            logger.error(f"Paradigm error: {e}", exc_info=True)
            raise
        
        finally:
            try:
                self.win.close()
                logger.debug("PsychoPy window closed")
            except Exception as e:
                logger.warning(f"Error closing window: {e}")
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
        self.fixation.draw()
        self._draw_all_dots()
        self.win.flip()

      # Cue: zvýrazníme příslušný bod před smyčkou
      dot = self.dots[trial.label]
      dot.radius = 0.05
      dot.fillColor = "yellow"

      clock.reset()
      push_marker(self.marker_outlet, str(trial.code))
      while clock.getTime() < cue_dur:
        self.fixation.draw()
        self._draw_all_dots()
        dot.draw()
        self.win.flip()

      # Fáze imaginace – bod vrátíme do standardní velikosti/barvy
      dot.radius = 0.03
      dot.fillColor = "white"
      clock.reset()
      while clock.getTime() < imagery_dur:
        self.fixation.draw()
        self._draw_all_dots()
        self.win.flip()

      # Inter-trial interval (odpočinek)
      clock.reset()
      while clock.getTime() < iti:
        self.fixation.draw()
        self._draw_all_dots()
        self.win.flip()

    self.win.close()
