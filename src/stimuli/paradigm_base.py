"""Abstract paradigm base class and implementations for flexible EEG experiments."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from ..config import AppConfig
from ..lsl_acquisition import StreamOutlet

logger = logging.getLogger(__name__)


class Paradigm(ABC):
    """Abstract base class for EEG paradigms."""
    
    def __init__(self, config: AppConfig, marker_outlet: StreamOutlet) -> None:
        self.config = config
        self.marker_outlet = marker_outlet
    
    @abstractmethod
    def run(self) -> None:
        """Run the paradigm loop."""
        pass


class MotorImageryParadigm(Paradigm):
    """Flexible motor imagery paradigm supporting N-class experiments.
    
    Dynamically positions stimuli based on number of classes.
    Supports 2-class (left/right), 3-class (left/right/rest), 4-class (up/down/left/right), etc.
    """
    
    def __init__(self, config: AppConfig, marker_outlet: StreamOutlet) -> None:
        super().__init__(config, marker_outlet)
        
        try:
            from psychopy import core, event, visual
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Motor Imagery paradigm requires psychopy. "
                "Install with: pip install psychopy"
            ) from e
        
        self.core = core
        self.event = event
        self.visual = visual
        
        self.win = self._create_window()
        self.class_map = self._validate_and_get_classes()
        self.n_classes = len(self.class_map)
        
        logger.info(f"Motor Imagery paradigm: {self.n_classes} classes")
        
        self.fixation = self.visual.TextStim(
            self.win, text="+", color=(1, 1, 1), pos=(0, 0)
        )
        
        # Create stimulus positions
        self.stimuli = self._create_stimuli()
        self.trials = self._build_trials()

        self._instructions = self.visual.TextStim(
            self.win,
            text=(
                "PŘED MĚŘENÍM\n\n"
                "1. Sedněte si pohodlně a nehýbejte se.\n"
                "2. Sledujte zvýrazněný podnět na obrazovce.\n"
                "3. V imagery fázi si pouze představujte daný pohyb.\n"
                "4. Nemluvte a omezte pohyby očí i těla na minimum.\n"
                "5. Ukončení měření: kdykoli stiskněte ESC.\n\n"
                "Pro pokračování stiskněte MEZERNÍK."
            ),
            color="#FFFFFF",
            height=0.06,
            wrapWidth=1.5,
            pos=(0, 0),
            alignText="center",
        )

        self._instruction_title = self.visual.TextStim(
            self.win,
            text="PŘED MĚŘENÍM",
            color="#FFD400",
            height=0.09,
            pos=(0, 0.56),
            alignText="center",
        )
    
    def _create_window(self):
        """Create PsychoPy window with fallback."""
        window_kwargs = dict(
            color=(0, 0, 0),
            units="norm",
            allowGUI=False,
            checkTiming=False,
            waitBlanking=False,
            useFBO=False,
        )
        
        try:
            return self.visual.Window(fullscr=True, **window_kwargs)
        except Exception as first_error:
            try:
                return self.visual.Window(fullscr=False, size=(1280, 720), **window_kwargs)
            except Exception as second_error:
                raise RuntimeError(
                    "Failed to create PsychoPy window. Check graphics driver or GPU support."
                ) from second_error
    
    def _validate_and_get_classes(self) -> Dict[str, int]:
        """Get and validate class configuration."""
        class_map = self.config.paradigm.get("classes", {})
        if not class_map:
            raise ValueError("Configuration missing: paradigm.classes")
        return class_map
    
    def _create_stimuli(self) -> Dict[str, object]:
        """Create visual stimuli with positions based on number of classes."""
        n = self.n_classes
        stimuli = {}
        
        # Calculate positions on a circle
        import math
        radius = 0.5
        
        labels = sorted(self.class_map.keys())
        
        for i, label in enumerate(labels):
            angle = 2 * math.pi * i / n - math.pi / 2  # Start from top
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            stimuli[label] = self.visual.Circle(
                self.win,
                radius=0.03,
                pos=(x, y),
                fillColor=(1, 1, 1),  # white
            )
            logger.debug(f"Stimulus '{label}' at position ({x:.2f}, {y:.2f})")
        
        return stimuli
    
    def _build_trials(self) -> List[Dict]:
        """Build randomized trial list.

        Pokud je n_averages > 1, každý stimul je zařazen n_averages-krát za sebou
        v jednom bloku (averaging block), pak následuje ITI do dalšího bloku.
        Při trénování se tyto opakování průměrují → vyšší SNR.
        """
        import random

        exp_cfg = self.config.experiment
        n_per_class = exp_cfg.trials_per_class
        n_averages = int(getattr(exp_cfg, "n_averages", 1))

        trials = []
        if n_averages > 1:
            # Averaging mode: skupiny opakování se drží pohromadě
            for label, code in self.class_map.items():
                if n_per_class % n_averages != 0:
                    raise ValueError(
                        "experiment.trials_per_class must be divisible by experiment.n_averages "
                        "when averaging mode is enabled"
                    )
                n_blocks = n_per_class // n_averages
                for _ in range(n_blocks):
                    block = [
                        {"label": str(label), "code": int(code), "rep": r, "n_reps": n_averages}
                        for r in range(n_averages)
                    ]
                    trials.append(block)  # blok opakování pro jeden stimul
            random.shuffle(trials)
            # Rozvinout bloky do plochého seznamu
            flat = [trial for block in trials for trial in block]
            logger.info(
                f"Built {len(flat)} trials ({n_per_class // n_averages} blocks × "
                f"{n_averages} reps per class, n_averages={n_averages})"
            )
            return flat
        else:
            for label, code in self.class_map.items():
                for _ in range(n_per_class):
                    trials.append({"label": str(label), "code": int(code), "rep": 0, "n_reps": 1})
            random.shuffle(trials)
            logger.info(f"Built {len(trials)} trials ({n_per_class} per class)")
            return trials
    
    def _draw_all_stimuli(self) -> None:
        """Draw all stimuli."""
        for stimulus in self.stimuli.values():
            stimulus.draw()

    def _show_instructions(self) -> None:
        """Show instructions before the first trial."""
        while True:
            if "escape" in self.event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Paradigm aborted before start")
            if "space" in self.event.getKeys(keyList=["space"]):
                return

            self._instruction_title.draw()
            self._instructions.draw()
            self.win.flip()
            self.core.wait(0.01)
    
    def run(self) -> None:
        """Run the motor imagery paradigm."""
        exp_cfg = self.config.experiment
        
        baseline = exp_cfg.baseline_duration
        cue_dur = exp_cfg.cue_duration
        imagery_dur = exp_cfg.imagery_duration
        iti = exp_cfg.iti_duration
        
        clock = self.core.Clock()
        trial_num = 0
        
        try:
            self._show_instructions()

            for trial in self.trials:
                if "escape" in self.event.getKeys(keyList=["escape"]):
                    logger.info("Paradigm stopped by user")
                    break
                
                trial_num += 1
                label = trial["label"]
                code = trial["code"]
                logger.debug(f"Trial {trial_num}/{len(self.trials)}: {label} (code {code})")
                
                # Baseline phase
                clock.reset()
                while clock.getTime() < baseline:
                    self.fixation.draw()
                    self._draw_all_stimuli()
                    self.win.flip()
                
                # Cue phase: highlight target stimulus
                stimulus = self.stimuli[label]
                stimulus.fillColor = (1, 1, 0)  # yellow
                stimulus.radius = 0.05
                
                clock.reset()
                from ..lsl_acquisition import push_marker
                push_marker(self.marker_outlet, str(code))
                
                while clock.getTime() < cue_dur:
                    self.fixation.draw()
                    self._draw_all_stimuli()
                    stimulus.draw()
                    self.win.flip()
                
                # Imagery phase: reset stimulus
                stimulus.fillColor = (1, 1, 1)  # white
                stimulus.radius = 0.03
                
                clock.reset()
                while clock.getTime() < imagery_dur:
                    self.fixation.draw()
                    self._draw_all_stimuli()
                    self.win.flip()
                
                # Inter-trial interval
                clock.reset()
                while clock.getTime() < iti:
                    self.fixation.draw()
                    self._draw_all_stimuli()
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
