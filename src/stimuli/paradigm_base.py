"""Motor imagery paradigm: arrow cue + separated imagery window.

Casovani jednoho trialu (BCI Competition IV 2a styl):

    |-- baseline --|-- cue (sipka) --|-- imagery (jen kriz) --|-- ITI --|
                                     ^
                                     marker (LSL) = zacatek imaginace

Sipka se zobrazi v CUE fazi (ucastnik se muze divat a mrkat).
Pak ZMIZI a zacne IMAGERY faze s pouhym fixacnim krizem -> nahrane
EEG okno neobsahuje ocni artefakty z presunu pohledu.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List

from ..config import AppConfig
from ..lsl_acquisition import StreamOutlet

logger = logging.getLogger(__name__)


# Smer sipky -> orientace ve stupnich (ShapeStim ori, po smeru hod. rucicek).
# Sipka je definovana smerem NAHORU (ori=0).
_ARROW_ORI = {
    "up": 0.0,
    "right": 90.0,
    "down": 180.0,
    "left": 270.0,
}

# Vrcholy sipky smerujici nahoru (norm jednotky, vystredeno v pocatku)
_ARROW_VERTICES = [
    (0.00, 0.32),    # spicka
    (-0.20, 0.08),   # leve krídlo
    (-0.08, 0.08),
    (-0.08, -0.30),  # leva spodni cast tela
    (0.08, -0.30),
    (0.08, 0.08),
    (0.20, 0.08),    # prave krídlo
]


class Paradigm(ABC):
    """Abstract base class for EEG paradigms."""

    def __init__(self, config: AppConfig, marker_outlet: StreamOutlet,
                 marker_queue=None) -> None:
        self.config = config
        self.marker_outlet = marker_outlet
        # IPC fronta do interniho recorderu (zaloha k LSL marker streamu)
        self.marker_queue = marker_queue

    @abstractmethod
    def run(self) -> None:
        ...

    def _emit_marker(self, code: int) -> None:
        """Posle marker pres LSL (externi nastroje) i IPC frontu (interni recorder).

        Do IPC fronty prilozi local_clock() timestamp - presny cas markeru
        ve stejnem hodinovem domenu jako EEG (po time_correction). Tim je
        marker v zaznamu zarovnan na presny EEG vzorek (synchronizace).
        """
        from ..lsl_acquisition import push_marker
        push_marker(self.marker_outlet, str(code))
        if self.marker_queue is not None:
            try:
                from pylsl import local_clock
                self.marker_queue.put_nowait((int(code), float(local_clock())))
            except Exception:
                pass


class MotorImageryParadigm(Paradigm):
    """4-tridni motoricka imaginace s arrow cue a oddelenou imaginaci."""

    def __init__(self, config: AppConfig, marker_outlet: StreamOutlet,
                 marker_queue=None) -> None:
        super().__init__(config, marker_outlet, marker_queue)

        try:
            from psychopy import core, event, visual
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Motor Imagery paradigm requires psychopy. Install: pip install psychopy"
            ) from e

        self.core = core
        self.event = event
        self.visual = visual

        self.win = self._create_window()
        self.class_map = self._validate_and_get_classes()
        self.n_classes = len(self.class_map)
        self.cues = self.config.paradigm.get("cues", {})
        self.separate = bool(self.config.paradigm.get("separate_cue_imagery", True))

        logger.info(
            "Motor Imagery: %d trid, separate_cue_imagery=%s",
            self.n_classes, self.separate,
        )

        # ── Vizualni prvky ────────────────────────────────────────────
        self.fixation = self.visual.TextStim(
            self.win, text="+", color="white", height=0.18, pos=(0, 0), bold=True
        )

        # Sipka posunuta NAHORU, aby se neprekryvala s textem uprostred
        self.arrow = self.visual.ShapeStim(
            self.win,
            vertices=_ARROW_VERTICES,
            fillColor="#4DA3FF",
            lineColor="#4DA3FF",
            pos=(0, 0.4),
            ori=0.0,
        )

        # Popisek smeru UPROSTRED obrazovky (hlavni prvek behem cue)
        self.cue_label = self.visual.TextStim(
            self.win, text="", color="white", height=0.16, pos=(0, 0), bold=True
        )

        # Text "klid" take UPROSTRED (aby clovek nebyl zmaten)
        self.rest_text = self.visual.TextStim(
            self.win, text="", color="#888888", height=0.10, pos=(0, 0)
        )

        self.trials = self._build_trials()

        self._instructions = self.visual.TextStim(
            self.win,
            text=(
                "MOTORICKA IMAGINACE\n\n"
                "1. Sedte v klidu, divejte se na kriz uprostred.\n"
                "2. Objevi se SIPKA = ktery pohyb si predstavit:\n"
                "      <-  leva ruka        ->  prava ruka\n"
                "      dolu obe nohy       nahoru jazyk\n"
                "3. Kdyz sipka ZMIZI, zacnete si pohyb PREDSTAVOVAT\n"
                "   (pouze predstava, NEHYBEJTE se).\n"
                "4. Behem predstavy se nedivejte jinam a nemrkejte.\n"
                "5. ESC kdykoli ukonci mereni.\n\n"
                "Pro start stisknete MEZERNIK."
            ),
            color="white",
            height=0.06,
            wrapWidth=1.6,
            pos=(0, 0.05),
            alignText="center",
        )
        self._instruction_title = self.visual.TextStim(
            self.win, text="PRED MERENIM", color="#FFD400", height=0.09, pos=(0, 0.7), bold=True
        )

    # ── Setup ─────────────────────────────────────────────────────────

    def _create_window(self):
        kwargs = dict(color="#101216", units="norm", allowGUI=False,
                      checkTiming=False, waitBlanking=False, useFBO=False)
        try:
            return self.visual.Window(fullscr=True, **kwargs)
        except Exception:
            try:
                return self.visual.Window(fullscr=False, size=(1280, 720), **kwargs)
            except Exception as e:
                raise RuntimeError(
                    "Nepodarilo se vytvorit PsychoPy okno (graficky driver/GPU)."
                ) from e

    def _validate_and_get_classes(self) -> Dict[str, int]:
        class_map = self.config.paradigm.get("classes", {})
        if not class_map:
            raise ValueError("Configuration missing: paradigm.classes")
        return class_map

    def _build_trials(self) -> List[Dict]:
        exp_cfg = self.config.experiment
        n_per_class = exp_cfg.trials_per_class
        n_averages = int(getattr(exp_cfg, "n_averages", 1))

        trials: List[Dict] = []
        if n_averages > 1:
            blocks = []
            for label, code in self.class_map.items():
                n_blocks = max(1, n_per_class // n_averages)
                for _ in range(n_blocks):
                    blocks.append([
                        {"label": str(label), "code": int(code)}
                        for _ in range(n_averages)
                    ])
            random.shuffle(blocks)
            trials = [t for block in blocks for t in block]
        else:
            for label, code in self.class_map.items():
                for _ in range(n_per_class):
                    trials.append({"label": str(label), "code": int(code)})
            random.shuffle(trials)

        logger.info("Built %d trials (%d per class, %d classes)",
                    len(trials), n_per_class, self.n_classes)
        return trials

    def _arrow_ori_for(self, label: str) -> float:
        cue = self.cues.get(label, {})
        arrow_dir = cue.get("arrow", "up")
        return _ARROW_ORI.get(str(arrow_dir).lower(), 0.0)

    def _label_text_for(self, label: str) -> str:
        cue = self.cues.get(label, {})
        return str(cue.get("label", label))

    # ── Smycky faze ────────────────────────────────────────────────────

    def _check_escape(self) -> bool:
        return "escape" in self.event.getKeys(keyList=["escape"])

    def _show_instructions(self) -> None:
        while True:
            keys = self.event.getKeys(keyList=["escape", "space"])
            if "escape" in keys:
                raise KeyboardInterrupt("Paradigm aborted before start")
            if "space" in keys:
                return
            self._instruction_title.draw()
            self._instructions.draw()
            self.win.flip()
            self.core.wait(0.01)

    def _phase_fixation(self, clock, duration: float) -> bool:
        """Jen fixacni kriz po dobu duration. Vraci False pri ESC."""
        clock.reset()
        while clock.getTime() < duration:
            if self._check_escape():
                return False
            self.fixation.draw()
            self.win.flip()
        return True

    def run(self) -> None:
        exp = self.config.experiment
        baseline = exp.baseline_duration
        cue_dur = exp.cue_duration
        imagery_dur = exp.imagery_duration
        iti = exp.iti_duration

        clock = self.core.Clock()
        trial_num = 0

        try:
            self._show_instructions()

            for trial in self.trials:
                if self._check_escape():
                    logger.info("Paradigm stopped by user")
                    break

                trial_num += 1
                label = trial["label"]
                code = trial["code"]
                logger.debug("Trial %d/%d: %s (code %d)",
                             trial_num, len(self.trials), label, code)

                # ── 1. Baseline: jen fixacni kriz ────────────────────
                if not self._phase_fixation(clock, baseline):
                    break

                # ── 2. Cue: sipka nahore + popisek UPROSTRED (BEZ markeru) ──
                # Fixacni kriz se nekresli, aby nebyl za textem uprostred.
                self.arrow.ori = self._arrow_ori_for(label)
                self.cue_label.text = self._label_text_for(label)
                clock.reset()
                while clock.getTime() < cue_dur:
                    if self._check_escape():
                        raise KeyboardInterrupt
                    self.arrow.draw()
                    self.cue_label.draw()
                    self.win.flip()

                # ── 3. Imagery: sipka ZMIZI, marker ZDE ──────────────
                # Marker se posila prave ted = zacatek ciste imaginace.
                if self.separate:
                    self._emit_marker(code)
                    clock.reset()
                    while clock.getTime() < imagery_dur:
                        if self._check_escape():
                            raise KeyboardInterrupt
                        self.fixation.draw()   # pouze kriz, zadna sipka
                        self.win.flip()
                else:
                    # Legacy rezim bez oddeleni - marker take na zacatku imaginace
                    self._emit_marker(code)
                    clock.reset()
                    while clock.getTime() < imagery_dur:
                        if self._check_escape():
                            raise KeyboardInterrupt
                        self.fixation.draw()
                        self.win.flip()

                # ── 4. ITI: klid (marker kod 0 = KLID interval) ──────
                self.rest_text.text = "klid"
                self._emit_marker(0)   # klidovy interval do anotaci
                clock.reset()
                while clock.getTime() < iti:
                    if self._check_escape():
                        raise KeyboardInterrupt
                    self.rest_text.draw()
                    self.win.flip()

            logger.info("Paradigm completed: %d trials", trial_num)

        except KeyboardInterrupt:
            logger.info("Paradigm preruseno uzivatelem (ESC)")
        except Exception as e:
            logger.error("Paradigm error: %s", e, exc_info=True)
            raise
        finally:
            try:
                self.win.close()
            except Exception:
                pass
