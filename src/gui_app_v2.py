"""Enhanced GUI with hierarchical navigation and configuration panel."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from multiprocessing import Process
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from .config import load_config
from .patient_profiles import PatientProfile, create_profile, list_profiles

logger = logging.getLogger(__name__)


def _has_psychopy() -> bool:
    try:
        import psychopy  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class GuiSelection:
    """User's mode and configuration selection."""
    mode: str
    offline_file: Optional[str] = None
    patient_profile_id: Optional[str] = None
    patient_profile_name: Optional[str] = None
    # In-memory config z GUI – obsahuje zmeny z Network EEG zalozky
    # (napr. vybrany stream). Pokud None, pouzije se load_config() ze souboru.
    config: Optional[object] = None


def _start_paradigm_proc(pid: Optional[str], pname: Optional[str]) -> None:
    """Launch the recording paradigm inside a spawned process.

    OPRAVA: Vytváří pouze marker outlet (ne EEG inlet).
    EEG inlet není pro záznam potřeba – data nahrává interní EegRecorder
    v hlavním procesu, nebo LabRecorder externě.
    """
    from .lsl_acquisition import create_marker_outlet
    from .stimuli.paradigm_base import MotorImageryParadigm

    cfg = load_config()
    if pname:
        print(f"Patient: {pname} (id={pid})")
    marker_outlet = create_marker_outlet(cfg)
    paradigm = MotorImageryParadigm(cfg, marker_outlet)
    paradigm.run()


class _QueueWriter:
    """Redirect stdout/stderr to queue (thread-safe GUI logging)."""
    
    def __init__(self, log_queue: queue.Queue[str]) -> None:
        self._queue = log_queue
    
    def write(self, text: str) -> None:
        self._queue.put(text)
    
    def flush(self) -> None:
        pass


class PatientProfileDialog:
    """Modal dialog for selecting an existing patient or creating a new one."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._dialog = tk.Toplevel(root)
        self._dialog.title("Patient Profile")
        self._dialog.resizable(False, False)
        self._dialog.transient(root)
        self._dialog.grab_set()

        self.result: Optional[PatientProfile] = None
        self._profiles = list_profiles()

        container = ttk.Frame(self._dialog, padding=15)
        container.grid(row=0, column=0, sticky="nsew")
        self._dialog.columnconfigure(0, weight=1)
        self._dialog.rowconfigure(0, weight=1)

        ttk.Label(container, text="Patient profile", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            container,
            text="Choose an existing patient profile or create a new one before recording.",
            style="Sub.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(container, text="Existing profiles", style="Section.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self._existing_var = tk.StringVar()
        self._existing_combo = ttk.Combobox(
            container,
            textvariable=self._existing_var,
            state="readonly",
            values=[profile.display_name for profile in self._profiles],
            width=42,
        )
        self._existing_combo.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        if self._profiles:
            self._existing_combo.current(0)

        ttk.Separator(container, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=10
        )

        ttk.Label(container, text="Create new profile", style="Section.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w"
        )

        self._first_name_var = tk.StringVar()
        self._last_name_var = tk.StringVar()
        self._dob_var = tk.StringVar()
        self._sex_var = tk.StringVar()
        self._notes_var = tk.StringVar()

        form = ttk.Frame(container)
        form.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        form.columnconfigure(1, weight=1)

        fields = [
            ("First name", self._first_name_var),
            ("Last name", self._last_name_var),
            ("Date of birth (DD.MM.YYYY)", self._dob_var),
            ("Sex", self._sex_var),
            ("Notes", self._notes_var),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(form, text=label + ":").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(form, textvariable=var, width=40).grid(row=row, column=1, sticky="ew", pady=2)

        button_row = ttk.Frame(container)
        button_row.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(button_row, text="Use selected", command=self._use_selected).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Save new", command=self._save_new).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Cancel", command=self._cancel).grid(row=0, column=2)

        self._dialog.protocol("WM_DELETE_WINDOW", self._cancel)
        self._dialog.bind("<Return>", lambda _event: self._use_selected())
        self._dialog.bind("<Escape>", lambda _event: self._cancel())

    def show(self) -> Optional[PatientProfile]:
        self._root.wait_window(self._dialog)
        return self.result

    def _use_selected(self) -> None:
        if not self._profiles:
            messagebox.showerror("No profiles", "Create a new patient profile first.", parent=self._dialog)
            return

        index = self._existing_combo.current()
        if index < 0 or index >= len(self._profiles):
            messagebox.showerror("Selection required", "Please select an existing profile.", parent=self._dialog)
            return

        self.result = self._profiles[index]
        self._dialog.destroy()

    def _save_new(self) -> None:
        first_name = self._first_name_var.get().strip()
        last_name = self._last_name_var.get().strip()
        date_of_birth = self._dob_var.get().strip()
        sex = self._sex_var.get().strip()
        notes = self._notes_var.get().strip()

        if not first_name or not last_name or not date_of_birth or not sex:
            messagebox.showerror(
                "Missing data",
                "Please fill in first name, last name, date of birth and sex.",
                parent=self._dialog,
            )
            return

        self.result = create_profile(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            sex=sex,
            notes=notes,
        )
        self._dialog.destroy()

    def _cancel(self) -> None:
        self.result = None
        self._dialog.destroy()


class RecordingInstructionsDialog:
    """High-contrast modal dialog shown before recording starts."""

    def __init__(self, root: tk.Tk, patient_name: str) -> None:
        self._root = root
        self._dialog = tk.Toplevel(root)
        self._dialog.title("Recording instructions")
        self._dialog.configure(bg="#000000")
        self._dialog.resizable(False, False)
        self._dialog.transient(root)
        self._dialog.grab_set()

        self.result = False

        container = tk.Frame(self._dialog, bg="#000000", padx=18, pady=18)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Before recording",
            bg="#000000",
            fg="#FFD400",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        body = (
            f"Patient: {patient_name}\n\n"
            "1. Sit still and look at the screen.\n"
            "2. Follow the highlighted cue.\n"
            "3. During the imagery phase, imagine only the indicated movement.\n"
            "4. Do not speak and minimize eye and body movement.\n"
            "5. Press ESC any time to stop the recording.\n\n"
            "Continue only when you are ready."
        )
        tk.Label(
            container,
            text=body,
            bg="#000000",
            fg="#FFFFFF",
            font=("Segoe UI", 12),
            justify="left",
            wraplength=540,
        ).pack(anchor="w", pady=(0, 18))

        button_row = tk.Frame(container, bg="#000000")
        button_row.pack(anchor="e")
        tk.Button(
            button_row,
            text="Continue",
            command=self._accept,
            bg="#FFD400",
            fg="#000000",
            activebackground="#FFE766",
            activeforeground="#000000",
            relief="flat",
            padx=14,
            pady=6,
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            button_row,
            text="Cancel",
            command=self._cancel,
            bg="#222222",
            fg="#FFFFFF",
            activebackground="#333333",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=14,
            pady=6,
        ).pack(side="left")

        self._dialog.protocol("WM_DELETE_WINDOW", self._cancel)
        self._dialog.bind("<Return>", lambda _event: self._accept())
        self._dialog.bind("<Escape>", lambda _event: self._cancel())

    def show(self) -> bool:
        self._root.wait_window(self._dialog)
        return self.result

    def _accept(self) -> None:
        self.result = True
        self._dialog.destroy()

    def _cancel(self) -> None:
        self.result = False
        self._dialog.destroy()


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
        self._profile_label_var = tk.StringVar(value="No patient selected")
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._task_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_selection: Optional[GuiSelection] = None
        self._patient_profile: Optional[PatientProfile] = None
        
        # Load config for info display
        try:
            self._config = load_config()
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to load config: {e}")
            self._config = None
        
        self._build_ui()
        self._on_page_change()
        self._root.after(150, self._prompt_for_patient_profile)
    
    # ── Moderni paleta ────────────────────────────────────────────────────
    BG       = "#F4F6F9"   # svetle pozadi obsahu
    SIDEBAR  = "#1E2128"   # tmavy sidebar
    SIDEBAR2 = "#272B34"   # tmavsi prvek v sidebaru
    ACCENT   = "#4DA3FF"   # modry akcent (ladi se sipkou)
    ACCENT_D = "#3B82D6"   # tmavsi akcent (hover)
    TEXT     = "#2C3038"   # tmavy text
    MUTED    = "#8A8F98"   # tlumeny text
    CARD     = "#FFFFFF"   # karty

    def _build_ui(self) -> None:
        """Build the UI with hierarchical navigation."""
        FONT = "Segoe UI"
        self._root.configure(bg=self.BG)

        style = ttk.Style(self._root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Zakladni prvky
        style.configure(".", font=(FONT, 10))
        style.configure("TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.CARD)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("Title.TLabel", font=(FONT, 18, "bold"), foreground=self.TEXT, background=self.BG)
        style.configure("Section.TLabel", font=(FONT, 12, "bold"), foreground=self.TEXT, background=self.BG)
        style.configure("Sub.TLabel", font=(FONT, 10), foreground=self.MUTED, background=self.BG)
        style.configure("Info.TLabel", font=("Consolas", 9), foreground="#4A4F58", background=self.BG)
        style.configure("TLabelframe", background=self.BG, bordercolor="#D8DCE2", relief="solid")
        style.configure("TLabelframe.Label", background=self.BG, foreground=self.TEXT, font=(FONT, 10, "bold"))
        style.configure("TEntry", fieldbackground=self.CARD, bordercolor="#C8CDD5")
        style.configure("TCombobox", fieldbackground=self.CARD)
        style.configure("TSeparator", background="#D8DCE2")

        # Tlacitka - flat, akcentni
        style.configure("TButton", font=(FONT, 10), padding=(12, 7),
                        background=self.CARD, foreground=self.TEXT,
                        bordercolor="#C8CDD5", relief="flat")
        style.map("TButton",
                  background=[("active", "#E9EDF3"), ("pressed", "#DDE3EB")])

        # Primarni (akcentni) tlacitko
        style.configure("Accent.TButton", font=(FONT, 11, "bold"),
                        padding=(14, 10), background=self.ACCENT,
                        foreground="white", relief="flat", bordercolor=self.ACCENT)
        style.map("Accent.TButton",
                  background=[("active", self.ACCENT_D), ("pressed", self.ACCENT_D)],
                  foreground=[("active", "white")])

        # Navigacni tlacitka v sidebaru
        style.configure("Nav.TButton", font=(FONT, 11), padding=(16, 12),
                        background=self.SIDEBAR, foreground="#C3C8D1",
                        relief="flat", anchor="w", bordercolor=self.SIDEBAR)
        style.map("Nav.TButton",
                  background=[("active", self.SIDEBAR2)],
                  foreground=[("active", "white")])
        style.configure("NavActive.TButton", font=(FONT, 11, "bold"), padding=(16, 12),
                        background=self.ACCENT, foreground="white",
                        relief="flat", anchor="w", bordercolor=self.ACCENT)
        style.map("NavActive.TButton",
                  background=[("active", self.ACCENT_D)],
                  foreground=[("active", "white")])

        # Main container
        main_frame = ttk.Frame(self._root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # left nav
        main_frame.columnconfigure(1, weight=1)  # content
        main_frame.rowconfigure(0, weight=1)

        # ── LEFT NAVIGATION (tmavy sidebar) ─────────────────────────────────
        nav_frame = tk.Frame(main_frame, width=220, bg=self.SIDEBAR)
        nav_frame.grid(row=0, column=0, sticky="ns")
        nav_frame.grid_propagate(False)

        tk.Label(nav_frame, text="  EEG BCI", bg=self.SIDEBAR, fg="white",
                 font=(FONT, 17, "bold")).pack(side="top", fill="x", padx=14, pady=(20, 4), anchor="w")
        tk.Label(nav_frame, text="  Motor Imagery", bg=self.SIDEBAR, fg=self.ACCENT,
                 font=(FONT, 9)).pack(side="top", fill="x", padx=14, pady=(0, 16), anchor="w")

        # Navigation buttons
        self._nav_buttons = {}
        nav_items = [
            ("info",    "  Prehled"),
            ("network", "  Sit / EEG stream"),
            ("record",  "  Zaznam mereni"),
            ("offline", "  Trenink + ERD"),
            ("online",  "  Online BCI"),
        ]

        for page_id, label in nav_items:
            btn = ttk.Button(
                nav_frame,
                text=label,
                style="Nav.TButton",
                command=lambda p=page_id: self._switch_page(p),
                takefocus=False,
            )
            btn.pack(side="top", fill="x", padx=8, pady=2)
            self._nav_buttons[page_id] = btn

        tk.Frame(nav_frame, height=1, bg=self.SIDEBAR2).pack(side="top", fill="x", padx=12, pady=14)

        # Config info (v sidebaru)
        tk.Label(nav_frame, text="KONFIGURACE", bg=self.SIDEBAR, fg=self.MUTED,
                 font=(FONT, 8, "bold")).pack(side="top", padx=14, pady=(0, 4), anchor="w")
        
        FONT = "Segoe UI"
        if self._config:
            n_cls = len(self._config.paradigm.get("classes", {}))
            config_text = (
                f"Trialu/trida : {self._config.experiment.trials_per_class}\n"
                f"Trid         : {n_cls}\n"
                f"Vzorkovani   : {self._config.preprocessing.sfreq:.0f} Hz\n"
                f"Filtr        : {self._config.preprocessing.l_freq:.0f}-{self._config.preprocessing.h_freq:.0f} Hz"
            )
            tk.Label(nav_frame, text=config_text, bg=self.SIDEBAR, fg="#A8AEB8",
                     font=("Consolas", 8), justify="left").pack(side="top", padx=14, anchor="w")

        tk.Frame(nav_frame, height=1, bg=self.SIDEBAR2).pack(side="top", fill="x", padx=12, pady=14)

        tk.Label(nav_frame, text="PACIENT", bg=self.SIDEBAR, fg=self.MUTED,
                 font=(FONT, 8, "bold")).pack(side="top", padx=14, pady=(0, 4), anchor="w")
        self._profile_sidebar_label = tk.Label(
            nav_frame, textvariable=self._profile_label_var, bg=self.SIDEBAR,
            fg="white", font=(FONT, 9), justify="left", wraplength=190,
        )
        self._profile_sidebar_label.pack(side="top", padx=14, anchor="w")
        ttk.Button(nav_frame, text="  Vybrat / vytvorit", style="Nav.TButton",
                   command=self._prompt_for_patient_profile, takefocus=False).pack(
            side="top", fill="x", padx=8, pady=(8, 0)
        )

        # ── RIGHT CONTENT AREA ─────────────────────────────────────────────
        self._content_frame = ttk.Frame(main_frame)
        self._content_frame.grid(row=0, column=1, sticky="nsew", padx=24, pady=22)
        self._content_frame.columnconfigure(0, weight=1)
        self._content_frame.rowconfigure(0, weight=1)
        
        # Create pages
        self._pages = {
            "info":    self._build_info_page,
            "network": self._build_network_page,
            "record":  self._build_record_page,
            "offline": self._build_offline_page,
            "online":  self._build_online_page,
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
        
        # Highlight aktivni navigacni tlacitko pomoci stylu
        for pid, btn in self._nav_buttons.items():
            btn.configure(style="NavActive.TButton" if pid == page_id else "Nav.TButton")

        self._mode_var.set(page_id)
    
    def _on_page_change(self) -> None:
        """Called when navigating pages."""
        page = self._mode_var.get()
        self._switch_page(page)
    
    # ── PAGE BUILDERS ──────────────────────────────────────────────────────
    
    def _card(self, parent, title: str) -> ttk.Frame:
        """Vytvori 'kartu' - ramecek se stinem-like ohraničenim a nadpisem."""
        outer = ttk.Labelframe(parent, text=title, padding=14)
        outer.pack(fill="x", pady=(0, 14))
        return outer

    def _build_info_page(self, parent: ttk.Frame) -> None:
        """Prehledova stranka."""
        ttk.Label(parent, text="Prehled systemu", style="Title.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            parent,
            text="Brain-Computer Interface pro rozpoznavani motoricke imaginace z EEG.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 16))

        # Workflow karta
        wf = self._card(parent, "Pracovni postup")
        steps = [
            ("1.  Sit / EEG stream", "Pripojeni k EEG zarizeni pres LSL po siti"),
            ("2.  Zaznam mereni", "4 tridy motoricke imaginace, ulozeni do BDF s markery"),
            ("3.  Trenink + ERD", "Trenovani modelu + analyza kontralateralniho utlumu"),
            ("4.  Online BCI", "Realtime predikce na zivem EEG streamu"),
        ]
        for title, desc in steps:
            row = ttk.Frame(wf)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=title, style="Section.TLabel", width=22).pack(side="left", anchor="w")
            ttk.Label(row, text=desc, style="Sub.TLabel").pack(side="left", anchor="w")

        if self._config:
            cfg = self._config
            classes = ", ".join(cfg.paradigm.get("classes", {}).keys())
            total = cfg.experiment.trials_per_class * len(cfg.paradigm.get("classes", {}))

            c = self._card(parent, "Aktualni konfigurace")
            grid = ttk.Frame(c)
            grid.pack(fill="x")
            rows = [
                ("Tridy", classes),
                ("Trialu celkem", f"{total}  ({cfg.experiment.trials_per_class} na tridu)"),
                ("Casovani", f"baseline {cfg.experiment.baseline_duration}s -> cue {cfg.experiment.cue_duration}s "
                             f"-> imaginace {cfg.experiment.imagery_duration}s -> pauza {cfg.experiment.iti_duration}s"),
                ("Vzorkovani", f"{cfg.preprocessing.sfreq:.0f} Hz"),
                ("Filtr", f"{cfg.preprocessing.l_freq:.0f}-{cfg.preprocessing.h_freq:.0f} Hz, notch {cfg.preprocessing.notch_freq:.0f} Hz"),
                ("Priznaky", f"{cfg.features.method.upper()}, pasma {cfg.features.bands}"),
                ("Klasifikator", cfg.classifier.algorithm.upper()),
            ]
            for i, (k, v) in enumerate(rows):
                ttk.Label(grid, text=k, style="Section.TLabel", width=16).grid(row=i, column=0, sticky="w", pady=2)
                ttk.Label(grid, text=v, style="Sub.TLabel").grid(row=i, column=1, sticky="w", pady=2)
    
    # ── Network EEG page ──────────────────────────────────────────────────────

    def _build_network_page(self, parent: ttk.Frame) -> None:
        """Stránka pro nastavení síťového EEG streamu přes LSL."""
        ttk.Label(parent, text="Network EEG – LSL Stream", style="Title.TLabel").pack(
            anchor="w", pady=(0, 6)
        )
        ttk.Label(
            parent,
            text=(
                "EEG zarizeni streamuje data pres LSL (Lab Streaming Layer) po siti.\n"
                "Na stejne lokalni siti funguje automaticka detekce.\n"
                "Pro prime IP pripojeni (jina podsit) zadejte IP adresu zarizeni."
            ),
            style="Sub.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        # ── IP nastaveni ──────────────────────────────────────────────
        ip_frame = ttk.LabelFrame(parent, text="IP adresa EEG zarizeni", padding=10)
        ip_frame.pack(fill="x", pady=(0, 10))
        ip_frame.columnconfigure(1, weight=1)

        ttk.Label(ip_frame, text="IP / hostname:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._net_ip_var = tk.StringVar(
            value=", ".join(self._config.lsl.known_peers) if self._config else ""
        )
        ip_entry = ttk.Entry(ip_frame, textvariable=self._net_ip_var, width=30)
        ip_entry.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(
            ip_frame,
            text="Priklad: 192.168.1.100   (vice adres oddelit carkou)\n"
                 "Nechat prazdne = autodetekce na lokalni siti (multicast).",
            style="Sub.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        btn_row = ttk.Frame(ip_frame)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(btn_row, text="Ulozit a zapsat lsl_api.cfg",
                   command=self._on_net_save).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Ping / Test IP",
                   command=self._on_net_ping).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Skenovat sit (LSL)",
                   command=self._on_net_scan).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Sken podsitě (vsechny PC)",
                   command=self._on_net_subnet_scan).pack(side="left")

        # ── Status ────────────────────────────────────────────────────
        status_frame = ttk.LabelFrame(parent, text="Dostupne LSL streamy", padding=10)
        status_frame.pack(fill="both", expand=True, pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)

        self._net_status_var = tk.StringVar(value="Kliknete na 'Skenovat sit' pro hledani EEG streamu.")
        ttk.Label(status_frame, textvariable=self._net_status_var,
                  style="Sub.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        # Listbox se streamy
        list_frame = ttk.Frame(status_frame)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._net_listbox = tk.Listbox(
            list_frame,
            font=("Courier", 9),
            height=8,
            selectmode=tk.SINGLE,
            bg="#1e1e1e",
            fg="#d4d4d4",
            selectbackground="#0066cc",
        )
        # Barevné označení: LOCAL = zelená, NETWORK = žlutá
        self._net_listbox.tag_configure = None  # placeholder
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                  command=self._net_listbox.yview)
        self._net_listbox.configure(yscrollcommand=scrollbar.set)
        self._net_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Legenda – primo pod listboxem uvnitr terminalu
        leg = ttk.Frame(status_frame)
        leg.grid(row=1, column=0, sticky="w", pady=(4, 2))
        tk.Label(leg, text="  [LOCAL]  ", bg="#1e1e1e", fg="#50fa7b",
                 font=("Courier", 8)).pack(side="left")
        tk.Label(leg, text="= primo na tomto pocitaci  ",
                 font=("Courier", 8)).pack(side="left")
        tk.Label(leg, text="  [NETWORK]  ", bg="#1e1e1e", fg="#ffb86c",
                 font=("Courier", 8)).pack(side="left")
        tk.Label(leg, text="= pres sit (jiny pocitac/zarizeni)  ",
                 font=("Courier", 8)).pack(side="left")
        tk.Label(leg, text="  [LSL]  ", bg="#1e1e1e", fg="#50fa7b",
                 font=("Courier", 8)).pack(side="left")
        tk.Label(leg, text="= otevreny LSL port",
                 font=("Courier", 8)).pack(side="left")

        # Tlacitko VEN z terminalu – v normalnim svetlem panelu
        btn_outer = ttk.Frame(parent)
        btn_outer.pack(fill="x", pady=(6, 4))
        ttk.Button(
            btn_outer,
            text="Pouzit vybrany stream jako EEG zdroj",
            command=self._on_net_use_selected,
        ).pack(side="left")
        ttk.Label(
            btn_outer,
            text="   (dvojklik na stream ma stejny efekt)",
            style="Sub.TLabel",
        ).pack(side="left")
        # Dvojklik na listbox = stejne jako tlacitko
        self._net_listbox.bind("<Double-Button-1>", lambda _e: self._on_net_use_selected())

        # ── lsl_api.cfg info ─────────────────────────────────────────
        cfg_frame = ttk.LabelFrame(parent, text="lsl_api.cfg", padding=8)
        cfg_frame.pack(fill="x")
        self._net_cfg_var = tk.StringVar(value="")
        self._net_cfg_label = ttk.Label(cfg_frame, textvariable=self._net_cfg_var,
                                        style="Info.TLabel", justify="left")
        self._net_cfg_label.pack(anchor="w")
        self._update_cfg_label()

    def _update_cfg_label(self) -> None:
        from .lsl_network import get_lsl_cfg_path, read_known_peers
        cfg = get_lsl_cfg_path()
        if cfg:
            peers = read_known_peers()
            self._net_cfg_var.set(
                f"Soubor: {cfg}\n"
                f"KnownPeers: {peers if peers else '(zadne – autodetekce)'}"
            )
        else:
            self._net_cfg_var.set("lsl_api.cfg nenalezen – pouziva se multicast autodetekce.")

    def _on_net_save(self) -> None:
        """Ulozi IP adresy do lsl_api.cfg."""
        from .lsl_network import configure_network
        raw = self._net_ip_var.get().strip()
        peers = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()] if raw else []
        path = configure_network(known_peers=peers)
        self._net_status_var.set(f"Ulozeno: {path}")
        self._update_cfg_label()
        # Aktualizovat config v pameti
        if self._config and peers != self._config.lsl.known_peers:
            self._config.lsl.known_peers = peers

    def _on_net_scan(self) -> None:
        """Spusti skenovani LSL streamu (subprocess – nacte cerstvy lsl_api.cfg)."""
        self._net_status_var.set("Skenuju sit... (az 8 sekund)")
        self._net_listbox.delete(0, tk.END)
        self._net_listbox.insert(tk.END, "  hledam streamy pres novy proces (nacte KnownPeers)...")

        def _scan() -> None:
            # Pouzit subprocess aby se nacetl cerstvy lsl_api.cfg s KnownPeers.
            # liblsl cte konfiguraci pri inicializaci – pokud uzivatel ulozil
            # lsl_api.cfg az po startu aplikace, subprocess ho nacte spravne.
            import subprocess, sys, json as _json, os as _os
            from pathlib import Path

            project_root = str(Path(__file__).resolve().parents[1])
            scan_script = (
                "import sys, json; sys.path.insert(0, r'" + project_root + "'); "
                "from pylsl import resolve_streams; "
                "streams = resolve_streams(wait_time=6.0); "
                "print(json.dumps([{"
                "'name': s.name(), 'type': s.type(), "
                "'channels': s.channel_count(), 'sfreq': s.nominal_srate(), "
                "'hostname': s.hostname(), 'source_id': s.source_id()"
                "} for s in streams]))"
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", scan_script],
                    capture_output=True, text=True, timeout=12,
                )
                raw_out = proc.stdout.strip()
                # Najit JSON radek (posledni radek)
                for line in reversed(raw_out.splitlines()):
                    line = line.strip()
                    if line.startswith("["):
                        data = _json.loads(line)
                        from .lsl_network import StreamDesc, _is_local
                        # Simulovat StreamDesc ze slovniku
                        class _FakeInfo:
                            def __init__(self, d):
                                self._d = d
                            def name(self): return self._d["name"]
                            def type(self): return self._d["type"]
                            def channel_count(self): return self._d["channels"]
                            def nominal_srate(self): return self._d["sfreq"]
                            def hostname(self): return self._d["hostname"]
                            def source_id(self): return self._d.get("source_id", "")
                        results = [StreamDesc(_FakeInfo(d)) for d in data]
                        break
                else:
                    results = []
                    if proc.stderr:
                        logger.debug("scan subprocess stderr: %s", proc.stderr[:300])
            except Exception as exc:
                logger.warning("Subprocess scan selhal: %s – fallback na primy scan", exc)
                from .lsl_network import scan_streams
                results = scan_streams(timeout=5.0)

            def _update() -> None:
                self._net_listbox.delete(0, tk.END)
                if not results:
                    self._net_status_var.set(
                        "Zadne LSL streamy nenalezeny. Zkontrolujte:\n"
                        "  * Je EEG zarizeni zapnuto a pripojeno ke stejne siti?\n"
                        "  * Je firewall nastaven pro LSL (port 16571-16604)?\n"
                        "  * Pro jinou podsit zadejte IP adresu vyse."
                    )
                    self._net_listbox.insert(tk.END, "  (zadne streamy)")
                else:
                    n_local   = sum(1 for s in results if s.is_local)
                    n_network = len(results) - n_local
                    self._net_status_var.set(
                        f"Nalezeno {len(results)} streamu  "
                        f"({n_local} LOCAL, {n_network} NETWORK). "
                        "Vyberte EEG stream a kliknete 'Pouzit vybrany stream'."
                    )
                    for i, s in enumerate(results):
                        eeg_marker = "EEG    " if s.stream_type.upper() == "EEG" else s.stream_type.ljust(7)
                        line = (
                            f"[{s.source_label:<7}] "
                            f"[{eeg_marker}] "
                            f"{s.name:<20} "
                            f"{s.channels}ch@{s.sfreq:.0f}Hz  "
                            f"{s.hostname}"
                        )
                        self._net_listbox.insert(tk.END, line)
                        # Barva: LOCAL=zelena, NETWORK=oranzova
                        color = "#50fa7b" if s.is_local else "#ffb86c"
                        self._net_listbox.itemconfig(i, fg=color)

                    # Oznacit prvni EEG stream
                    for i, s in enumerate(results):
                        if s.stream_type.upper() == "EEG":
                            self._net_listbox.selection_set(i)
                            break

                self._net_scan_results = results

            self._root.after(0, _update)

        self._net_scan_results = []
        import threading
        threading.Thread(target=_scan, daemon=True).start()

    def _on_net_ping(self) -> None:
        """Ping / TCP test zadane IP adresy."""
        ip = self._net_ip_var.get().strip().split(",")[0].strip()
        if not ip:
            self._net_status_var.set("Zadejte IP adresu do pole vyse, pak kliknete Ping / Test IP.")
            return
        self._net_status_var.set(f"Testuji {ip} ...")
        self._net_listbox.delete(0, tk.END)
        self._net_listbox.insert(tk.END, f"  ping {ip} ...")

        def _ping() -> None:
            from .lsl_network import ping_host
            status = ping_host(ip, count=3, timeout_sec=2.0)
            def _update() -> None:
                self._net_listbox.delete(0, tk.END)
                if status.reachable:
                    ping_str = f"{status.ping_ms:.0f} ms" if status.ping_ms >= 0 else "neznamo"
                    lsl_str  = "LSL port 16571 OTEVRENY" if status.lsl_port_open else "LSL port 16571 zavreny"
                    line = f"  [OK]  {ip}   ping={ping_str}   {lsl_str}"
                    self._net_listbox.insert(tk.END, line)
                    self._net_listbox.itemconfig(0, fg="#50fa7b")
                    self._net_status_var.set(
                        f"Pocitac {ip} je dosazitelny (ping={ping_str}).\n"
                        + (
                            "LSL port je OTEVRENY – muzete spustit sken LSL streamu."
                            if status.lsl_port_open
                            else "LSL port je zavreny – spustte LSL software na cilovem PC (LabRecorder, OpenVibe...)."
                        )
                    )
                else:
                    line = f"  [FAIL] {ip}  NEDOSTUPNY – {status.error}"
                    self._net_listbox.insert(tk.END, line)
                    self._net_listbox.itemconfig(0, fg="#ff5555")
                    self._net_status_var.set(
                        f"Pocitac {ip} neodpovida.\n"
                        "Zkontrolujte:\n"
                        "  * Je zarizeni zapnuto a pripojeno ke stejnemu switchi/AP?\n"
                        "  * Neni firewall blokujici ICMP ping?\n"
                        "  * Je IP adresa spravna? (zkuste Sken podsitě)"
                    )
            self._root.after(0, _update)

        import threading
        threading.Thread(target=_ping, daemon=True).start()

    def _on_net_subnet_scan(self) -> None:
        """Prohledá celou podsíť a zobrazí všechny aktivní pocitace."""
        self._net_status_var.set("Skenuji podsit... (muze trvat 15-30 sekund)")
        self._net_listbox.delete(0, tk.END)
        self._net_listbox.insert(tk.END, "  skenování 254 adres paralelne...")

        def _scan() -> None:
            from .lsl_network import scan_subnet
            subnet_hint = ""
            raw_ip = self._net_ip_var.get().strip().split(",")[0].strip()
            if raw_ip:
                parts = raw_ip.split(".")
                if len(parts) >= 3:
                    subnet_hint = ".".join(parts[:3])
            results = scan_subnet(subnet=subnet_hint, timeout_sec=0.4, max_workers=64)

            def _update() -> None:
                self._net_listbox.delete(0, tk.END)
                if not results:
                    self._net_listbox.insert(tk.END, "  zadne aktivni pocitace nenalezeny")
                    self._net_status_var.set(
                        "Zadne pocitace nenalezeny. Zkontrolujte pripojeni k siti."
                    )
                    return

                self._net_status_var.set(
                    f"Nalezeno {len(results)} aktivnich zarízení na siti. "
                    "Zelene = ma otevreny LSL port."
                )
                for s in results:
                    ping_str = f"{s.ping_ms:.0f}ms" if s.ping_ms >= 0 else "?"
                    lsl_tag  = " [LSL]" if s.lsl_port_open else ""
                    line = f"  {s.host:<16}  ping={ping_str:<8}{lsl_tag}"
                    self._net_listbox.insert(tk.END, line)
                    # Barva: LSL otevreny = zelena, jen dosazitelny = bila
                    color = "#50fa7b" if s.lsl_port_open else "#d4d4d4"
                    idx = self._net_listbox.size() - 1
                    self._net_listbox.itemconfig(idx, fg=color)

            self._root.after(0, _update)

        import threading
        threading.Thread(target=_scan, daemon=True).start()

    def _on_net_use_selected(self) -> None:
        """Nastavi vybrany stream jako EEG zdroj v konfiguraci."""
        results = getattr(self, "_net_scan_results", [])
        sel = self._net_listbox.curselection()
        if not sel or not results:
            return
        idx = sel[0]
        if idx >= len(results):
            return
        s = results[idx]
        if self._config:
            self._config.lsl.eeg_stream_name = s.name
            self._config.lsl.eeg_stream_type = s.stream_type
        source_info = (
            "primo na tomto pocitaci (LOCAL)"
            if s.is_local
            else f"pres sit ze zarizeni '{s.hostname}' (NETWORK)"
        )
        self._net_status_var.set(
            f"Nastaveno: {s.name} [{s.stream_type}] "
            f"{s.channels}ch @ {s.sfreq:.0f}Hz\n"
            f"Zdroj: {source_info}\n"
            f"Toto nastaveni plati pro tuto session. "
            f"Pro trvale ulozeni zmente config.yaml: lsl.eeg_stream_name: \"{s.name}\""
        )

    def _build_record_page(self, parent: ttk.Frame) -> None:
        """Stranka zaznamu motoricke imaginace."""
        ttk.Label(parent, text="Zaznam mereni", style="Title.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            parent,
            text="Spusti vizualni paradigma motoricke imaginace a soucasne nahrava EEG z LSL do BDF.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 16))

        # Prubeh trialu
        flow = self._card(parent, "Prubeh jednoho trialu")
        cfg = self._config
        if cfg:
            e = cfg.experiment
            ttk.Label(
                flow,
                text=(
                    f"1. Fixacni kriz ({e.baseline_duration}s)  ->  "
                    f"2. SIPKA smer ({e.cue_duration}s)  ->  "
                    f"3. IMAGINACE bez sipky ({e.imagery_duration}s)  ->  "
                    f"4. pauza ({e.iti_duration}s)\n\n"
                    "Marker se posila na zacatku faze 3 (imaginace), kdy uz na obrazovce\n"
                    "neni sipka -> nahrane okno neobsahuje ocni artefakty z presunu pohledu."
                ),
                style="Sub.TLabel", justify="left",
            ).pack(anchor="w")

        # Tridy / smery
        cls = self._card(parent, "Tridy a smery sipek")
        ttk.Label(
            cls,
            text=("<-  LEVA RUKA          ->  PRAVA RUKA\n"
                  "dolu  OBE NOHY       nahoru  JAZYK"),
            style="Info.TLabel", justify="left",
        ).pack(anchor="w")

        # Predpoklady
        req = self._card(parent, "Pred spustenim")
        ttk.Label(
            req,
            text=("- EEG stream dostupny (zalozka Sit / EEG stream -> Skenovat)\n"
                  "- vybrany pacient (panel vlevo dole)\n"
                  "- nainstalovany PsychoPy\n"
                  "- ESC kdykoli behem mereni ukonci session"),
            style="Sub.TLabel", justify="left",
        ).pack(anchor="w")

        ttk.Button(parent, text="Spustit zaznam mereni", style="Accent.TButton",
                   command=self._on_start_record).pack(fill="x", pady=(8, 0))

    def _build_offline_page(self, parent: ttk.Frame) -> None:
        """Stranka treninku modelu + ERD analyzy."""
        ttk.Label(parent, text="Trenink modelu + ERD analyza", style="Title.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            parent,
            text="Nacte nahravku (BDF/EDF/FIF), natrenuje klasifikator a spocita ERD/koherenci.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 16))

        # Vyber souboru
        file_frame = self._card(parent, "EEG soubor (BDF / EDF / FIF)")
        entry_row = ttk.Frame(file_frame)
        entry_row.pack(fill="x")
        ttk.Entry(entry_row, textvariable=self._file_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(entry_row, text="Prochazet...", command=self._browse_file).pack(side="left")

        # Akce
        actions = self._card(parent, "Akce")
        ttk.Label(
            actions,
            text="Trenink: extrahuje priznaky (FBCSP), natrenuje LDA/SVM, ulozi model.\n"
                 "ERD analyza: spocita kontralateralni utlum mu/beta + koherenci hemisfer.",
            style="Sub.TLabel", justify="left",
        ).pack(anchor="w", pady=(0, 10))

        btn_row = ttk.Frame(actions)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Trenovat model", style="Accent.TButton",
                   command=self._on_start_offline).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="ERD / koherence analyza",
                   command=self._on_start_erd).pack(side="left")
    
    def _build_online_page(self, parent: ttk.Frame) -> None:
        """Stranka realtime predikce."""
        ttk.Label(parent, text="Online BCI - realtime predikce", style="Title.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            parent,
            text="Pripoji se k zivemu EEG streamu a prubezne klasifikuje motorickou imaginaci "
                 "natrenovanym modelem.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 16))

        req = self._card(parent, "Pred spustenim")
        ttk.Label(
            req,
            text=("- existuje natrenovany model (zalozka Trenink + ERD)\n"
                  "- EEG stream dostupny pres LSL (zalozka Sit / EEG stream)\n"
                  "- stejny pocet kanalu a pasem jako pri treninku\n"
                  "- predikce se vypisuji do logu, zastaveni: zavrenim okna logu"),
            style="Sub.TLabel", justify="left",
        ).pack(anchor="w")

        ttk.Button(parent, text="Spustit Online BCI", style="Accent.TButton",
                   command=self._on_start_online).pack(fill="x", pady=(8, 0))

    def _prompt_for_patient_profile(self) -> None:
        """Open the modal profile picker and store the selected patient."""
        dialog = PatientProfileDialog(self._root)
        profile = dialog.show()
        if profile is None:
            if self._patient_profile is None:
                self._profile_label_var.set("Zadny pacient nevybran")
            return

        self._patient_profile = profile
        self._profile_label_var.set(profile.display_name)
        logger.info("Selected patient profile: %s", profile.display_name)

    def _record_instructions(self) -> bool:
        """Show pre-recording instructions and confirm start."""
        profile_name = self._patient_profile.display_name if self._patient_profile else "anonymous patient"
        dialog = RecordingInstructionsDialog(self._root, profile_name)
        return dialog.show()
    
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
        if self._patient_profile is None:
            self._prompt_for_patient_profile()
        if self._patient_profile is None:
            messagebox.showerror("Patient required", "Please select or create a patient profile before recording.")
            return
        if not _has_psychopy():
            messagebox.showerror(
                "Psychopy missing",
                "Record mode requires psychopy, which is not installed in the current environment.\n\n"
                "Install it in .venv310 with:\n"
                "  .\\.venv310\\Scripts\\Activate.ps1\n"
                "  python -m pip install psychopy\n\n"
                "Then start the app again.",
            )
            return
        if not self._record_instructions():
            return

        # Kontrola LSL streamu pred zahajenim zaznamu
        if not self._check_lsl_stream_before_record():
            return

        selection = GuiSelection(
            mode="record",
            patient_profile_id=self._patient_profile.profile_id,
            patient_profile_name=self._patient_profile.display_name,
            config=self._config,   # predat in-memory config se zmenami z Network EEG
        )
        self._start_task(selection)

    def _check_lsl_stream_before_record(self) -> bool:
        """Rychla kontrola LSL streamu pred zahajenim. Vraci True = pokracovat."""
        import subprocess, sys, json as _json
        from pathlib import Path
        project_root = str(Path(__file__).resolve().parents[1])
        scan_script = (
            "import sys, json; sys.path.insert(0, r'" + project_root + "'); "
            "from pylsl import resolve_streams; "
            "s = resolve_streams(wait_time=4.0); "
            "print(json.dumps([{'name': x.name(), 'type': x.type()} for x in s]))"
        )
        streams = []
        try:
            proc = subprocess.run(
                [sys.executable, "-c", scan_script],
                capture_output=True, text=True, timeout=8,
            )
            for line in reversed(proc.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("["):
                    streams = _json.loads(line)
                    break
        except Exception:
            pass

        if streams:
            s = streams[0]
            logger.info("LSL EEG stream nalezen: %s [%s]", s.get("name"), s.get("type"))
            return True

        # Stream nenalezen - zeptat se uzivatele
        answer = messagebox.askyesno(
            "LSL EEG stream nenalezen",
            "Zadny LSL EEG stream nebyl nalezen na siti.\n\n"
            "Mozne priciny:\n"
            "  • EEG zarizeni neni zapnuto\n"
            "  • LSL software na druhem PC neni spusten\n"
            "  • Firewall blokuje port 16571\n"
            "  • IP adresa neni nastavena (zkuste zalozku Network EEG)\n\n"
            "Chcete pokracovat i bez EEG streamu?\n"
            "(Paradigma se spusti, ale EEG data se nenahraji)",
            icon="warning",
        )
        return answer
    
    def _on_start_offline(self) -> None:
        """Start offline training mode."""
        if not self._file_var.get():
            messagebox.showerror("Chybi soubor", "Nejprve vyberte EEG soubor (BDF/EDF/FIF).")
            return
        selection = GuiSelection(mode="offline", offline_file=self._file_var.get(), config=self._config)
        self._start_task(selection)

    def _on_start_erd(self) -> None:
        """Spustit ERD / koherence analyzu."""
        if not self._file_var.get():
            messagebox.showerror("Chybi soubor", "Nejprve vyberte EEG soubor (BDF/EDF/FIF).")
            return
        selection = GuiSelection(mode="erd", offline_file=self._file_var.get(), config=self._config)
        self._start_task(selection)

    def _on_start_online(self) -> None:
        """Start online BCI mode."""
        selection = GuiSelection(mode="online", config=self._config)
        self._start_task(selection)
    
    def _start_task(self, selection: GuiSelection) -> None:
        """Start background task."""
        self._running = True
        self._last_selection = selection
        
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

        # If the finished task was a recording, offer to run offline training
        try:
            if self._last_selection and self._last_selection.mode == "record":
                do_train = messagebox.askyesno(
                    "Train model?",
                    "Recording finished. Do you want to train a model now using a saved EDF/BDF file?",
                    parent=self._root,
                )
                if do_train:
                    path = filedialog.askopenfilename(
                        title="Select EEG file for training",
                        filetypes=[("EEG files", "*.edf *.bdf"), ("All files", "*.*")],
                    )
                    if path:
                        selection = GuiSelection(mode="offline", offline_file=path)
                        self._start_task(selection)
        except Exception:
            logger.exception("Error offering train-after-record flow")


def _run_selection(selection: GuiSelection) -> None:
    """Execute the selected mode (runs in background thread)."""
    if selection.mode == "record":
        from .config import load_config as _load
        from .eeg_recorder import EegRecorder

        # Pouzit in-memory config z GUI (obsahuje vybrany stream ze zalozky
        # Network EEG). Pokud neni k dispozici, nacist ze souboru.
        cfg = selection.config if selection.config is not None else _load()
        patient_name = selection.patient_profile_name or "unknown"

        print(
            f"EEG stream: typ='{cfg.lsl.eeg_stream_type}' "
            f"jmeno='{cfg.lsl.eeg_stream_name}' "
            f"timeout={cfg.lsl.resolution_timeout}s"
        )

        recorder = EegRecorder(cfg)
        recorder.start()

        proc = Process(
            target=_start_paradigm_proc,
            args=(selection.patient_profile_id, selection.patient_profile_name),
        )
        proc.start()
        proc.join()

        saved_path = recorder.stop(patient_name=patient_name)
        if saved_path:
            print(f"\nEEG zaznam ulozen: {saved_path}")
            print("   Pro trenovani: Train Model -> vyberte tento soubor")
        else:
            print(
                "\nVAROVANI: Interni nahravani nebylo dostupne (zadny LSL stream).\n"
                "   Overeni:\n"
                "   1. Je LSL software na EEG pocitaci spusten?\n"
                "   2. Zalozka Network EEG -> Skenovat sit -> stream musi byt videt\n"
                "   3. Po vybrani streamu v Network EEG kliknete 'Pouzit vybrany stream'\n"
                "   4. Teprve pak spustte Record\n"
                f"   Hledany stream: typ='{cfg.lsl.eeg_stream_type}' jmeno='{cfg.lsl.eeg_stream_name}'"
            )

    elif selection.mode == "offline":
        from .offline_analysis import run_offline_from_file

        if not selection.offline_file:
            raise ValueError("No EEG file selected")

        acc, n_epochs = run_offline_from_file(selection.offline_file)
        print(f"\nTrenovani dokonceno: {n_epochs} epoch, presnost: {acc:.4f}")

    elif selection.mode == "erd":
        from .analysis_erd import run_erd_analysis

        if not selection.offline_file:
            raise ValueError("Nebyl vybran soubor")

        cfg = selection.config
        result = run_erd_analysis(selection.offline_file, config=cfg)
        print(result["report_text"])
        if result.get("figure_path"):
            print(f"\nGraf ulozen: {result['figure_path']}")

    elif selection.mode == "online":
        from .config import load_config
        from .online_bci import run_online_bci

        config = selection.config if selection.config is not None else load_config()
        run_online_bci(config)


def run_gui() -> None:
    """Launch the enhanced GUI."""
    root = tk.Tk()
    EegAppGui(root)
    root.mainloop()
