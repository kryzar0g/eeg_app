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
            ("info",    "ℹ️  Overview"),
            ("network", "🌐 Network EEG"),
            ("record",  "▶️  Record"),
            ("offline", "📊 Train Model"),
            ("online",  "🔴 Online BCI"),
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

        ttk.Separator(nav_frame, orient="horizontal").pack(
            side="top", fill="x", padx=5, pady=10
        )

        ttk.Label(nav_frame, text="Patient", style="Section.TLabel").pack(
            side="top", padx=10, pady=(0, 5), anchor="w"
        )
        ttk.Label(nav_frame, textvariable=self._profile_label_var, style="Info.TLabel", justify="left").pack(
            side="top", padx=10, anchor="w"
        )
        ttk.Button(nav_frame, text="Select / Create", command=self._prompt_for_patient_profile).pack(
            side="top", fill="x", padx=5, pady=(6, 0)
        )
        
        # ── RIGHT CONTENT AREA ─────────────────────────────────────────────
        self._content_frame = ttk.Frame(main_frame)
        self._content_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
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

        ttk.Button(
            status_frame,
            text="Pouzit vybrany stream (nastavit jako EEG zdroj)",
            command=self._on_net_use_selected,
        ).grid(row=1, column=0, sticky="w", pady=(8, 4))

        # Legenda – pod tlacitkem, aby nezasahovala do terminaloveho vystupu
        leg = ttk.Frame(status_frame)
        leg.grid(row=2, column=0, sticky="w", pady=(0, 2))
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
        """Spusti skenovani LSL streamu na siti (v pozadi)."""
        self._net_status_var.set("Skenuju sit... (az 5 sekund)")
        self._net_listbox.delete(0, tk.END)
        self._net_listbox.insert(tk.END, "  hledam streamy...")

        def _scan() -> None:
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
                 "Markers will be sent via LSL to synchronize with the visual paradigm.\n"
                 "Press ESC during the paradigm to stop the session safely.",
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

    def _prompt_for_patient_profile(self) -> None:
        """Open the modal profile picker and store the selected patient."""
        dialog = PatientProfileDialog(self._root)
        profile = dialog.show()
        if profile is None:
            if self._patient_profile is None:
                self._profile_label_var.set("No patient selected")
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
        selection = GuiSelection(
            mode="record",
            patient_profile_id=self._patient_profile.profile_id,
            patient_profile_name=self._patient_profile.display_name,
        )
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
        # PsychoPy/pyglet must run in the main thread of a process. Spawn a separate
        # process for the paradigm so its event loop runs in that process's main thread.
        #
        # Interní EEG recorder běží v tomto vlákně – nahrává EEG z LSL a ukládá FIF.
        # Pokud LSL není dostupné (žádné EEG zařízení), recorder tiše přeskočí
        # a zobrazení paradigmatu proběhne normálně.
        from .config import load_config as _load
        from .eeg_recorder import EegRecorder

        cfg = _load()
        patient_name = selection.patient_profile_name or "unknown"

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
            print(f"\n💾 EEG záznam uložen: {saved_path}")
            print(
                "   Pro trénování: Train Model → vyberte tento soubor\n"
                "   (FIF formát – kompatibilní s offline analýzou)"
            )
        else:
            print(
                "\n⚠️  Interní nahrávání nebylo dostupné (žádný LSL stream).\n"
                "   Pokud jste používali LabRecorder, soubor najdete v jeho výstupní složce."
            )

    elif selection.mode == "offline":
        from .offline_analysis import run_offline_from_file

        if not selection.offline_file:
            raise ValueError("No EEG file selected")

        acc, n_epochs = run_offline_from_file(selection.offline_file)
        print(f"\n✓ Trénování dokončeno: {n_epochs} epoch, přesnost: {acc:.4f}")

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
