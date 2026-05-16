# eeg_app

### Instalace
```powershell
# 1. Otevřete PowerShell v adresáři projektu
cd C:\Users\kryst\OneDrive\Documents\skola\eeg_app

# 2. Aktivujte Conda prostředí
conda activate eeg_py310

# 3. Spusťte aplikaci
python run_app.py
```

**Poznámka:** Pokud Conda nenajde prostředí `eeg_py310`, vytvořte ho:
```powershell
conda create -n eeg_py310 python=3.10 -y
conda activate eeg_py310
pip install -r requirements.txt
python run_app.py
```

### po spuštění
1. Otevře se **grafické menu** (GUI)
2. V levém panelu vidíte 4 možnosti:
   - **Overview** - zobrazení aktuální konfigurace
   - **Record** - spuštění experimentu (vyžaduje EEG zařízení)
   - **Train Model** - trénování klasifikátoru (vybere EEG soubor)
   - **Online BCI** - realtime klasifikace (vyžaduje LSL stream)

---

## Pracovní postup

### Normální workflow (doporučeno)
```
1. Record      → Spusťte motorickou imaginaci (generuje markery)
                 Záznam: LabRecorder zachytí EEG + markery
   
2. Train Model → Vyberte nahraný EEG soubor (.edf/.bdf)
                 Systém: extrahuje příznaky → trénuje model
                 Výstup: model → models/model_latest.joblib
   
3. Online BCI  → Připojte živý EEG stream (LSL)
                 Systém: načte model → realtime klasifikace
                 Výstup: logy v logs/eeg_app.log
```

### Příklady konfigurací

Aplikace obsahuje šablony pro různé nastavení:

| Soubor | Popis | Třídy | Kanály |
|--------|-------|-------|--------|
| `config.yaml` | Výchozí 4-třídní | UP, DOWN, LEFT, RIGHT | Auto |
| `config_2class.yaml` | 2-třídní  | LEFT, RIGHT | Auto |
| `config_6class.yaml` | 6-třídní  | 6 končetin | Auto |
| `config_8ch.yaml` | Přenosné zařízení | 4 třídy | 8 elektrod |
| `config_64ch.yaml` | Laboratorní system | 4 třídy | 64 elektrod |

**Jak používat jinou konfiguraci:**
```powershell
# Nastavte proměnnou prostředí před spuštěním
$env:EEG_CONFIG_PATH = "config/config_2class.yaml"
python run_app.py
```

## Struktura adresářů

```
eeg_app/
├── run_app.py              ← Spusťte tímto
├── requirements.txt        ← Závislosti
│
├── config/
│   ├── config.yaml         ← Hlavní konfigurace
│   ├── config_2class.yaml  ← Šablona: 2 třídy
│   ├── config_6class.yaml  ← Šablona: 6 tříd
│   ├── config_8ch.yaml     ← Šablona: 8 kanálů
│   └── config_64ch.yaml    ← Šablona: 64 kanálů
│
├── src/                    ← Zdrojový kód
│   ├── main.py            ← Vstupní bod
│   ├── config.py          ← Validace konfigurace
│   ├── gui_app_v2.py      ← Nový GUI s navigací
│   ├── offline_analysis.py ← Trénování
│   ├── online_bci.py      ← Realtime klasifikace
│   ├── lsl_acquisition.py ← Připojení k EEG
│   ├── preprocessing.py   ← Filtrování
│   ├── features.py        ← Extrakce příznaků
│   ├── classifier.py      ← modely
│   ├── logging_config.py  ← Logování
│   └── stimuli/
│       └── paradigm_base.py ← N-třídní MI paradigma
│
├── models/
│   └── model_latest.joblib ← Natrénovaný model
│
├── logs/
│   └── eeg_app.log        ← Záznamy běhu
│
├── README.md              ← Tento soubor
├── README_ENHANCED.md     ← Rozšíření: příklady, troubleshooting
├── CONFIG_GUIDE.md        ← Detailní konfigurace
└── ARCHITECTURE.md        ← Technická architektura
```

## Instalace prostředí

```powershell
# Ověřte že máte Conda
conda --version

# Vytvořte Python 3.10 prostředí
conda create -n eeg_py310 python=3.10 -y

# Aktivujte prostředí
conda activate eeg_py310

# Nainstalujte závislosti
pip install -r requirements.txt

# Spusťte aplikaci
python run_app.py
```

## Řešení problémů

### Nelze najít modul 'src'
- Ujistěte se že jste v adresáři `eeg_app`
- Spusťte: `cd C:\Users\kryst\OneDrive\Documents\skola\eeg_app`

### Chyba: No module named 'psychopy'
- Aktualizujte prostředí: `pip install -r requirements.txt`
- Nebo reinstalujte: `pip install psychopy`

### Nelze připojit k EEG zařízení (Online BCI)
- Spusťte LabRecorder nebo jiný LSL zdroj
- Ověřte název streamu v `config.yaml` (výchozí: "EEG")
- Počkejte 10 sekund na timeout

### GUI se neotevře
- Zkontrolujte že máte Python 3.10+ aktivní
- Zkuste spustit s debug logem: `python run_app.py -l DEBUG`
- Podívejte se do `logs/eeg_app.log`

## Další dokumentace

- **README_ENHANCED.md** - Úplný uživatelský průvodce
- **CONFIG_GUIDE.md** - Detailní konfigurace
- **ARCHITECTURE.md** - Technický design a API

---

