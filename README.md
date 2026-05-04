# eeg_app

BCI aplikace založená na motorické imaginaci (MI) pro 4 směry
(bod nahoře, dole, vlevo, vpravo) s využitím EEG a Lab Streaming
Layer (LSL).

## Cíle

- Připojení k EEG zařízení přes LSL v reálném čase.
- Zobrazování jednoduchých 2D vizuálních podnětů se 4 směry a
	přesně časovanými událostmi (markery).
- Segmentace dat a extrakce příznaků vhodných pro MI BCI.
- Trénink a vyhodnocení klasifikátoru (např. LDA/SVM) a následné
	online rozpoznávání.

## Struktura projektu

- `src/` – zdrojový kód aplikace
	- `main.py` – vstupní bod aplikace (CLI / režimy)
	- `config.py` – načítání konfigurace
	- `lsl_acquisition.py` – připojení k EEG přes LSL
	- `preprocessing.py` – předzpracování a epochování
	- `features.py` – extrakce příznaků
	- `classifier.py` – trénink a uložení modelu
	- `offline_analysis.py` – offline pipeline pro trénink a evaluaci
	- `online_bci.py` – online BCI smyčka
	- `stimuli/four_dots_paradigm.py` – vizuální MI paradigma se 4 body
- `config/config.yaml` – parametry experimentu a zpracování
- `data/raw/` – syrová naměřená data
- `data/processed/` – epoched/feature data
- `models/` – uložené klasifikátory
- `reports/` – výsledky vyhodnocení
- `logs/` – logy běhu aplikace

## Základní workflow

1. Nainstalujte závislosti: `pip install -r requirements.txt`.
2. Spusťte zaznamenání trénovacího sezení (MI paradigma se 4 směry).
3. Proveďte offline analýzu a natrénujte klasifikátor.
4. Spusťte online BCI režim s natrénovaným modelem.
