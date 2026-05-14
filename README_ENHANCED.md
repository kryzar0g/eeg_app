# EEG Motor Imagery BCI Application

A flexible, production-ready Brain-Computer Interface system for motor imagery classification with support for any number of EEG channels and experimental classes.

## Features

✅ **Universal EEG Support**
- Works with 8, 16, 32, 64, 128+ channel EEG systems
- Automatic channel detection from LSL streams or EDF/BDF files
- No manual channel configuration needed

✅ **Flexible Paradigm**
- N-class motor imagery experiments (2-class, 4-class, 6-class, etc.)
- Dynamically positioned visual stimuli based on number of classes
- Support for custom class labels (LEFT/RIGHT, UP/DOWN/etc.)

✅ **Production Ready**
- Comprehensive logging to file and console
- Configuration validation with pydantic
- Proper error handling and recovery
- Environment variable support for custom paths

✅ **Enhanced User Interface**
- Hierarchical navigation menu
- Configuration overview panel
- Real-time log output window
- Per-mode instructions and requirements

✅ **Robust Online Classification**
- Real-time EEG processing with configurable buffer
- Serialized raw-epoch pipeline shared by offline and online mode
- Feature validation against trained models
- Mismatch detection (channels, epoch length, configuration)
- Graceful error handling

✅ **Fast CSP-Based Feature Learning**
- Filter Bank CSP features learned during offline training
- Optional classic CSP and PSD/Welch fallback for comparison
- Sliding-window crop augmentation to reduce calibration time
- Same learned pipeline is reused in online inference via `joblib`

✅ **Patient Profiles**
- Select an existing patient profile at startup or create a new one
- Patient metadata is stored as JSON files in `data/patients/`
- Record mode shows a short instruction screen before the first trial

## Quick Start

### Installation

```bash
# Clone or download the repository
cd eeg_app

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python run_app.py
```

This launches the GUI. Choose a mode from the left navigation:

1. **Overview** - View current configuration
2. **Record** - Run motor imagery paradigm (requires LabRecorder)
3. **Train Model** - Load EEG file and train classifier
4. **Online BCI** - Real-time classification (requires trained model)

When the GUI starts, it asks you to select an existing patient profile or create a new one. The selected profile is saved as JSON and shown in the sidebar.

## Configuration

### Quick Examples

**2-Class Setup (Left vs Right):**
```bash
export EEG_CONFIG_PATH=config/config_2class.yaml
python run_app.py
```

**6-Class Setup (Multiple limbs):**
```bash
export EEG_CONFIG_PATH=config/config_6class.yaml
python run_app.py
```

**Default 4-Class Setup:**
```bash
python run_app.py
```

### Detailed Configuration

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for:
- How to configure for different channel counts
- Frequency band selection
- Timing parameters
- Classifier algorithms
- Environment variable overrides

## System Architecture

```
run_app.py
└── src/main.py
    ├── record: MotorImageryParadigm (stimuli) → LSL markers
    ├── offline: offline_analysis.py (train) → classifier.py
    ├── online: online_bci.py (predict) ← model
    └── gui: gui_app_v2.py (enhanced UI with navigation)

Core modules:
├── config.py: Pydantic validation and config aliases
├── logging_config.py: Structured logging
├── lsl_acquisition.py: LSL stream handling
├── preprocessing.py: EEG filtering and array preprocessing
├── features.py: CSP / PSD feature transformers
├── classifier.py: Model training & evaluation pipeline
├── stimuli/paradigm_base.py: Flexible N-class paradigm
└── online_bci.py: Real-time classification
```

## Usage Examples

### Train Model from 8-Channel EEG (any sampling rate)
```bash
python run_app.py offline -f path/to/eeg_8ch.edf
```
The system automatically:
- Detects 8 channels
- Filters each epoch, learns CSP features per band, then trains LDA classifier
- Saves model to `models/model_latest.joblib`

### Train Model from 64-Channel EEG
```bash
python run_app.py offline -f path/to/eeg_64ch.edf
```
Works identically - no configuration changes needed!

### Run Online Classification (any channel count)
```bash
# Stream EEG via LabRecorder (or any LSL source)
python run_app.py online
```
The system:
- Connects to EEG stream (auto-detects channels)
- Loads trained model
- Reuses the same serialized preprocessing + CSP pipeline from offline training
- Runs real-time classification

### CSP vs PSD fallback
- Default feature method is `fbcsp`
- FBCSP is computed per band and stored inside the saved model pipeline
- Set `features.method: csp` if you want a single-band CSP baseline
- Set `features.method: psd` if you want the older Welch/log-bandpower path instead
- Current FBCSP setup is implemented as per-band one-vs-rest CSP with optional mutual-information selection, so it works best when you retrain on the same class layout used in config

### Faster calibration
- `training.window_length_sec` controls the model input window length
- `training.crop_enabled: true` turns on sliding-window augmentation during offline training
- `training.crop_step_sec` controls the crop stride
- For lower latency, keep the model window shorter than the recorded imagery interval and let cropping reuse the longer recording

### Shorter recording preset
- Recommended faster preset for a good speed/reliability trade-off:
    - `baseline_duration: 1.5`
    - `cue_duration: 0.8`
    - `imagery_duration: 3.0`
    - `iti_duration: 1.5`
    - `trials_per_class: 36`
- This reduces one trial from 9.0 s to 6.8 s while still leaving enough imagery time for a 2 s FBCSP window.
- If you want an even shorter session, lower `trials_per_class` first before shrinking `imagery_duration` below 3 s.

### Custom Configuration with 3 Classes
1. Create `config/config_custom.yaml`:
```yaml
paradigm:
  classes:
    ACTION_A: 1
    ACTION_B: 2
    ACTION_C: 3

# ... rest of config
```

2. Run with custom config:
```bash
export EEG_CONFIG_PATH=config/config_custom.yaml
python run_app.py
```

## Environment Variables

Override default behavior with environment variables:

```bash
# Use custom config file
export EEG_CONFIG_PATH=/path/to/config.yaml

# Override LSL stream resolution timeout
export EEG_LSL_TIMEOUT=15.0

# Use custom trained model
export EEG_MODEL_PATH=/path/to/model.joblib

# Set logging level
python run_app.py offline -f data.edf -l DEBUG
```

## File Formats

### Input: EDF/BDF EEG Files
- Supported formats for offline training
- Must contain EEG channels (marked as `eeg` type)
- Any channel count supported

### Output: Trained Models
- Format: joblib pickle
- Location: `models/model_latest.joblib`
- Includes:
  - Scaler (StandardScaler)
  - Classifier (LDA or SVM)
  - Feature dimensions (for validation)

### Logs
- Location: `logs/eeg_app.log`
- Format: Rotating file handler (10MB max, 5 backups)
- Contains all events, errors, and predictions

## Troubleshooting

### "No LSL stream found"
- Ensure LabRecorder or similar is running
- Check stream name/type in config
- Increase timeout: `export EEG_LSL_TIMEOUT=20.0`

### "Feature mismatch: model expects N features, got M"
- You trained on different channel count than now running
- Train a new model with current EEG setup
- Or ensure same channel count for online mode

### "Configuration validation failed"
- Check config.yaml syntax (YAML indentation matters)
- Ensure h_freq > l_freq
- Ensure test_size between 0.01 and 0.99
- See [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

## Modes Explained

### Record Mode
- Displays visual stimuli (dynamically positioned based on class count)
- Streams event markers via LSL
- You record EEG using LabRecorder or similar
- Produces synchronized EEG + marker data
- Shows pre-recording instructions before the first trial and allows stopping with `ESC`

### Train Mode (Offline Analysis)
1. Load EEG file (any channel count)
2. Detect events from CSV or EDF stimulus channel
3. Create epochs from events
4. Apply notch + band-pass filter, then optional CAR
5. Crop the epochs into shorter model windows and optionally augment them with overlap
6. Learn FBCSP features per band, or use CSP/PSD fallback if configured
7. Train LDA/SVM classifier
8. Evaluate on test set (20% by default)
9. Save model to `models/model_latest.joblib`

### Online BCI Mode
1. Connect to EEG LSL stream
2. Load trained model
3. Buffer incoming data
4. Feed raw epoch windows into the saved pipeline
5. Predict class and log result
6. Continue until Ctrl+C

## Configuration Details

### Paradigm (Flexible N-Class)

The paradigm automatically handles any number of classes:

```
2-class:    LEFT vs RIGHT
4-class:    UP, DOWN, LEFT, RIGHT (classic)
6-class:    HAND_L, HAND_R, FOOT_L, FOOT_R, TONGUE, REST
N-class:    Any N labels with unique codes
```

Stimuli are positioned on a circle:
- Number of positions = number of classes
- Angle = 2π × class_index / n_classes
- Easily scalable without code changes

### Feature Extraction

For each epoch and frequency band:
1. Compute power spectral density (Welch method)
2. Average PSD within band
3. Log-transform for stability
4. Stack across channels

Feature matrix shape: `(n_epochs, n_channels × n_bands)`

Example with 8 channels and 2 bands:
- Each epoch → 8 × 2 = 16 features
- Fully adaptive to channel count

### Classifier

Supported algorithms:
- **LDA** (Linear Discriminant Analysis) - Recommended for BCI
- **SVM** (Support Vector Machine) - Alternative

Always includes:
- StandardScaler for normalization
- Train/test split with stratification

## Development

### Adding Custom Features

To add new features beyond bandpower:

1. Create function in `features.py`:
```python
def compute_my_features(epochs, param):
    # ... compute features ...
    return X, y, feature_names
```

2. Call in `offline_analysis.py` and `online_bci.py`

### Adding Custom Paradigms

To create non-motor-imagery paradigms:

1. Create paradigm class in `stimuli/`:
```python
from stimuli.paradigm_base import Paradigm

class MyParadigm(Paradigm):
    def __init__(self, config, marker_outlet):
        super().__init__(config, marker_outlet)
        # Custom setup
    
    def run(self):
        # Custom stimulus logic
```

2. Use in `main.py` or `gui_app_v2.py`

## Logs and Debugging

### View Live Logs

During execution, watch logs in `logs/eeg_app.log`:

```bash
tail -f logs/eeg_app.log
```

### Set Debug Level

```bash
python run_app.py offline -f data.edf -l DEBUG
```

Produces verbose output including:
- Config loading
- Channel detection
- Feature extraction
- Classification results

## Citation

If you use this system in research, please cite:

```
EEG Motor Imagery BCI Application v1.0
https://github.com/...
```

## License

MIT (Modify and use freely)

## Contact

For issues or suggestions: [your contact info]
