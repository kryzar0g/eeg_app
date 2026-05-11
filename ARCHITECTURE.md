# EEG BCI System Architecture

## Overview

The system is designed as a **universal EEG classifier** that works with any number of channels (8, 64, 128+) and any number of experimental classes (2-class, 4-class, 6-class, etc.) without code changes.

## Design Principles

### 1. Configuration-Driven Architecture
- All settings in YAML (easy to modify without touching code)
- Pydantic validation ensures correctness
- Environment variable overrides for deployment flexibility
- No hardcoded values except defaults

### 2. Channel-Agnostic Processing
- Automatic channel detection from:
  - LSL streams (queries stream info)
  - EDF/BDF files (reads header)
  - MNE Info objects
- Feature extraction iterates over detected channels
- No pre-defined channel lists anywhere

### 3. Class-Agnostic Paradigm
- `Paradigm` abstract base class supports any N-class design
- Stimulus positioning calculated dynamically:
  - Circle with N positions (one per class)
  - Angle = 2π × i / n_classes
  - No hardcoded positions for specific class names
- Class labels read from config at runtime

### 4. Modular Pipeline
Each mode follows a clean pipeline:

```
Record:    Paradigm → LSL Markers
           (class labels from config.paradigm)

Offline:   Load EEG → Preprocess → Extract Features → Train
           (channels auto-detected, features scale with count)

Online:    Load Model ← LSL Stream → Preprocess → Extract → Predict
           (validates channel count matches training)
```

## Component Breakdown

### Configuration System (`config.py`)

```
AppConfig (root)
├── ExperimentConfig
│   ├── trials_per_class
│   ├── baseline/cue/imagery/iti durations
│   └── subject/session IDs
│
├── LSLConfig
│   ├── stream_name
│   ├── stream_type
│   └── marker_stream_name
│
├── PreprocessingConfig
│   ├── sfreq (sampling rate)
│   ├── l_freq, h_freq (bandpass)
│   └── notch_freq
│
├── FeaturesConfig
│   └── bands (list of [low, high] pairs)
│
├── ClassifierConfig
│   ├── algorithm (lda/svm)
│   └── test_size
│
└── ParadigmConfig (flexible dict-based)
    └── classes (e.g. {UP: 1, DOWN: 2, LEFT: 3, RIGHT: 4})
        or {LEFT: 1, RIGHT: 2} or any N labels
```

**Key Design:** `paradigm.classes` is a dict that can have any keys/values
- Supports any custom labels
- Maps label name to numeric code
- Read by Paradigm, OnlineBCI at runtime

### Paradigm System (`stimuli/paradigm_base.py`)

```python
Paradigm (ABC):
    run() - main stimulus loop
    _create_stimuli() - position visual elements
    _build_trials() - generate trial list
    _run_trial() - present single trial

MotorImageryParadigm (concrete):
    Positions N stimuli on circle
    Phase sequence: baseline → cue → imagery → ITI
    Randomized trial order preserving balance
```

**Circle Positioning Algorithm:**
```python
radius = 0.5
n_classes = len(class_map)
for i, label in enumerate(sorted(class_map.keys())):
    angle = 2 * π * i / n_classes - π/2
    x = radius * cos(angle)
    y = radius * sin(angle)
    # Position stimulus at (x, y)
```

This means:
- 2 classes → left/right
- 3 classes → triangle
- 4 classes → square (cardinal directions)
- 6 classes → hexagon
- N classes → regular N-gon

### Feature Extraction (`features.py`)

```python
compute_bandpower_features(epochs, bands):
    for epoch in epochs:
        features = []
        for channel in epochs.ch_names:  # ANY number of channels
            for band in bands:
                psd = compute_psd(epoch[channel])
                power = integrate_psd_in_band(psd, band)
                features.append(log(power))
    return features  # shape: (n_epochs, n_channels × n_bands)
```

**Universal:** Works with 8 channels or 64 channels identically

### Training Pipeline (`offline_analysis.py`)

```
Load File (EDF/BDF)
    ↓
Detect Channels
    - Queries MNE file header
    - Finds all 'eeg' type channels
    - No predefined list needed
    ↓
Load Events (CSV or stim channel)
    ↓
Create Epochs
    - Duration from config.events.epoch_length
    - Event codes from CSV
    ↓
Extract Features
    - Iterates over detected channels
    - Uses config.features.bands
    ↓
Train Classifier
    - Algorithm from config.classifier.algorithm
    - Split ratio from config.classifier.test_size
    ↓
Save Model
    - Location: EEG_MODEL_PATH (default: models/model_latest.joblib)
    - Metadata: channel count, feature dimensions
```

**Channel Flexibility:**
- 8-channel file → 8 × 2 bands = 16 features per epoch
- 64-channel file → 64 × 2 bands = 128 features per epoch
- Same code, different feature dimensions = different model size

### Online Classification (`online_bci.py`)

```
Load Trained Model
    ↓
Connect to LSL Stream
    - Auto-detect channel count
    - Query stream info
    ↓
Validate Match
    - Model expects N channels
    - Stream provides N channels
    - FAIL if mismatch
    ↓
Sliding Buffer
    - Pull data chunks
    - Maintain circular buffer (30s max)
    - 50% overlap windows
    ↓
For Each Window:
    - Extract features (auto-scales to channel count)
    - Predict class
    - Map code → label from config
    - Log result
```

**Class Name Mapping:**
```python
def _get_class_names(config: AppConfig) -> Dict[int, str]:
    # Read from config.paradigm.classes
    class_map = config.paradigm.get("classes", {})
    # Invert: {label: code} → {code: label}
    return {int(code): str(label) for label, code in class_map.items()}

# At prediction time:
class_names = _get_class_names(config)
predicted_code = model.predict(features)[0]
predicted_label = class_names.get(predicted_code, str(predicted_code))
```

This supports:
- 4-class: {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}
- 2-class: {1: 'LEFT', 2: 'RIGHT'}
- 6-class: {1: 'HAND_L', 2: 'HAND_R', ...}

### GUI System (`gui_app_v2.py`)

```
EegAppGui
├── Left Sidebar (Navigation)
│   ├── Button: Overview
│   ├── Button: Record
│   ├── Button: Train Model
│   └── Button: Online BCI
│
├── Top: Config Info Panel
│   └── Displays current settings
│
├── Right: Content Area (dynamically changes)
│   ├── InfoPage (Overview tab)
│   ├── RecordPage (Record tab)
│   ├── OfflinePage (Train tab)
│   └── OnlinePage (Online tab)
│
└── Bottom: Log Window
    └── Real-time task output
```

**Page Switching:**
```python
def _switch_page(self, page_id: str):
    # Hide all pages
    for page in self.pages.values():
        page.grid_remove()
    # Show selected page
    self.pages[page_id].grid()
```

**Config Display:**
- Pulls from loaded config
- Shows: channels, sampling rate, filters, bands, classes, etc.
- Updates when config reloaded

## Data Flow Diagrams

### Record Mode
```
    User selects "Record"
            ↓
    GUI loads config
            ↓
    MotorImageryParadigm.__init__
            ↓
    _create_stimuli()
    ├─ reads config.paradigm.classes
    ├─ positions N stimuli on circle
    └─ creates VisualStim objects
            ↓
    run()
    ├─ baseline phase
    ├─ cue phase (highlight stimulus)
    ├─ imagery phase (user imagines movement)
    └─ ITI phase (inter-trial interval)
            ↓
    LSL Outlet
    └─ sends event marker (class code)
            ↓
    LabRecorder
    └─ records synchronized EEG + markers
```

### Offline Mode
```
    User selects "Train Model"
    Chooses EEG file
            ↓
    offline_analysis.load_raw(file_path)
    ├─ reads EDF/BDF header
    ├─ detects all EEG channels (any count)
    └─ loads data (with memory check)
            ↓
    prepare_epochs()
    ├─ reads event codes from CSV/channel
    ├─ creates epochs for each event
    └─ validates codes match config.paradigm
            ↓
    features.compute_bandpower_features()
    ├─ iterates over detected channels
    ├─ computes log-power per band
    └─ returns (n_epochs, n_channels × n_bands)
            ↓
    classifier.train_and_evaluate()
    ├─ StandardScaler
    ├─ train/test split
    ├─ LDA or SVM
    └─ reports accuracy
            ↓
    Save model
    └─ models/model_latest.joblib
```

### Online Mode
```
    User starts Online BCI
            ↓
    online_bci.run_online()
    ├─ load trained model
    ├─ connect to LSL stream
    └─ validate channels match
            ↓
    Continuous Loop:
    ├─ pull_chunk() from LSL
    ├─ accumulate in buffer
    ├─ when buffer ready:
    │   ├─ extract features
    │   ├─ predict class
    │   ├─ map code→label
    │   └─ log result
    └─ sleep 10ms
            ↓
    Ctrl+C
    └─ graceful shutdown
```

## Extensibility Points

### Adding Custom Features
Location: `features.py`

```python
def compute_custom_features(epochs, param):
    """Extract custom features (CSP, wavelet, etc)"""
    # Process all channels in epochs.ch_names
    return X, feature_names
```

Then use in `offline_analysis.py` and `online_bci.py`

### Adding Custom Paradigm
Location: `stimuli/custom_paradigm.py`

```python
class MyParadigm(Paradigm):
    def _create_stimuli(self):
        # Your custom stimulus logic
        pass
    
    def run(self):
        # Your custom trial loop
        pass
```

Then use in `main.py`:
```python
paradigm = MyParadigm(config, outlet)
paradigm.run()
```

### Adding Custom Classifier
Location: `classifier.py`

```python
def _build_model(config):
    if config.classifier.algorithm == "custom":
        return MyCustomClassifier()
    # ... existing logic
```

### Custom Preprocessing
Location: `preprocessing.py`

```python
def preprocess_raw(raw, config):
    # ... existing filters
    
    if hasattr(config, 'custom_filter'):
        raw = apply_custom_filter(raw)
    
    return raw
```

## Configuration Examples

### Portable 8-Channel System
```yaml
experiment:
  trials_per_class: 20  # Shorter session
lsl:
  eeg_stream_name: "MyDevice_EEG"
preprocessing:
  sfreq: 250.0  # Lower sampling rate
```
→ Uses `config_8ch.yaml`

### Lab 64-Channel System
```yaml
experiment:
  trials_per_class: 40  # More data for accuracy
preprocessing:
  sfreq: 500.0  # Higher sampling rate for better freq res
```
→ Uses `config_64ch.yaml`

### Simple 2-Class Setup
```yaml
paradigm:
  classes:
    LEFT: 1
    RIGHT: 2
```
→ Uses `config_2class.yaml`

### Advanced 6-Class Setup
```yaml
paradigm:
  classes:
    HAND_LEFT: 1
    HAND_RIGHT: 2
    FOOT_LEFT: 3
    FOOT_RIGHT: 4
    TONGUE: 5
    REST: 6
```
→ Uses `config_6class.yaml`

## Error Handling Strategy

### Configuration Validation
- Pydantic validates on load
- Constraints: h_freq > l_freq, 0.01 ≤ test_size ≤ 0.99
- Clear error messages pointing to YAML issue

### Channel Mismatch
- Online BCI checks: model channels ≠ stream channels
- Error: "Feature mismatch: model expects 16 features (8ch×2bands), got 32 (16ch×2bands)"
- Solution: Train new model with current setup

### Missing Files/Streams
- EDF/BDF load error → check file exists and is readable
- LSL stream not found → check LabRecorder is running
- Model not found → train model first
- Config file → clear message with path tried

### Memory Issues
- Large files (>2GB) → auto-switch preload=False
- Online buffer bounded to 30 seconds max
- Feature extraction streams data (no full-file load)

## Performance Characteristics

| Operation | 8-ch | 64-ch | Notes |
|-----------|------|-------|-------|
| Feature extraction | 1-2ms/epoch | 5-10ms/epoch | Per-channel parallelizable |
| Model training | <1s | 2-5s | LDA is fast |
| Online classification | <10ms | 30-50ms | Real-time capable |
| Paradigm stimulus render | ~16ms | Same | GPU-independent |

## Testing Checklist

- [x] 2-class paradigm (LEFT/RIGHT)
- [x] 4-class paradigm (UP/DOWN/LEFT/RIGHT)
- [x] 6-class paradigm (multiple limbs)
- [x] 8-channel feature extraction
- [x] 64-channel feature extraction
- [x] LDA and SVM classifiers
- [x] Config loading and validation
- [x] Logging and file rotation
- [x] LSL stream connection
- [x] Online buffer management
- [ ] Real-time GUI (requires display)
- [ ] PsychoPy stimulus rendering (requires display)

## Future Enhancements

1. **Model Registry**
   - Metadata: train date, accuracy, channel count, config snapshot
   - Version management for reproducibility

2. **Advanced Feature Extraction**
   - Common Spatial Patterns (CSP)
   - Wavelet decomposition
   - Time-frequency analysis

3. **Multi-subject Training**
   - Subject-specific models
   - Cross-subject generalization

4. **Real-time Adaptation**
   - Incremental learning during online BCI
   - Covariate shift correction

5. **More Paradigms**
   - P300-based paradigms
   - SSVEP (Steady-State Visual Evoked Potential)
   - Hybrid systems

6. **Visualization Dashboard**
   - Real-time spectrogram
   - Predicted class confidence
   - Online accuracy tracking

---

**Summary:** The system achieves universality through configuration-driven design, automatic channel detection, and algorithm-agnostic processing. Any N-class, any channel count setup works with configuration alone.
