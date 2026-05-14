# EEG Configuration Guide

This configuration file defines all parameters for the EEG Motor Imagery BCI system.

## Quick Start Examples

### Example 1: 8-Channel Setup (Portable Device)
```yaml
experiment:
  trials_per_class: 20  # Fewer trials for quick testing
  baseline_duration: 1.5
  cue_duration: 0.8
  imagery_duration: 3.0
  iti_duration: 1.5

preprocessing:
  sfreq: 256.0  # Sampling rate of your device
  l_freq: 8.0
  h_freq: 30.0
  notch_freq: 50.0  # Or 60 Hz in US

features:
  method: "fbcsp"  # "csp", "fbcsp" or "psd"
  bands:
    - [4.0, 8.0]    # Optional lower band for FBCSP
    - [8.0, 12.0]   # Alpha
    - [12.0, 30.0]  # Beta
    - [30.0, 40.0]  # Optional higher band for FBCSP
  csp_components: 2  # Number of CSP components per band
  fbcsp_top_k: 8     # Keep only the best FBCSP features after MI ranking

training:
  window_length_sec: 2.0  # Model input window length
  crop_enabled: true      # Enable sliding-window augmentation in offline training
  crop_step_sec: 0.5      # Overlap stride for crop augmentation

classifier:
  algorithm: "lda"
  test_size: 0.2
```

### Example 2: 64-Channel Setup (Full Montage)
```yaml
experiment:
  trials_per_class: 40  # Standard number
  baseline_duration: 2.0
  cue_duration: 1.0
  imagery_duration: 4.0
  iti_duration: 2.0

preprocessing:
  sfreq: 250.0  # Standard BrainVision rate
  l_freq: 8.0
  h_freq: 30.0
  notch_freq: 50.0

features:
  bands:
    - [8.0, 12.0]   # Alpha
    - [12.0, 30.0]  # Beta

classifier:
  algorithm: "lda"
  test_size: 0.2
```

### Example 3: 2-Class Setup (Left vs Right)
```yaml
paradigm:
  classes:
    LEFT: 1
    RIGHT: 2

# Rest of config as above
```

### Example 4: 6-Class Setup
```yaml
paradigm:
  classes:
    HAND_L: 1
    HAND_R: 2
    FOOT_L: 3
    FOOT_R: 4
    TONGUE: 5
    REST: 6

# Rest of config as above
```

## Configuration Sections

### experiment
Defines trial timings and repetitions.

- `trials_per_class`: Number of trials per class (default: 36 in the short preset)
- `baseline_duration`: Baseline period before cue (seconds)
- `cue_duration`: Visual cue presentation (seconds)
- `imagery_duration`: Motor imagery period (seconds)
- `iti_duration`: Inter-trial interval (seconds)

Total trial duration = baseline + cue + imagery + iti

Recommended short-but-reliable preset:
- `baseline_duration: 1.5`
- `cue_duration: 0.8`
- `imagery_duration: 3.0`
- `iti_duration: 1.5`
- `trials_per_class: 36`

With 4 classes, this reduces session time from about 24 minutes to about 16 minutes, while still leaving enough imagery time for a 2 s model window and crop augmentation.

### lsl
LSL stream parameters (rarely need change).

- `eeg_stream_name`: Name of EEG stream (default: "EEG")
- `eeg_stream_type`: Type of EEG stream (default: "EEG")
- `marker_stream_name`: Name of marker stream (default: "Markers")
- `resolution_timeout`: Timeout for stream resolution (default: 10.0s)

### paradigm
Motor imagery classes and mapping.

- `classes`: Dictionary mapping class labels to event codes
  * Labels can be any string: UP, DOWN, LEFT, RIGHT, ACTION_A, etc.
  * Codes must be unique integers (typically 1, 2, 3, ...)
  * Number of classes should match your experimental design

Example:
```yaml
paradigm:
  classes:
    UP: 1
    DOWN: 2
    LEFT: 3
    RIGHT: 4
```

### preprocessing
EEG signal preprocessing parameters.

- `sfreq`: Sampling frequency in Hz (must match your device!)
- `l_freq`: Lower frequency for band-pass filter (Hz)
- `h_freq`: Upper frequency for band-pass filter (Hz)
- `notch_freq`: Frequency for notch filter, typically 50Hz (Europe) or 60Hz (US)
- `car`: Apply common average reference after filtering (`true`/`false`)

Constraints:
- 0 ≤ l_freq < h_freq
- l_freq ≥ 0, h_freq ≥ 0
- notch_freq > 0

### features
Feature extraction parameters.

- `method`: Feature method used in the saved model pipeline
  * `csp`: Common Spatial Pattern on a single band
  * `fbcsp`: Filter Bank CSP across multiple bands, with optional mutual-information selection
  * `psd`: Legacy Welch/log-bandpower fallback
- `bands`: List of frequency bands for feature extraction
  * Each band: [fmin, fmax]
  * Common bands:
    - [8, 12]: Alpha
    - [12, 30]: Beta
    - [30, 40]: Gamma
  * You can add as many bands as needed
- `csp_components`: Number of CSP components to keep per band
  * Recommended start: 2
  * Higher values increase feature count and runtime slightly
- `fbcsp_top_k`: Number of best FBCSP features to keep after mutual information ranking
  * Set to `0` to keep all features
  * A good start is 8 to 12

### training
Offline training and online model-window settings.

- `window_length_sec`: Input window length used by the model and online inference
- `crop_enabled`: If `true`, the offline trainer generates overlapping crop windows from each epoch
- `crop_step_sec`: Step between crop windows in seconds

Example:
```yaml
features:
  bands:
    - [8.0, 12.0]    # Alpha
    - [12.0, 30.0]   # Beta
    - [30.0, 40.0]   # Gamma
```

### classifier
Model training and evaluation parameters.

- `algorithm`: Classifier type
  * "lda": Linear Discriminant Analysis (recommended for BCI)
  * "svm": Support Vector Machine
- `test_size`: Fraction of data used for testing (0.0-1.0)
  * Common: 0.2 (80% train, 20% test)

Constraints:
- 0.01 ≤ test_size ≤ 0.99

### events
Event detection for offline analysis.

- `mode`: How to detect events
  * "stim": Detect from stimulus channel in EDF/BDF
  * "csv": Use timestamps from CSV file
- `stim_channel`: Name of stimulus channel in EDF/BDF (for mode="stim")
- `tmin`: Start time relative to event (seconds)
- `tmax`: End time relative to event (seconds)
- `timestamp_csv_path`: Path to CSV file with event times (for mode="csv")
- `epoch_length`: Length of epoch for CSV mode (seconds)

## Tips for Different Setups

### Portable/Mobile EEG (8-16 channels)
- Lower `trials_per_class` (15-25) to keep session short
- Use fewer frequency bands
- Consider shorter timings (baseline: 1.0-1.5s, imagery: 2.5-3.5s)

### Faster calibration without losing much robustness
- Keep `imagery_duration` at least 3.0 s if you use a 2.0 s model window
- Prefer lowering `trials_per_class` only after you have enough data for stable validation
- Use crop augmentation so the model sees more training windows from the same recording

### Lab Setup (32+ channels)
- Standard `trials_per_class` (35-50)
- Can use more frequency bands
- Standard timings

### Low Noise Environment
- Can use narrower frequency bands
- Can use smaller `test_size` for more training data

### High Noise Environment
- Use broader frequency bands
- Use larger `test_size` for more robust testing

## Automatic Channel Detection

The system automatically detects the number of EEG channels from:
1. **Online mode**: LSL stream channel count
2. **Offline mode**: EDF/BDF file channel count

No manual configuration needed! The paradigm and classification work with any number of channels.

## Patient Profiles

When the GUI starts, it asks you to either select an existing patient profile or create a new one.

- Profiles are stored as JSON files in `data/patients/`
- Required fields: first name, last name, date of birth, sex
- Optional fields: notes
- The selected profile is shown in the GUI sidebar and logged when recording starts

## Recording Instructions

Before the first trial of record mode, the app shows a short instruction screen:

- Sit still and look at the screen
- Follow the highlighted cue
- In the imagery phase, only imagine the indicated movement
- Use `ESC` to stop the recording at any time

## CSP Notes

The current fast pipeline uses FBCSP as the default feature method.

- CSP/FBCSP is learned during offline training and saved inside the `joblib` pipeline.
- Online mode loads the same pipeline, so feature extraction stays identical.
- The implementation uses a per-band, one-vs-rest strategy to support the current multi-class setup.
- `training.window_length_sec` can be shorter than `experiment.imagery_duration`; cropping reuses the longer recording for more samples.
- If you switch to `features.method: psd`, the application falls back to the older Welch/log-bandpower path.
- Keep `training.window_length_sec`, `preprocessing.sfreq`, and `features.method` aligned between training and online use.

### Session duration estimate

For the default short preset:
- One trial lasts about 6.8 s
- 36 trials per class × 4 classes = 144 trials total
- Total session time is about 16.3 minutes, excluding small pauses between blocks

## Environment Variables

You can override configuration with environment variables:

```bash
# Use custom config file
export EEG_CONFIG_PATH=/path/to/custom/config.yaml

# Override LSL timeout
export EEG_LSL_TIMEOUT=15.0

# Override model path
export EEG_MODEL_PATH=/path/to/trained_model.joblib

# Set logging level
python run_app.py offline -f data.edf -l DEBUG
```

## Validation

Configuration is automatically validated when loaded. Common errors:

- `h_freq must be greater than l_freq`: Check preprocessing bands
- `test_size must be between 0.01 and 0.99`: Check classifier settings
- `paradigm.classes is empty`: Define at least 2 classes
- `Configuration file not found`: Check EEG_CONFIG_PATH

## Next Steps

1. Adjust timings to match your paradigm
2. Set `sfreq` to match your EEG device
3. Define `paradigm.classes` for your motor imagery tasks
4. Choose frequency bands based on your brain activity
5. Select classifier algorithm (LDA recommended for BCI)
