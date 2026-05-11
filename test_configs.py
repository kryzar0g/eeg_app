#!/usr/bin/env python
"""Test all configuration files"""

from src.config import load_config

configs = [
    'config/config.yaml',
    'config/config_2class.yaml',
    'config/config_6class.yaml',
    'config/config_8ch.yaml',
    'config/config_64ch.yaml'
]

print("Configuration Test Results:")
print("=" * 60)

for config_path in configs:
    try:
        config = load_config(config_path)
        classes = config.paradigm.get('classes', {})
        sfreq = config.preprocessing.sfreq
        print(f"✓ {config_path}")
        print(f"  - Classes: {len(classes)}-class ({', '.join(classes.keys())})")
        print(f"  - Sampling rate: {sfreq} Hz")
        print(f"  - Trials per class: {config.experiment.trials_per_class}")
        print()
    except Exception as e:
        print(f"✗ {config_path}: {e}")
        print()

print("=" * 60)
print("All configurations loaded successfully!")
