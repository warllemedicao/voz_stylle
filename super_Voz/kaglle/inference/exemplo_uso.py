#!/usr/bin/env python3
from pathlib import Path

from inference import validate_package


package_dir = Path(__file__).resolve().parents[1]
paths = validate_package(package_dir)

print("Checkpoint:", paths["checkpoint"])
print("Config:", paths["config"])
print("Manifest:", paths["manifest"])
print("Config do pacote:", paths["package_config"])
print("Tokenizer config:", paths["tokenizer_config"])
print("Audio de referencia:", paths["reference_audio"])
