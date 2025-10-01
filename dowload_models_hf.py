import json
import os
import shutil
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

def create_and_save_magic_pdf_config(model_dir: Path, layoutreader_model_dir: Path, home_dir: Path):
    config_file_name = 'magic-pdf.json'
    config_file = home_dir / config_file_name

    config_content = {
        "bucket_info": {}, 
        "models-dir": str(model_dir), 
        "layoutreader-model-dir": str(layoutreader_model_dir), 
        "device-mode": "cuda",
        "layout-config": {
            "model": "doclayout_yolo"
        },
        "formula-config": {
            "mfd_model": "yolo_v8_mfd",
            "mfr_model": "unimernet_small",
            "enable": True
        },
        "table-config": {
            "model": "rapid_table",
            "enable": True,
            "max_time": 400
        },
        "config_version": "1.0.0" 
    }

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_content, f, ensure_ascii=False, indent=4)
        print(f'The configuration file has been configured successfully at: {config_file}')
    except Exception as e:
        print(f"ERROR: Could not save magic-pdf.json to {config_file}: {e}")
        print("Please check directory permissions.")


if __name__ == '__main__':
    print("--- STARTING MODEL DOWNLOAD AND CONFIGURATION ---")

    mineru_patterns = [
        # "models/Layout/LayoutLMv3/*", # Commented out in original script
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
        # "models/TabRec/TableMaster/*", # Commented out in original script
        # "models/TabRec/StructEqTable/*", # Commented out in original script
    ]
    
    home_dir = Path(os.path.expanduser('~'))

    print("\nDownloading main MinerU models from 'opendatalab/PDF-Extract-Kit-1.0'...")
    try:
        model_root_dir_cache = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns)
        model_dir = Path(model_root_dir_cache) / 'models' # Đường dẫn thực tế của các models chính
        print(f'MinerU models downloaded to: {model_dir}')
    except Exception as e:
        print(f"ERROR: Failed to download MinerU models: {e}")
        print("Please check your internet connection or Hugging Face Hub access.")
        exit(1)


    layoutreader_pattern = [
        "*.json",
        "*.safetensors",
    ]
    print("\nDownloading LayoutReader models from 'hantian/layoutreader'...")
    try:
        layoutreader_model_dir_cache = snapshot_download('hantian/layoutreader', allow_patterns=layoutreader_pattern)
        layoutreader_model_dir = Path(layoutreader_model_dir_cache)
        print(f'LayoutReader models downloaded to: {layoutreader_model_dir}')
    except Exception as e:
        print(f"ERROR: Failed to download LayoutReader models: {e}")
        print("Please check your internet connection or Hugging Face Hub access.")
        exit(1)

    print("\nCreating magic-pdf.json configuration file...")
    create_and_save_magic_pdf_config(model_dir, layoutreader_model_dir, home_dir)
    print("--- MODEL DOWNLOAD AND CONFIGURATION COMPLETE ---")