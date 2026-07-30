import sys

print(f"Running on Python {sys.version}")

packages_to_test = [
    "streamlit",
    "docling",
    "torch",
    "cv2",
    "PIL",
    "numpy",
    "pandas",
    "langchain_core",
    "easyocr",
    "whisper"
]

for package in packages_to_test:
    try:
        __import__(package)
        print(f"[SUCCESS] Successfully imported {package}")
    except Exception as e:
        print(f"[ERROR] Failed to import {package}: {e}")
