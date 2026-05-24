#!/usr/bin/env python3
import importlib

modules = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("streamlit", "streamlit"),
]

def try_import(name, modname):
    try:
        m = importlib.import_module(modname)
        ver = getattr(m, "__version__", None)
        print(f"{name}: OK, version={ver}")
        if name == "torch":
            try:
                import torch
                print(f"  cuda_available: {torch.cuda.is_available()}")
            except Exception as e:
                print(f"  torch runtime check failed: {e}")
    except Exception as e:
        print(f"{name}: FAILED -> {e.__class__.__name__}: {e}")

if __name__ == '__main__':
    for name, mod in modules:
        try_import(name, mod)
