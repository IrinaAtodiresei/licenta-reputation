import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None:
    print("Torch NOT installed")
    exit(1)
else:
    import torch
    print("Torch version:", torch.__version__)
    exit(0)
