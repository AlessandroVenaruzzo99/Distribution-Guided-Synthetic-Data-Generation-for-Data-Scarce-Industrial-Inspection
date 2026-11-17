"""
Lightweight test configuration.

The pipeline modules import torch / torchvision / ultralytics at import time
(through ``utils.py``). The unit tests below only exercise the pure
NumPy / OpenCV / pandas logic, so these heavy packages are replaced with
empty stub modules when they are not installed. No model is ever built,
trained or run by the test-suite, and no GPU is used.
"""
import os
import sys
import types
from pathlib import Path

# Keep the tests single-threaded and CPU only.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_stubs():
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = _stub("torch")
        torch.utils = _stub("torch.utils")
        torch.utils.data = _stub("torch.utils.data", DataLoader=object, Dataset=object)
        torch.cuda = _stub("torch.cuda", manual_seed=lambda *_: None, manual_seed_all=lambda *_: None)
        torch.backends = _stub("torch.backends")
        torch.backends.cudnn = _stub("torch.backends.cudnn")
        torch.manual_seed = lambda *_: None
    try:
        import torchvision  # noqa: F401
    except ImportError:
        _stub("torchvision", transforms=None, datasets=None)
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        _stub("ultralytics", YOLO=None)


_install_stubs()
