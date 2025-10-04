from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import pytest


def _load_train_mae1d():
    spec = spec_from_file_location(
        "pipeline.ssl", Path(__file__).resolve().parents[1] / "pipeline" / "ssl.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)

    inserted_stub = False
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.DataFrame = object
        pandas_stub.Timestamp = object
        sys.modules["pandas"] = pandas_stub
        inserted_stub = True

    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_stub:
            sys.modules.pop("pandas", None)

    return module.train_mae1d


def test_train_mae1d_cpu_handles_large_batches():
    torch = pytest.importorskip("torch", reason="Requires torch for SSL training test")
    np = pytest.importorskip("numpy", reason="Requires numpy for SSL training test")

    train_mae1d = _load_train_mae1d()
    torch.manual_seed(0)
    np.random.seed(0)

    # Large enough to require multiple inference batches while remaining quick to train.
    sequences = np.random.randn(2048, 3, 64).astype(np.float32)

    model, embeddings = train_mae1d(
        sequences,
        epochs=1,
        batch_size=256,
        lr=1e-3,
        mask_ratio=0.2,
        mask_len=3,
        device="cpu",
    )

    assert next(model.parameters()).device.type == "cpu"
    assert embeddings.shape[0] == sequences.shape[0]
    # Embeddings should reflect the model head size (default hidden width).
    assert embeddings.shape[1] == 64
    assert embeddings.dtype == np.float32
