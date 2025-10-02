import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from pipeline.ssl import train_mae1d


def test_train_mae1d_cpu_handles_large_batches():
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
