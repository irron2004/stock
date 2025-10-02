from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from pipeline.ssl import train_mae1d


def test_train_mae1d_handles_partial_batch():
    sequences = np.random.randn(4, 3, 5).astype(np.float32)
    batch_size = 8

    with patch.object(torch.optim.Adam, "step", wraps=torch.optim.Adam.step) as mock_step:
        train_mae1d(sequences, epochs=1, batch_size=batch_size)

    assert mock_step.call_count >= 1
