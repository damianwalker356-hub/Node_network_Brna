from types import SimpleNamespace

import numpy as np
import pytest

from brna.hf_adapter import LocalLoRABackbone


@pytest.mark.req("R-HF-02")
def test_soft_prefix_graph_backpropagates_through_gates_and_context():
    """R-HF-02: preserve gradients through graph edge gates to prefix embeddings."""
    torch = pytest.importorskip("torch")
    backbone = object.__new__(LocalLoRABackbone)
    backbone._torch = torch
    backbone.device = "cpu"
    backbone.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    graph = backbone.create_soft_prefix_graph(
        node_count=3,
        adjacency=np.asarray(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=float,
        ),
        latent_size=4,
        hops=2,
        seed=7,
    )
    contexts = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    contexts.requires_grad_()

    output = graph(contexts)
    output["prefix_embeddings"].square().sum().backward()

    assert output["prefix_embeddings"].shape == (3, 1, 4)
    assert contexts.grad is not None
    assert torch.isfinite(contexts.grad).all()
    assert graph.edge_gates.grad is not None
    assert torch.isfinite(graph.edge_gates.grad).all()


@pytest.mark.req("R-HF-02")
def test_soft_prefix_graph_rejects_context_size_mismatch():
    """R-HF-02: validate the batch-free context shape before graph propagation."""
    torch = pytest.importorskip("torch")
    backbone = object.__new__(LocalLoRABackbone)
    backbone._torch = torch
    backbone.device = "cpu"
    backbone.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    graph = backbone.create_soft_prefix_graph(
        node_count=2,
        adjacency=np.asarray([[0, 1], [1, 0]], dtype=float),
        latent_size=3,
        hops=1,
        seed=8,
    )

    with pytest.raises(ValueError, match="context_states must have shape"):
        graph(torch.zeros((2, 5)))
