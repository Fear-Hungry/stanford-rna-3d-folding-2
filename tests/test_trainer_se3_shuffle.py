from __future__ import annotations

import torch

from rna3d_local.training.trainer_se3 import _epoch_graph_indices


def test_epoch_graph_indices_is_full_permutation() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    order = _epoch_graph_indices(
        graph_count=11,
        generator=generator,
        stage="TEST",
        location="tests/test_trainer_se3_shuffle.py:test_epoch_graph_indices_is_full_permutation",
    )
    assert len(order) == 11
    assert sorted(order) == list(range(11))


def test_epoch_graph_indices_is_reproducible_and_changes_next_epoch() -> None:
    g1 = torch.Generator(device="cpu")
    g2 = torch.Generator(device="cpu")
    g1.manual_seed(77)
    g2.manual_seed(77)
    order_1 = _epoch_graph_indices(
        graph_count=12,
        generator=g1,
        stage="TEST",
        location="tests/test_trainer_se3_shuffle.py:test_epoch_graph_indices_is_reproducible_and_changes_next_epoch",
    )
    order_1_bis = _epoch_graph_indices(
        graph_count=12,
        generator=g2,
        stage="TEST",
        location="tests/test_trainer_se3_shuffle.py:test_epoch_graph_indices_is_reproducible_and_changes_next_epoch",
    )
    assert order_1 == order_1_bis

    order_2 = _epoch_graph_indices(
        graph_count=12,
        generator=g1,
        stage="TEST",
        location="tests/test_trainer_se3_shuffle.py:test_epoch_graph_indices_is_reproducible_and_changes_next_epoch",
    )
    assert sorted(order_2) == list(range(12))
    assert order_2 != order_1
