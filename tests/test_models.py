import pytest
import torch

from src.models import CNNBaseline, CNNMultiHeadAttention


def test_cnn_forward_shape() -> None:
    model = CNNBaseline(cnn_channels=32)
    x = torch.randn(4, 128, 9)
    assert model(x).shape == (4, 6)


def test_cnn_attention_forward_shape() -> None:
    model = CNNMultiHeadAttention(cnn_channels=32, embed_dim=64, num_heads=4)
    x = torch.randn(4, 128, 9)
    assert model(x).shape == (4, 6)


def test_cnn_to_attention_representation() -> None:
    model = CNNMultiHeadAttention(cnn_channels=32, embed_dim=64, num_heads=4)
    x = torch.randn(2, 128, 9)
    sequence, attentions = model.encode(x, return_attention=True)
    assert sequence.shape == (2, 32, 64)
    assert attentions[0].shape == (2, 4, 32, 32)


@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_different_attention_heads(heads: int) -> None:
    model = CNNMultiHeadAttention(embed_dim=64, num_heads=heads)
    x = torch.randn(2, 128, 9)
    assert model(x).shape == (2, 6)


def test_invalid_head_configuration() -> None:
    with pytest.raises(ValueError):
        CNNMultiHeadAttention(embed_dim=30, num_heads=8)
