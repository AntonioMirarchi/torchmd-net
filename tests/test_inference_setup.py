# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

"""Inference embedding caches preserve model energies and coordinate gradients."""

import pytest
import torch

from torchmdnet.models.model import create_model
from torchmdnet.scripts.train import get_argparse


@pytest.mark.parametrize("model_name", ["tensornet", "tensornet2"])
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inference_setup_preserves_predictions(model_name, static_shapes, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    args = vars(get_argparse().parse_args([]))
    args.update(
        model=model_name,
        q_dim=2 if model_name == "tensornet2" else 0,
        embedding_dimension=8,
        num_layers=1,
        num_rbf=4,
        max_num_neighbors=16,
        precision=64 if dtype == torch.float64 else 32,
        derivative=False,
        static_shapes=static_shapes,
    )
    model = create_model(args).to(device).eval()
    model.requires_grad_(False)
    z = torch.tensor([1, 6, 8], device=device)
    batch = torch.zeros_like(z)
    q = torch.tensor([0.5], device=device, dtype=dtype)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device=device, dtype=dtype, requires_grad=True,
    )
    energy, _ = model(z, pos, batch=batch, q=q)
    forces = -torch.autograd.grad(energy.sum(), pos)[0]
    original_keys = set(model.state_dict())
    representation = model.representation_model
    representation.setup_for_inference(z, batch)
    embedding = representation.tensor_embedding
    assert representation.inference_mode and embedding.inference_mode
    assert embedding.Zij_map.device == pos.device
    assert embedding.Zij_map.dtype == dtype
    assert not embedding.Zij_map.requires_grad
    assert set(model.state_dict()) == original_keys
    # Include species zero to check padding has the same messages as the eager path.
    zero_pair = embedding.emb2(torch.cat((embedding.emb(z.new_tensor(0)),) * 2))
    torch.testing.assert_close(embedding.Zij_map[0, 0], zero_pair)
    cached_energy, _ = model(z, pos, batch=batch, q=q)
    cached_forces = -torch.autograd.grad(cached_energy.sum(), pos)[0]
    rtol, atol = (1e-9, 1e-10) if dtype == torch.float64 else (1e-4, 1e-5)
    torch.testing.assert_close(cached_energy, energy, rtol=rtol, atol=atol)
    torch.testing.assert_close(cached_forces, forces, rtol=rtol, atol=atol)
    representation.setup_for_inference(z, batch)
    torch.testing.assert_close(model(z, pos, batch=batch, q=q)[0], cached_energy)
