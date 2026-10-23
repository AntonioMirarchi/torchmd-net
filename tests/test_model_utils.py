# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from pytest import mark, raises
import torch
from torch.autograd import grad
from torchmdnet.models.model import create_model
from torchmdnet.models.utils import OptimizedDistance
from utils import load_example_args


@mark.parametrize("cutoff_lower", [0, 2])
@mark.parametrize("cutoff_upper", [5, 10])
@mark.parametrize("return_vecs", [False, True])
@mark.parametrize("loop", [False, True])
def test_distance_calculation(cutoff_lower, cutoff_upper, return_vecs, loop):
    dist = OptimizedDistance(
        cutoff_lower,
        cutoff_upper,
        max_num_pairs=-100,
        return_vecs=return_vecs,
        loop=loop,
    )

    batch = torch.tensor([0, 0])

    loop_extra = len(batch) if loop else 0

    # two atoms, distance between lower and upper cutoff
    pos = torch.tensor(
        [[0, 0, 0], [(cutoff_lower + cutoff_upper) / 2, 0, 0]], dtype=torch.float
    )
    edge_index, edge_weight, edge_vec = dist(pos, batch)
    assert (
        edge_index.size(1) == 2 + loop_extra
    ), "Didn't return right number of neighbors"

    # check return_vecs
    if return_vecs:
        assert (
            edge_vec is not None
        ), "Edge vectors were requested but Distance returned None"

    # two atoms, distance lower than lower cutoff
    if cutoff_lower > 0:
        pos = torch.tensor([[0, 0, 0], [cutoff_lower / 2, 0, 0]], dtype=torch.float)
        edge_index, edge_weight, edge_vec = dist(pos, batch)
        assert edge_index.size(1) == loop_extra, "Returned too many neighbors"

    # two atoms, distance larger than upper cutoff
    pos = torch.tensor([[0, 0, 0], [cutoff_upper + 1, 0, 0]], dtype=torch.float)
    edge_index, edge_weight, edge_vec = dist(pos, batch)
    assert edge_index.size(1) == loop_extra, "Returned too many neighbors"

    # check large number of atoms
    batch = torch.zeros(100, dtype=torch.long)
    pos = torch.rand(100, 3)
    edge_index, edge_weight, edge_vec = dist(pos, batch)

    loop_extra = len(batch) if loop else 0

    if cutoff_lower > 0:
        assert edge_index.size(1) == loop_extra, "Expected only self loops"
    else:
        assert edge_index.size(1) == (
            len(batch) * (len(batch) - 1) + loop_extra
        ), "Expected all neighbors to match"


def test_neighbor_count_error():
    dist = OptimizedDistance(0, 5, max_num_pairs=-32)

    # single molecule that should produce an error due to exceeding
    # the maximum number of neighbors
    pos = torch.rand(100, 3)
    batch = torch.zeros(pos.size(0), dtype=torch.long)

    with raises(RuntimeError):
        dist(pos, batch)

    # example data where the second molecule should produce an error
    # due to exceeding the maximum number of neighbors
    pos = torch.rand(100, 3)
    batch = torch.tensor([0] * 20 + [1] * 80, dtype=torch.long)

    with raises(RuntimeError):
        dist(pos, batch)


def test_gated_eq_gradients():
    model = create_model(
        load_example_args(
            "equivariant-transformer", prior_model=None, cutoff_upper=5, derivative=True
        )
    )

    # generate example where one atom is outside the cutoff radius of all others
    z = torch.tensor([1, 1, 8])
    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=torch.float)

    _, forces = model(z, pos)

    # compute gradients of forces with respect to the model's emebdding weights
    deriv = grad(forces.sum(), model.representation_model.embedding.weight)[0]
    assert (
        not deriv.isnan().any()
    ), "Encountered NaN gradients while backpropagating the force loss"


@mark.parametrize("lower,upper", [(0.0, 5.0), (2.0, 7.0)])
@mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polynomial_cutoff(lower, upper, dtype):
    """Check values and analytic derivatives, including both constant extensions."""
    from torchmdnet.models.utils import PolynomialCutoff

    cutoff = PolynomialCutoff(lower, upper)
    t = torch.tensor([-0.1, 0, 0.25, 0.5, 0.75, 1, 1.1], dtype=dtype)
    distances = (lower + (upper - lower) * t).requires_grad_()
    values = cutoff(distances)
    expected = torch.tensor([1, 1, 0.896484375, 0.5, 0.103515625, 0, 0], dtype=dtype)
    torch.testing.assert_close(values, expected)
    first = grad(values.sum(), distances, create_graph=True)[0]
    second = grad(first.sum(), distances)[0]
    t = t.clamp(0, 1)
    torch.testing.assert_close(
        first, -30 * t.square() * (1 - t).square() / (upper - lower)
    )
    torch.testing.assert_close(
        second, -60 * t * (1 - t) * (1 - 2 * t) / (upper - lower) ** 2
    )
    torch.testing.assert_close(torch.jit.script(cutoff)(distances), values)

    # Dense samples catch cancellation or negative values near the upper bound.
    distances = torch.linspace(lower, upper, 10001, dtype=dtype)
    values = cutoff(distances)
    assert ((values >= 0) & (values <= 1 + torch.finfo(dtype).eps)).all()
    # Near the flat lower endpoint, successive values differ by rounding noise.
    assert (values.diff() <= 4 * torch.finfo(dtype).eps).all()


@mark.parametrize(
    "bounds", [(-1, 5), (5, 5), (6, 5), (0, float("inf")), (float("nan"), 5)]
)
def test_polynomial_cutoff_invalid_bounds(bounds):
    """Reject undefined switching intervals at construction."""
    from torchmdnet.models.utils import PolynomialCutoff

    with raises(ValueError, match="Cutoff bounds"):
        PolynomialCutoff(*bounds)
