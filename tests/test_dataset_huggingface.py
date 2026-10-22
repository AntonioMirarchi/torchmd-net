# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import json

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

hf = pytest.importorskip("datasets")
from torchmdnet.data import DataModule
from torchmdnet.datasets import HFDataset
from utils import load_example_args


@pytest.fixture
def parquet_file(tmp_path):
    """Provide small local Parquet fixtures with configurable schemas and row values."""

    def write(name="samples.parquet", count=8, charges=True, **columns):
        """Write three-atom conformations, applying supplied column overrides last."""
        data = {
            "atomic_numbers": [[1, 6, 8]] * count,
            "positions": [[[i, 0, 0], [i, 1, 0], [i, 0, 1]] for i in range(count)],
            "forces": [[[0.1, 0, 0]] * 3] * count,
            "mol_id": ["molecule"] * count,
            "conf_id": list(range(count)),
        }
        if charges:
            data["pq"] = [[0.5, -0.25, -0.25]] * count
        data.update(columns)
        path = tmp_path / name
        hf.Dataset.from_dict(data).to_parquet(path)
        return path

    return write


def test_hf_processing_and_batching(tmp_path, parquet_file):
    """Keep atom fields aligned after masking and preserve PyG batching/transforms."""
    path = parquet_file(box=[np.eye(3).tolist()] * 8)
    dataset = HFDataset(
        tmp_path / "cache", path, stride=2, mask_hydrogen=True, use_box=True
    )
    assert len(dataset) == 4
    sample = dataset[1]
    assert sample.z.dtype == torch.long
    assert sample.z.tolist() == [6, 8]
    assert sample.pos.dtype == sample.neg_dy.dtype == torch.float32
    assert sample.pos[:, 0].tolist() == [2, 2]
    assert sample.pq.tolist() == [-0.25, -0.25]
    assert sample.q.item() == 0  # Includes the charge of removed hydrogen atoms.
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch.pos.shape == batch.neg_dy.shape == (4, 3)
    assert batch.pq.shape == (4,)
    assert batch.q.shape == (2,)
    assert batch.box.shape == (2, 3, 3)
    nonperiodic = HFDataset(tmp_path / "cache", path, stride=2, mask_hydrogen=True)
    assert "box" not in nonperiodic[0]
    assert nonperiodic.processed_paths != dataset.processed_paths
    assert "box" not in nonperiodic.dataset.column_names
    # Use PyG's native slicing and transform behavior.
    assert dataset[[2, 0]][0].pos[0, 0] == 4

    def shift(data):
        """Make transform application observable without changing the cached sample."""
        data.pos += 1
        return data

    dataset.transform = shift
    assert dataset[0].pos[0, 0] == 1


@pytest.mark.parametrize("max_gradient", [None, 1.0])
def test_hf_filters(tmp_path, parquet_file, max_gradient):
    """Reject nonfinite forces and apply force/atom limits before hydrogen removal."""
    forces = [[[value, 0, 0]] * 3 for value in (0.1, np.nan, np.inf, 1, 2, 0.2)]
    path = parquet_file(count=6, forces=forces)
    dataset = HFDataset(tmp_path / "cache", path, max_gradient=max_gradient)
    assert len(dataset) == (4 if max_gradient is None else 2)
    with pytest.raises(ValueError, match="No conformations remain"):
        HFDataset(tmp_path / "cache", path, max_num_atoms=2, mask_hydrogen=True)


def test_hf_charge_precedence(tmp_path, parquet_file):
    """An explicit molecular charge takes precedence over summed partial charges."""
    path = parquet_file(q=[2.0] * 8)
    dataset = HFDataset(tmp_path / "cache", path, mask_hydrogen=True)
    assert dataset[0].q == 2


def test_hf_json_overrides_and_partial_cache(tmp_path, parquet_file, monkeypatch):
    """Per-file overrides and partial cache reuse must preserve input ordering."""
    first = parquet_file("a.parquet", count=6)
    second = parquet_file("b.parquet", count=6)
    root = tmp_path / "cache"
    # Only the second shard is cached initially. This must not reorder samples.
    HFDataset(root, second, stride=2)
    config = tmp_path / "inputs.json"
    config.write_text(
        json.dumps({"data": [{"path": first.name, "stride": 3}, {"path": second.name}]})
    )
    dataset = HFDataset(root, config, stride=2)
    assert [dataset[i].pos[0, 0].item() for i in range(len(dataset))] == [0, 3, 0, 2, 4]

    def no_processing(*args):
        """Fail if loading a completed cache attempts preprocessing again."""
        pytest.fail("Existing caches must not be processed again")

    monkeypatch.setattr("torchmdnet.datasets.huggingface._process_shard", no_processing)
    cached = HFDataset(root, config, stride=2)
    assert len(cached) == len(dataset)
    assert cached.processed_paths == dataset.processed_paths
    assert isinstance(cached.dataset._data, hf.table.ConcatenationTable)


def test_hf_cache_invalidation_and_sampling(tmp_path, parquet_file):
    """Sampling is reproducible; changed inputs or settings select new caches."""
    path = parquet_file(count=10)
    root = tmp_path / "cache"
    original = HFDataset(root, path)
    sampled = HFDataset(root, path, sample_fraction=0.4)
    assert len(sampled) == 4
    repeated = HFDataset(tmp_path / "other-cache", path, sample_fraction=0.4)
    assert [sampled[i].pos.tolist() for i in range(4)] == [
        repeated[i].pos.tolist() for i in range(4)
    ]
    assert original.processed_paths != sampled.processed_paths
    parquet_file(count=12)
    changed = HFDataset(root, path)
    assert len(changed) == 12
    assert original.processed_paths != changed.processed_paths


def test_hf_empty_shard_and_parallel_processing(tmp_path, parquet_file):
    """Parallel processing can combine an empty shard with a different valid schema."""
    empty = parquet_file("empty.parquet", forces=[[[np.nan, 0, 0]] * 3] * 8)
    valid = parquet_file("valid.parquet", charges=False)
    dataset = HFDataset(tmp_path / "cache", [empty, valid], hf_workers=2)
    assert len(dataset) == 8
    assert not dataset.has_q
    assert "q" not in dataset[0]


@pytest.mark.parametrize("options", [{"sample_fraction": 0}, {"mask_hydrogen": True}])
def test_hf_empty_processed_dataset(tmp_path, parquet_file, options):
    """Report a clear error when sampling or hydrogen removal leaves no samples."""
    path = parquet_file(atomic_numbers=[[1, 1, 1]] * 8)
    with pytest.raises(ValueError, match="No conformations remain"):
        HFDataset(tmp_path / "cache", path, **options)


@pytest.mark.parametrize("charges", [True, False])
def test_hf_datamodule(tmp_path, parquet_file, charges):
    """Exercise cache preparation, ordinary sample splits, loaders and charge checks."""
    path = parquet_file(count=10, charges=charges)
    args = load_example_args("graph-network")
    args.update(
        dataset="HFDataset",
        dataset_root=str(tmp_path / "cache"),
        dataset_arg={"paths": str(path)},
        log_dir=str(tmp_path),
        train_size=6,
        val_size=2,
        test_size=2,
        batch_size=2,
        inference_batch_size=2,
        num_workers=0,
        standardize=False,
        charge=charges,
    )
    module = DataModule(args)
    module.prepare_data()
    assert module.dataset is None
    module.setup("fit")
    assert [
        len(module.train_dataset),
        len(module.val_dataset),
        len(module.test_dataset),
    ] == [6, 2, 2]
    assert next(iter(module.train_dataloader())).z.shape == (6,)
    assert (tmp_path / "splits.npz").is_file()
    indices = torch.cat((module.idx_train, module.idx_val, module.idx_test))
    assert sorted(indices.tolist()) == list(range(10))
    assert module.hparams.dataset_arg == {"paths": str(path)}
    if not charges:
        args["charge"] = True
        with pytest.raises(ValueError, match="charge=True requires"):
            DataModule(args).setup("fit")


@pytest.mark.parametrize(
    "options",
    [
        {"stride": 0},
        {"sample_fraction": -0.1},
        {"sample_fraction": 1.1},
        {"hf_workers": 0},
    ],
)
def test_hf_invalid_options(tmp_path, parquet_file, options):
    """Reject invalid processing settings before generating a cache."""
    with pytest.raises(ValueError):
        HFDataset(tmp_path / "cache", parquet_file(), **options)


def test_hf_missing_columns(tmp_path, parquet_file):
    """Report missing molecular fields before streaming Parquet rows."""
    path = tmp_path / "missing.parquet"
    hf.Dataset.from_dict({"positions": [[[0, 0, 0]]]}).to_parquet(path)
    with pytest.raises(ValueError, match="Missing required columns"):
        HFDataset(tmp_path / "cache", path)


def test_hf_column_projection_and_mixed_schemas(tmp_path, parquet_file, monkeypatch):
    """Force-only loading projects required columns despite differing source metadata."""
    first = parquet_file("md.parquet", charges=False, pdb=[b"unused"] * 8)
    second = parquet_file(
        "sm.parquet", partial_charges=[[0.1, 0.2, 0.3]] * 8, smiles=["CO"] * 8
    )
    original_load = hf.load_dataset
    projected = []

    def load(*args, **kwargs):
        """Record the reader projection while retaining real Parquet I/O."""
        projected.append(kwargs["columns"])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(hf, "load_dataset", load)
    dataset = HFDataset(tmp_path / "cache", [first, second], properties=[])
    assert len(dataset) == 16
    columns = {"atomic_numbers", "positions", "forces"}
    assert all(set(selected) == columns for selected in projected)
    assert set(dataset.dataset.column_names) == columns
    assert set(next(iter(DataLoader(dataset, batch_size=16))).keys()) == {
        "z",
        "pos",
        "neg_dy",
        "batch",
        "ptr",
    }


@pytest.mark.parametrize("properties", [("q",), ("pq",), ("q", "pq")])
def test_hf_charge_column_aliases(tmp_path, parquet_file, properties):
    """Normalize charge aliases and derive molecular charge before atom masking."""
    path = parquet_file(
        charges=False,
        partial_charges=[[0.5, -0.25, -0.25]] * 8,
        formal_charges=[[1, 0, 0]] * 8,
    )
    dataset = HFDataset(
        tmp_path / "cache", path, properties=properties, mask_hydrogen=True
    )
    assert ("q" in dataset[0]) == ("q" in properties)
    assert ("pq" in dataset[0]) == ("pq" in properties)
    if "q" in properties:
        assert dataset[0].q == 1
    if "pq" in properties:
        assert dataset[0].pq.tolist() == [-0.25, -0.25]


def test_hf_force_scale_and_per_file_stride(tmp_path, parquet_file):
    """JSON strides override global defaults and force filtering uses converted units."""
    first = parquet_file("first.parquet", count=6)
    second = parquet_file("second.parquet", count=6)
    config = tmp_path / "inputs.json"
    config.write_text(
        json.dumps(
            {
                "data": [
                    {"path": first.name, "stride": 2},
                    {"path": second.name, "stride": 3, "force_scale": 10},
                ]
            }
        )
    )
    dataset = HFDataset(tmp_path / "cache", config, stride=6, properties=[])
    assert [dataset[i].pos[0, 0].item() for i in range(len(dataset))] == [0, 2, 4, 0, 3]
    assert dataset[0].neg_dy[0, 0].item() == pytest.approx(0.1)
    assert dataset[-1].neg_dy[0, 0].item() == pytest.approx(1)
    # The second file is rejected only after force conversion.
    filtered = HFDataset(tmp_path / "cache", config, properties=[], max_gradient=0.5)
    assert len(filtered) == 3
