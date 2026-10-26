# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import hashlib
import gc
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def _load_mol_ids_to_exclude(value, base):
    """Resolve an ID list or text file to sorted, unique string identifiers."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        path = (base / Path(value).expanduser()).resolve()
        if not path.is_file():
            raise ValueError(
                f"mol_ids_to_exclude file does not exist: {path}. "
                "Supply molecule IDs as a list or a path to a text file."
            )
        with path.open(encoding="utf-8") as f:
            value = f.read().splitlines()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("mol_ids_to_exclude must be an ID list or a text file path.")
    if any(
        not isinstance(item, (str, int, np.integer)) or isinstance(item, bool)
        for item in value
    ):
        raise ValueError("mol_ids_to_exclude entries must be strings or integers.")
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _iter_samples(dataset, options):
    """Yield filtered, normalized rows from a projected HF iterable dataset.

    ``options`` contains the resolved settings for one file. Stride uses original
    row indices; atom-count and scaled-force filters precede hydrogen removal.
    Charge derivation uses all original atoms, even when hydrogens are removed.
    """
    excluded_ids = set(options.get("mol_ids_to_exclude", []))
    # Keep discarded rows as Arrow slices instead of decoding their atom arrays.
    for idx, table in enumerate(dataset.with_format("arrow")):
        if idx % options["stride"]:
            continue
        if excluded_ids:
            mol_id = table["mol_id"][0].as_py()
            if mol_id is not None and str(mol_id).strip() in excluded_ids:
                continue
        sample = table.to_pylist()[0]
        # Identifiers are used only for filtering, not as model input or output.
        sample.pop("mol_id", None)
        if "partial_charges" in sample:
            sample["pq"] = sample.pop("partial_charges")
        z = np.asarray(sample["atomic_numbers"])
        if options["max_num_atoms"] is not None and len(z) > options["max_num_atoms"]:
            continue
        pos = np.asarray(sample["positions"])
        # Compare force magnitudes in the requested output units.
        forces = np.asarray(sample["forces"]) * options["force_scale"]
        if pos.shape != (len(z), 3) or forces.shape != (len(z), 3):
            raise ValueError(f"Invalid positions or forces shape at row {idx}.")
        if len(z) == 0 or not np.isfinite(forces).all():
            continue
        if options["max_gradient"] is not None:
            if np.linalg.norm(forces, axis=1).max() >= options["max_gradient"]:
                continue
        sample["forces"] = forces.tolist()
        if "formal_charges" in sample:
            sample["q"] = float(np.sum(sample.pop("formal_charges")))
        if "pq" in sample:
            if np.asarray(sample["pq"]).shape != z.shape:
                raise ValueError(
                    f"Partial charges must match atomic_numbers at row {idx}."
                )
            if "q" in options["properties"] and sample.get("q") is None:
                # Preserve the molecular charge before removing any atoms.
                sample["q"] = float(np.sum(sample["pq"]))
            if "pq" not in options["properties"]:
                del sample["pq"]
        if options["mask_hydrogen"]:
            keep = z != 1
            if not keep.any():
                continue
            # Keep per-atom fields aligned; molecular charge stays unchanged.
            for key in ("atomic_numbers", "positions", "forces", "pq"):
                if key in sample:
                    sample[key] = np.asarray(sample[key])[keep].tolist()
        yield sample


def _process_shard(task):
    """Build one Arrow cache from a ``(source_path, cache_path, options)`` task.

    Paths are local ``Path`` objects and options are resolved by ``HFDataset``.
    Completed caches are reused. New data is streamed into a temporary directory
    and moved into place only after saving succeeds, including for empty shards.
    """
    import pyarrow.parquet as parquet
    from datasets import (
        Dataset as ArrowDataset,
        Features,
        Sequence,
        Value,
        load_dataset,
    )

    path, cache_path, options = task
    if (cache_path / "state.json").is_file():
        return

    with TemporaryDirectory(dir=cache_path.parent) as tmp:
        features = {
            "atomic_numbers": Sequence(Value("int32")),
            "positions": Sequence(Sequence(Value("float32"), length=3)),
            "forces": Sequence(Sequence(Value("float32"), length=3)),
        }
        # Read the footer first so projection happens in the Parquet reader,
        # before decoding rows (which can contain large PDB/PSF blobs).
        available = set(parquet.read_schema(path).names)
        missing = features.keys() - available
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        columns = list(features)
        if options.get("mol_ids_to_exclude"):
            if "mol_id" not in available:
                raise ValueError(
                    f"mol_ids_to_exclude requires a 'mol_id' column in {path}."
                )
            columns.append("mol_id")
        properties = options["properties"]
        # Prefer explicit total/formal charge; read partial charges only if needed.
        if "q" in properties:
            if "q" in available:
                columns.append("q")
            elif "formal_charges" in available:
                columns.append("formal_charges")
        if "pq" in properties or (
            "q" in properties and not {"q", "formal_charges"}.intersection(columns)
        ):
            if "pq" in available:
                columns.append("pq")
            elif "partial_charges" in available:
                columns.append("partial_charges")
        has_pq = "pq" in columns or "partial_charges" in columns
        # Output features can differ from input columns used to derive charges.
        if "q" in properties and (
            "q" in columns or "formal_charges" in columns or has_pq
        ):
            features["q"] = Value("float32")
        if "pq" in properties and has_pq:
            features["pq"] = Sequence(Value("float32"))
        if options["use_box"]:
            if "box" not in available:
                raise ValueError(f"use_box=True requires a 'box' column in {path}.")
            columns.append("box")
            features["box"] = Sequence(Sequence(Value("float32"), length=3), length=3)
        source = load_dataset(
            "parquet",
            data_files=str(path),
            columns=columns,
            batch_size=256,
            split="train",
            streaming=True,
            cache_dir=tmp,
        )
        features = Features(features)

        # Probe a fresh iterator: from_generator cannot build an empty split.
        # The nonempty branch restarts the stream, so its first row is retained.
        if next(_iter_samples(source, options), None) is None:
            dataset = ArrowDataset.from_dict(
                {k: [] for k in features}, features=features
            )
        else:
            dataset = ArrowDataset.from_generator(
                _iter_samples,
                gen_kwargs={"dataset": source, "options": options},
                features=features,
                cache_dir=tmp,
                writer_batch_size=500,
            )
        fraction = options["sample_fraction"]
        if fraction is not None and fraction < 1:
            # Sample accepted rows reproducibly, independently of worker scheduling.
            seed = int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16)
            dataset = dataset.shuffle(seed=seed).select(
                range(int(len(dataset) * fraction))
            )
        output = Path(tmp) / "dataset"
        if len(dataset):
            dataset.save_to_disk(str(output), max_shard_size="1GB")
        else:
            # An explicit shard is needed to reload empty datasets in datasets 4.x.
            dataset.save_to_disk(str(output), num_shards=1)
        # The temporary directory shares the destination filesystem for atomic rename.
        output.rename(cache_path)
        # Release memory maps before removing generator files on shared filesystems.
        del dataset
        gc.collect()


class HFDataset(Dataset):
    """Molecular Parquet data processed into memory-mapped Hugging Face datasets.

    Requires the optional ``datasets`` dependency (``pip install torchmd-net[hf]``).
    Input rows must contain ``atomic_numbers``, ``positions`` and ``forces``.
    Optional ``q`` and ``pq`` columns are returned with the sample. Box vectors
    are returned only when ``use_box=True``.
    Only the columns needed for the selected properties are read and cached.
    Positions retain their input units; forces are multiplied by ``force_scale``.

    PyG processes missing caches during construction, then each instance opens
    the completed Arrow files through memory mapping. Nonempty input shards must
    have matching output columns. Empty shards are skipped; an entirely empty
    result raises ``ValueError``. Dataset splitting is handled by ``DataModule``.

    Args:
        root (str): Directory for processed Arrow caches.
        paths (str, os.PathLike or list): Parquet file, directory, list of files/directories,
            or JSON mapping subset names to lists of records containing ``path``.
            JSON records override stride, filter, masking, sampling and force-scale
            settings; properties, use_box and hf_workers remain dataset-wide.
            Paths in JSON are relative to that file; other paths use the working
            directory. Directory inputs include immediate ``*.parquet`` files.
        stride (int): Keep every nth row of each input file, starting at row zero
            before other filters (default: 1).
        max_gradient (float, optional): Exclude samples with any force norm at
            or above this value after force scaling. Nonfinite forces are always
            excluded; None disables only the magnitude limit.
        max_num_atoms (int, optional): Maximum atom count before hydrogen removal.
        mask_hydrogen (bool): Remove hydrogen atoms and their per-atom properties.
            Total charge is retained, or derived from all original partial charges.
        sample_fraction (float, optional): Fraction in [0, 1] of accepted rows to
            retain per file, rounding down. Sampling follows filtering and uses
            a seed derived from the source path. None retains all accepted rows.
        force_scale (float): Multiply forces by this positive factor before
            filtering, for explicit unit conversion (default: 1).
        properties (list or tuple of str): Optional properties to return when available,
            chosen from ``q`` and ``pq`` (default: both). Use an empty list for
            force-only datasets with different charge schemas. ``partial_charges``
            is accepted as ``pq``; total charge comes from ``q``, summed
            ``formal_charges``, or summed partial charges, in that order.
        hf_workers (int): Number of files to process in parallel (default: 1).
        use_box (bool): Require and return box vectors for periodic calculations
            (default: False). No box column is read when False.
        transform (callable, optional): Transform applied to each retrieved sample.
        mol_ids_to_exclude (list, str or os.PathLike, optional): Molecule IDs to
            exclude, or a UTF-8 text file with one ID per line. Requires a
            ``mol_id`` column when nonempty. All conformations with matching IDs
            are removed before sampling. IDs are matched as case-sensitive
            strings after trimming whitespace; integer IDs are converted to
            strings. Global file paths use the working directory; file paths
            in JSON records use the JSON directory. A record replaces the global
            exclusion list; None or an empty list disables it for that record.
    """

    def __init__(
        self,
        root,
        paths,
        stride=1,
        max_gradient=None,
        max_num_atoms=None,
        mask_hydrogen=False,
        sample_fraction=None,
        force_scale=1.0,
        properties=("q", "pq"),
        hf_workers=1,
        use_box=False,
        transform=None,
        mol_ids_to_exclude=None,
    ):
        """Resolve input files and cache keys, process missing shards, and open them."""
        from datasets import concatenate_datasets, load_from_disk

        defaults = dict(
            stride=stride,
            max_gradient=max_gradient,
            max_num_atoms=max_num_atoms,
            mask_hydrogen=mask_hydrogen,
            sample_fraction=sample_fraction,
            force_scale=force_scale,
            mol_ids_to_exclude=_load_mol_ids_to_exclude(mol_ids_to_exclude, Path.cwd()),
        )
        if isinstance(properties, str) or set(properties) - {"q", "pq"}:
            raise ValueError(
                "properties must be a list or tuple containing 'q' and/or 'pq'."
            )
        if not isinstance(hf_workers, int) or hf_workers < 1:
            raise ValueError("hf_workers must be a positive integer.")
        self.hf_workers = hf_workers
        # Normalize file lists and named JSON collections into ordered records.
        base = Path.cwd()
        if isinstance(paths, (str, os.PathLike)):
            paths = Path(paths).expanduser()
            if paths.suffix == ".json":
                base = paths.resolve().parent
                with paths.open() as f:
                    metadata = json.load(f)
                records = [record for subset in metadata.values() for record in subset]
            else:
                records = [{"path": paths}]
        else:
            records = [{"path": path} for path in paths]

        self.tasks = []
        processed_dir = Path(root).expanduser().resolve() / "processed"
        for record in records:
            unknown = record.keys() - defaults.keys() - {"path"}
            if unknown:
                raise ValueError(
                    f"Unknown HFDataset processing options: {sorted(unknown)}"
                )
            # A record overrides global processing defaults only for its own file(s).
            options = {**defaults, **{k: v for k, v in record.items() if k != "path"}}
            if "mol_ids_to_exclude" in record:
                options["mol_ids_to_exclude"] = _load_mol_ids_to_exclude(
                    record["mol_ids_to_exclude"], base
                )
            # Keep existing unfiltered cache keys unchanged. Nonempty exclusions
            # are hashed by their normalized contents, not the list-file path.
            if not options["mol_ids_to_exclude"]:
                del options["mol_ids_to_exclude"]
            if not isinstance(options["stride"], int) or options["stride"] < 1:
                raise ValueError("stride must be a positive integer.")
            if options["max_num_atoms"] is not None and (
                not isinstance(options["max_num_atoms"], int)
                or options["max_num_atoms"] < 1
            ):
                raise ValueError("max_num_atoms must be a positive integer or None.")
            fraction = options["sample_fraction"]
            if fraction is not None and not 0 <= fraction <= 1:
                raise ValueError("sample_fraction must be between 0 and 1.")
            if not np.isfinite(options["force_scale"]) or options["force_scale"] <= 0:
                raise ValueError("force_scale must be finite and positive.")
            options.update(properties=sorted(set(properties)), use_box=use_box)
            path = (base / Path(record["path"]).expanduser()).resolve()
            files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
            if not files:
                raise ValueError(f"No Parquet files found in {path}.")
            for path in files:
                stat = path.stat()
                # The final key element is the cache schema version; bump it when
                # changing stored data semantics. File identity uses size and mtime.
                key = json.dumps(
                    [str(path), stat.st_size, stat.st_mtime_ns, options, 2],
                    sort_keys=True,
                )
                digest = hashlib.sha256(key.encode()).hexdigest()
                self.tasks.append((path, processed_dir / f"hf-{digest}", options))
        if not self.tasks:
            raise ValueError("No Parquet files were specified.")

        # PyG checks processed_file_names and invokes process() for missing caches.
        super().__init__(str(Path(root).expanduser().resolve()), transform=transform)
        shards = [
            load_from_disk(str(task[1]), keep_in_memory=False) for task in self.tasks
        ]
        # Empty sources need not supply optional fields used by retained sources.
        shards = [shard for shard in shards if len(shard)]
        if not shards:
            raise ValueError("No conformations remain after HFDataset processing.")
        columns = set(shards[0].column_names)
        if any(set(shard.column_names) != columns for shard in shards[1:]):
            raise ValueError("Parquet shards must provide the same optional columns.")
        self.has_q = "q" in columns
        self.dataset = concatenate_datasets(shards).with_format("torch")

    @property
    def processed_file_names(self):
        """Return completion files used by PyG to decide whether processing is needed."""
        return [str(task[1] / "state.json") for task in self.tasks]

    def process(self):
        """Build missing caches, using at most hf_workers independent file workers."""
        # Build repeated cache paths once. Loading still follows self.tasks order,
        # so partial cache reuse and worker completion cannot reorder samples.
        tasks = list(
            {
                str(task[1]): task
                for task in self.tasks
                if not (task[1] / "state.json").is_file()
            }.values()
        )
        if self.hf_workers > 1 and len(tasks) > 1:
            # Spawn avoids inheriting initialized Torch/CUDA state from the launcher.
            with ProcessPoolExecutor(
                max_workers=min(self.hf_workers, len(tasks)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                list(
                    tqdm(
                        pool.map(_process_shard, tasks),
                        total=len(tasks),
                        desc="Processing Parquet",
                    )
                )
        else:
            for task in tqdm(tasks, desc="Processing Parquet"):
                _process_shard(task)

    def len(self):
        """Return the number of retained conformations across all input shards."""
        return len(self.dataset)

    def get(self, idx):
        """Return one conformation as PyG Data, before the optional transform.

        Atomic numbers ``z`` are int64; ``pos`` and ``neg_dy`` are float32 (N, 3).
        Optional outputs are scalar ``q``, per-atom ``pq`` and ``box`` (1, 3, 3).
        Force signs are preserved: ``neg_dy`` contains the stored physical forces.
        """
        sample = self.dataset[int(idx)]
        data = Data(
            z=sample["atomic_numbers"].long(),
            pos=sample["positions"].float(),
            neg_dy=sample["forces"].float(),
        )
        if "q" in sample:
            data.q = sample["q"].float().reshape(())
        if "pq" in sample:
            data.pq = sample["pq"].float()
        if "box" in sample:
            # PyG concatenates this leading dimension into (batch_size, 3, 3).
            data.box = sample["box"].float().view(1, 3, 3)
        return data
