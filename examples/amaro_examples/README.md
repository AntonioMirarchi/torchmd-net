# TensorNet smoke test with three mdCATH temperatures

Run from the repository root in an environment with TorchMD-Net and its `hf`
extra installed:

```bash
python -m torchmdnet.scripts.train --conf examples/amaro_examples/input.yaml
```

This trains a one-layer TensorNet with 32 embedding channels and 8 radial basis
functions for one epoch, then evaluates its best checkpoint. It uses force labels
only, removes hydrogen atoms, and reads one mdCATH Parquet file at each of
300 K, 320 K and 348 K. It does not use charges or periodic boxes. The split is
random over conformations, as in the rest of the repository.
This is a training smoke test, not a benchmark split.

With this configuration, dataset preparation and setup run inside Lightning's
training lifecycle. Global rank zero builds the shared caches, all ranks wait,
and then each rank opens them. Lightning partitions sample indices across GPU
DataLoaders. All ranks must see the same cache and source paths. Configurations
using standardization or priors retain eager setup before model construction;
this example uses neither.

`three_sources.json` uses paths relative to that JSON file, pointing to the local
`hf/data/mdCATH*` directories. Adjust these paths for your data location. The
YAML's paths are relative to the working directory. Caches go to
`.cache/amaro-hf-smoke` and logs and checkpoints to `logs/amaro-hf-smoke`.

| Collection | Stride | Force multiplier |
| --- | ---: | ---: |
| mdCATH300K | 5000 | 1 |
| mdCATH320K | 10000 | 1 |
| mdCATH348K | 20000 | 1 |

Strides count rows independently in each source file, starting at row zero.
Atom-count and force filters can discard additional rows. These mdCATH inputs
use angstroms for positions and kcal/mol/angstrom for forces, so all force
multipliers are 1. `max_gradient` uses the same force units.

With `properties: []` and `use_box: false`, the Parquet reader loads only
`atomic_numbers`, `positions`, and `forces`. PDB/PSF blobs, identifiers, energies,
charges and other metadata are not decoded or cached. The first run still scans
the selected columns of the source files; a large stride does not make these
multi-GB inputs into small downloads. Later runs reuse the processed caches.

## Configuration precedence

For ordinary training arguments, argparse applies options in command-line order.
Place `--conf` first and overrides after it:

```bash
torchmd-train \
  --conf examples/amaro_examples/input.yaml --num-epochs 2 --batch-size 2
```

Dataset processing resolves in two stages:

1. The YAML supplies `dataset_arg`. A later CLI `--dataset-arg` replaces that
   entire dictionary; it does not merge individual keys. Arguments before
   `--conf` can instead be overwritten by the YAML.
2. Each JSON record overrides the resulting global processing defaults for
   that file. Thus the JSON strides above win over YAML `stride: 1`, including
   a global stride supplied through the CLI. Fields absent from the JSON use
   `dataset_arg`, then the `HFDataset` constructor defaults.

For example, this changes the atom-count limit while keeping the JSON's strides:

```bash
torchmd-train --conf examples/amaro_examples/input.yaml \
  --dataset-arg '{"paths":"examples/amaro_examples/three_sources.json","properties":[],"mask_hydrogen":true,"max_num_atoms":8192,"max_gradient":500,"hf_workers":1,"use_box":false}'
```

To change a per-file stride, edit its JSON record. `properties`, `use_box`, and
`hf_workers` are dataset-wide settings supplied in `dataset_arg`, not JSON records.
