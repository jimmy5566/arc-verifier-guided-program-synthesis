# Public NVARC augmentation code references

Source: [`ARC-AGI1/002_ivan_arc1.ipynb`](https://raw.githubusercontent.com/1ytic/NVARC/846d0198efa752534594e321fc3289fc0a06c657/ARC-AGI1/002_ivan_arc1.ipynb) at commit `846d0198efa752534594e321fc3289fc0a06c657` (SHA-256 `ec0ac985d7de396c7fb7bede70cc0de40a2161d3b6ec00a123d41024fb8e3104`).

The references below use **logical source-code lines** formed by concatenating code-cell `source` arrays in notebook order; the embedded `%%writefile` targets identify the intended Python file.

| Control | Exact source reference | Audit reading |
|---|---|---|
| Colour permutation | embedded `arc_loader.py`, lines 24–39 (`permute_mod`, `permute_rnd_all_`) | A permutation descriptor is a 10-element bijection. For 2D grids it maps old colour `i` to descriptor digit `i`; inversion uses the inverse permutation. |
| D4 transform / inverse | embedded `arc_loader.py`, lines 91–119 (`ArcDataset.forward_mod`, `ArcDataset.invert_mod`) | `transpose`, then three rotations, yields all eight dihedral representations; output arrays are inverse-transformed before storage. |
| Dataset expansion | embedded `arc_loader.py`, lines 183–191 and 251–258 (`ArcDataset.mod`, `ArcDataset.augment`) | `augment` sets NumPy's global seed, builds transpose × four rotations, then makes `n` independent colour draws and finally calls `shuffle_ex`. |
| Train/demo ordering | embedded `arc_loader.py`, lines 237–249 (`ArcDataset.shuffle_ex`) | One `np.random.permutation(n_train)` draw is used per augmented view. It is deterministic only conditional on seed, NumPy behavior, and task train-count; it is not a universal fixed order. |
| TTT data | embedded `arc_solver.py`, lines 724–741 | `puzzle_ds.augment(n=16, shfl_keys=True, seed=1)` makes 8 D4 × 16 colour samples = 128 TTT records before length cutting. |
| Inference cells | embedded `arc_solver.py`, lines 759–815 | `eval_ds.augment(n=2, seed=2)` makes 8 D4 × 2 colour samples = 16 subkeys. Sorted subkeys are arranged into four physical batches and each batch calls `inference_turbo_dfs`. |
| DFS physical batch | embedded `arc_solver.py`, lines 769–795 and 810–815 | Batches are `[0,1,4,5]`, `[2,3,6,7]`, `[8,9,12,13]`, `[10,11,14,15]` of sorted subkeys: four batches of four. |
| Candidate inverse | embedded `arc_solver.py`, lines 817–831 | Each decoded array is inverted with `puzzle_ds_multi.invert_mod(..., inv_perm=True)`. |
| Candidate rescoring | embedded `arc_solver.py`, lines 833–852 | Each unique candidate grid is transformed with `aug_dataset.augment(seed=hash(bk) % 1024**2)` using default `n=1`: 8 D4 × 1 colour draw = 8 score views, evaluated as two score batches of four. |
| Final selector | embedded `arc_decoder.py`, lines 294–320 and 345–346 | `score_kgmon` groups identical inverse-transformed grids; it uses support count minus mean augmentation score. |

## Mirror provenance

At the same pinned commit, `ARC-AGI1/README.md` says this folder's code is the same as the winning Kaggle notebook and names `002_ivan_arc1.ipynb` as its evaluation code; it limits the stated difference to use of public ARC-AGI 2024 evaluation data for test-time fine-tuning. The direct Kaggle page identifies version `276863151` through its oEmbed metadata, but anonymous source-export endpoints were unavailable during this audit. Therefore the fixed Git mirror is the controlling code source and the direct Kaggle source is recorded as discovered but not independently byte-compared.
