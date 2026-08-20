# DeepS4

Code, pre-trained weights, and training logs for "Leveraging Spatial Embeddings to
Account for Location-Specific Influences on Agricultural Land Suitability"
(DeepS4 and DeepS3++). The earlier DeepS3 model lives at
https://github.com/AmanjotBhullar/DeepS3.

## Contents

### Model code
| File | Purpose |
|---|---|
| `Env_Yield_noA_lrloop.py` | DeepS4 training (spatial model, 73,007 parameters) |
| `Env_Yield_nonspatial_lrloop.py` | DeepS3++ training (non-spatial, 13,756 parameters) |
| `eval_tf15_paperarch.py` | Canada-only evaluation of the released DeepS4 checkpoint (reproduces Table 4) |
| `eval_ns_base.py` | Canada-only evaluation of the released DeepS3++ checkpoint |
| `ablation_s4.py`, `ablation_ns.py` | Variable-importance analyses (permutation and inference-time ablation) |
| `sens_coords.py` | Coordinate-substitution sensitivity analysis reported in the Supplementary Information (own centroid vs nearest other district vs random in-bag coordinates) |

### Figure code
| File | Purpose |
|---|---|
| `render_figure1.py` | Figure 1 (training and test coverage map) |
| `pixel_maps_s4.py` | Per-pixel DeepS4 suitability predictions over the TFRecords (taps the model's pre-aggregation layer; writes the gridded npz rendered by `render_figure3_grid.py`) |
| `render_figure3_grid.py` | Figure 3 (suitability maps, single page, within-hull tertiles) |
| `split_figure2.py` | Figure 2 (architecture, three panels) |
| `oob_embedding_fix.py` | Per-pixel own-coordinate embedding used by the primary maps |

### Artifacts
| File | Purpose |
|---|---|
| `Env_Yield_noA_lrloop_tf15.weights.h5` | Released DeepS4 checkpoint |
| `Env_Yield_nonspatial_lrloop_paper3.weights.h5` | Released DeepS3++ checkpoint |
| `canada_yield_ranges.json` | Canada-only per-crop standardized yield ranges. Required to reproduce Table 4; ranges computed over the full US and Canada data give wrong relative MAE. |
| `training_logs_env2yield.xlsx` | Complete training logs for the released models, documenting the staged batch-size and learning-rate schedule. |

The evaluation notebook (`validation_noA_tf19.ipynb`, ~30 MB) is included in the
Zenodo archive of this repository rather than in the Git tree.

## Reproducing Table 4

relMAE(crop) = MAE(crop, held-out Canadian districts) / range from
`canada_yield_ranges.json`. Load the DeepS4 checkpoint, evaluate on records with
the `CA/` name prefix, and divide each per-crop MAE by that crop's range.

## Requirements

Python 3, TensorFlow (the released DeepS4 checkpoint was trained under TF 2.15),
NumPy, Matplotlib. Training data are TFRecords built from Google Earth Engine
exports; see the paper's Data Availability statement.
