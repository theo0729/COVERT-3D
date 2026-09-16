# COVERT

COVERT is a training-free method for point-cloud defect segmentation. This
repository contains the source and frozen experiment definitions used by the
associated ESWA manuscript: the main method, four controlled baselines,
ablations, sensitivity analysis, MVTec 3D-AD conversion, and generic
single-cloud evaluation for the RealSense and LiDAR case studies.

Third-party datasets, trained models, caches, predictions, and experiment
outputs are not included. The two author-owned real-sensor case-study point
clouds are included under `data/real_cases/`.

## Environment

Create the frozen Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate covert
```

`requirements.txt` contains the corresponding Python package pins for an
existing environment managed without Conda.

## Quick start

Run COVERT on one point cloud. Ground truth is optional and is used only for
evaluation:

```bash
python covert_sample.py --pc-path path/to/sample.pcd
python covert_sample.py --pc-path path/to/sample.pcd --gt-path path/to/sample_gt.txt
```

Visualization is disabled by default. Add `--visualize` for the interactive
Open3D view. The authoritative frozen configuration snapshot is
`covert_mainline_config.json`; the equivalent runtime configuration is
`covert_sample.DEFAULT_CONFIG`.

## Data preparation

No third-party dataset is redistributed. Keep datasets outside version control
or place them in the repository-relative locations below.

### Real3D-AD

Obtain Real3D-AD from its [official repository](https://github.com/M-3LAB/Real3D-AD).
The submitted benchmark uses `candybar`, `diamond`, `fish`, and `toffees`.
Arrange the point-cloud version as follows:

```text
data/real3d_ad/
├── candybar/
│   ├── test/<sample>.pcd
│   └── gt/<sample>.txt
├── diamond/
├── fish/
└── toffees/
```

Each defective cloud must have a matching GT text file with the same stem. The
loader accepts Real3D-AD-style `x y z label` rows. Defect-free samples are
discovered from point-cloud names containing the `good` token and do not need
a GT file. Override this root with `--dataset-root` for main and ablation batch
runs, or `--data-root` for sensitivity runs.

### MVTec 3D-AD

Obtain the original data from the [official MVTec dataset page](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
Convert the benchmark categories `bagel`, `carrot`, `peach`, and `potato` one at a
time. For example:

```bash
python tools/convert_mvtec3d_test_to_covert.py --source path/to/mvtec3d/potato --output data/mvtec_3d_ad_converted/potato
```

The converter reads `<category>/test/<defect>/xyz/<id>.tiff` and the matching
PNG masks, removes invalid depth and, by default, the support plane, then writes:

```text
data/mvtec_3d_ad_converted/
└── potato/
    ├── test/<id>_<defect>.pcd
    ├── gt/<id>_<defect>.txt       # defective samples only
    └── conversion_manifest.json
```

PCD output is binary little-endian float32 XYZ. GT rows are `x y z label` and
remain index-aligned with the converted cloud. Use `--dry-run` or
`--limit-per-class 1` for a lightweight conversion check; run the converter
with `--help` for all validated options. Override the converted root with
`--mvtec-root`.

### RealSense and LiDAR case studies

The two author-owned real-sensor case-study point clouds are included at:

```text
data/real_cases/
├── realsense/realsense_blade_points.pcd
└── lidar/mid360_lidar_points.pcd
```

These author-owned files provide the RealSense and LiDAR examples reported in
the case study. Optional aligned annotations can be stored beside a cloud as
`<case_id>_gt.txt`. Supported input formats are `.pcd`, `.ply`, `.npy`, `.txt`,
`.xyz`, and `.csv`; text GT may contain one binary label per row or aligned
`x y z label` rows.

```bash
python covert_sample.py --pc-path data/real_cases/realsense/realsense_blade_points.pcd
python covert_sample.py --pc-path data/real_cases/lidar/mid360_lidar_points.pcd
```

## Reproducing the paper

Run all commands from the repository root after preparing the datasets and
activating the environment.

### Main COVERT benchmark

The submitted eight-category benchmark contains four Real3D-AD categories and
four converted MVTec 3D-AD categories:

```bash
python covert_batch.py --dataset-root data/real3d_ad --mvtec-root data/mvtec_3d_ad_converted --categories candybar diamond fish toffees bagel carrot peach potato --output-dir results/covert_benchmark --sample-workers 4 --query-workers 1 --good-samples
```

Add `--resume` to continue a compatible interrupted run. Checkpoint and
progress files do not alter the frozen detector configuration.

### Controlled baselines

All four baselines consume one shared deterministic 10,000-point cache. Create
it once:

```bash
python baseline/common/export_benchmark_cache.py --dataset-root data/real3d_ad --mvtec-root data/mvtec_3d_ad_converted
```

Then run the frozen configurations:

```bash
python baseline/SOR/run_experiment.py
python baseline/FPFH_IF/run_experiment.py
python baseline/RG/run_experiment.py
python -m baseline.PointSGRADE.run_experiment --allow-full-benchmark
```

Each baseline's `config.yaml` is authoritative. For a bounded smoke run, add
`--limit 1` and an isolated `--output-dir`; PointSGRADE also accepts repeatable
`--sample CATEGORY/SAMPLE_ID` selectors. A guarded PointSGRADE smoke example is:

```bash
python -m baseline.PointSGRADE.run_experiment --limit 1 --output-dir baseline/PointSGRADE/results/smoke --overwrite
```

### Ablations

Run the six main ablations through the cross-platform Python entry point:

```bash
python ablations/covert_ablation_batch.py --ablation-mode full --sample-workers 4 --query-workers 1 --good-samples --resume
python ablations/covert_ablation_batch.py --ablation-mode laplacian_restoration --sample-workers 4 --query-workers 1 --good-samples --resume
python ablations/covert_ablation_batch.py --ablation-mode wo_va_hks --sample-workers 4 --query-workers 1 --good-samples --resume
python ablations/covert_ablation_batch.py --ablation-mode wo_structural_suppression --sample-workers 4 --query-workers 1 --good-samples --resume
python ablations/covert_ablation_batch.py --ablation-mode wo_seed_growth --sample-workers 4 --query-workers 1 --good-samples --resume
python ablations/covert_ablation_batch.py --ablation-mode wo_counterfactual_verification --sample-workers 4 --query-workers 1 --good-samples --resume
```

Run the reported local-control calibration ablation:

```bash
python ablations/covert_ablation_batch.py --ablation-mode wo_local_control_calibration --sample-workers 4 --query-workers 1 --good-samples --resume
```

Run the fixed absolute-threshold verification, including its frozen fitting and
held-out protocol:

```bash
python ablations/fixed_absolute_threshold_experiment.py run --sample-workers 4 --query-workers 1
```

On Windows, the equivalent sequential convenience launchers are:

```powershell
.\run_all_ablations_main8.ps1
.\ablations\run_fixed_absolute_threshold_experiment.ps1
```

The supported modes and their scientific contracts are declared in
`ablations/ablation_modes.py`. Frozen Dev10 physical-group identities used for
held-out exclusion are stored in `experiments/dev10_physical_groups.csv`.

### Sensitivity runs

Run the five formal families with their cross-platform Python entry points:

```bash
python sensitivity_analysis/01_graph_geometry_coupling/run_sensitivity.py
python sensitivity_analysis/02_vahks_modulation/run_sensitivity.py
python sensitivity_analysis/03_ac_boundary/run_sensitivity.py
python sensitivity_analysis/04_candidate_recovery/run_sensitivity.py
python sensitivity_analysis/05_gtr_local_calibration/run_sensitivity.py
```

On Windows, the following convenience launcher runs them sequentially; add
`-Smoke` for its bounded one-sample-per-family validation:

```powershell
.\sensitivity_analysis\run_all_sensitivity_main8.ps1
```

### Expected outputs

- Main COVERT writes to the selected `--output-dir`, including
  `batch_progress.json`, per-sample metrics, aggregate CSV/JSON, and
  `good_metrics.json`.
- Baselines write per-sample and summary CSVs, resolved configuration,
  environment provenance, run manifest, predictions, and good-sample metrics
  below each configured `paths.output_dir`.
- Ablations write isolated metrics, configuration metadata, progress, and
  good-sample results below `ablations/<mode>/` by default.
- Sensitivity families write raw per-configuration artifacts and aggregate
  summaries below their `results/` directories; canonical figures are written
  to `sensitivity_analysis/figures/` by default.

Generated artifacts are excluded by `.gitignore`. No cross-machine runtime is
asserted because this release contains no verified hardware/runtime record.
Worker counts are execution controls rather than frozen scientific parameters;
choose them for the available CPU and memory while keeping method parameters
fixed.

## Frozen baseline configurations

All methods use the ordered, normalized 10,000-point shared cache and report
per-sample Precision, Recall, F1, and IoU with category-balanced macro
aggregation. Good-sample FPR evaluation is enabled.

| Baseline | Frozen configuration | Positive class |
| --- | --- | --- |
| SOR | `nb_neighbors=5`; `std_ratio=0.25` | Point removed by Open3D statistical outlier removal |
| FPFH+IF | KNN normals `k=8`; consistent tangent-plane orientation; KNN FPFH `k=24`; Isolation Forest `n_estimators=100`, `max_samples=4096`, `contamination=auto`, `random_state=0` | Isolation Forest label `-1` |
| Region Growing | Index-preserving local-plane smoothing `k=30`, 1 iteration; local-PCA normals `k=50`; region-neighbour KNN `k=30`; smoothness `5.0°`; curvature `0.002`; minimum cluster size `1`; sign-invariant normal comparison | All points outside the largest region, including unassigned points |
| PointSGRADE | `lambda0=0.007`; `epsilon=0.001`; `num_neighbor=1500`; `num_neighbor_max=800`; `threshold_angle=0.17453292519943295` rad; `threshold_dist=0.5`; `sigma=2.0`; `random_state=0` | Recovered anomaly displacement norm above `1e-3` |

Formal outputs default to
`baseline/{SOR,FPFH_IF,RG,PointSGRADE}/results/<experiment_name>/`, where the
exact experiment name is recorded in the corresponding `config.yaml`.

## Sensitivity analysis

Every sensitivity configuration starts from `covert_sample.DEFAULT_CONFIG` and
changes only its declared target fields. The formal sweep contains 37
configurations:

| Family | Configurations | Entry point |
| --- | ---: | --- |
| Graph-geometry coupling | 6 | `sensitivity_analysis/01_graph_geometry_coupling/run_sensitivity.py` |
| VA-HKS modulation | 6 | `sensitivity_analysis/02_vahks_modulation/run_sensitivity.py` |
| AC boundary | 5 | `sensitivity_analysis/03_ac_boundary/run_sensitivity.py` |
| Candidate recovery | 15 | `sensitivity_analysis/04_candidate_recovery/run_sensitivity.py` |
| GTR local-normal calibration | 5 | `sensitivity_analysis/05_gtr_local_calibration/run_sensitivity.py` |

Run one family directly, for example:

```bash
python sensitivity_analysis/01_graph_geometry_coupling/run_sensitivity.py
```

Useful lightweight controls include `--dry-run`, `--quick`,
`--only CONFIG_ID`, `--categories CATEGORY ...`, and
`--sample-key CATEGORY/SAMPLE_ID`. Use `--data-root` and `--mvtec-root` to
override the repository-relative dataset locations.

The single canonical figure generator reads completed aggregate CSV files:

```bash
python sensitivity_analysis/plot_sensitivity_results.py
```

It rejects incomplete formal sweeps unless the diagnostic
`--allow-incomplete` option is supplied. The frozen Dev10 exclusion manifest is
`experiments/dev10_physical_groups.csv`; a `*_cut` sample and its base sample
are treated as one physical group.

## Repository structure

- `vast/`: COVERT data loading, preprocessing, graph, feature, boundary,
  scoring, restoration, and visualization modules.
- `covert_sample.py`, `covert_batch.py`: public single-sample and batch entry
  points.
- `baseline/`: shared benchmark-cache tooling plus controlled SOR, FPFH+IF,
  Region Growing, and PointSGRADE implementations and frozen configurations.
- `ablations/`: reported main and supplementary ablation implementations.
- `sensitivity_analysis/`: five frozen sensitivity families, their Windows
  convenience launcher, and the canonical plotting entry point.
- `tools/`: MVTec 3D-AD conversion utility.
- `experiments/benchmark_categories.py`: the final eight-category benchmark
  definition; `experiments/` also contains the frozen Dev10 cohort manifest.
- `data/`: repository-relative third-party dataset locations plus the included
  author-owned RealSense and LiDAR case-study clouds.
- `LICENSE`: MIT License for COVERT-authored source code.

## Third-party software

Runtime Python dependencies are listed in `environment.yml` and
`requirements.txt` but are not redistributed.

The PointSGRADE controlled baseline includes only the minimal frozen upstream
solver subset imported by the adapter, under
`baseline/PointSGRADE/vendor/pointSGRADE`. The complete upstream project is
[ctaoaa/pointSGRADE](https://github.com/ctaoaa/pointSGRADE). PointSGRADE is
provided under the MIT License; its accompanying license is retained verbatim
at `baseline/PointSGRADE/vendor/pointSGRADE/LICENSE`.

## License

COVERT-authored source code is released under the MIT License in the root
`LICENSE`. Bundled third-party code remains subject to its own accompanying
license. Real3D-AD and MVTec 3D-AD are not redistributed and remain subject to
the terms of their original providers. No real author names are included while
the manuscript is under double-anonymized review.
