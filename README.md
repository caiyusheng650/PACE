# PACE

Code for **PACE: A Pseudo-label Adaptive Confidence Ensemble Framework for Cognitive Diagnosis**.

PACE combines state-dependent auxiliary-supervision weighting (ESRG), validation-guided committee selection (ESDC), and adaptive response thresholds (ADOP). Supported backbones: NeuralCD (`ncdm`), MIRT, KaNCD, and KSCD.

## Setup

Python 3.11 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

The dependency versions describe the release verification environment, not the original training environment. For GPU training, install a PyTorch build compatible with your CUDA installation.

## Data

Prepare each dataset in the layout described in [DATA.md](DATA.md). Raw datasets and trained checkpoints are not included.

## Run

From the repository root:

```bash
python multibackbone/run.py --repo /path/to/prepared_dataset --output outputs/mirt --backbone mirt --epochs 20 --batch-size 256 --seeds 2024 42 7 --evaluate-test
```

`--repo` is the prepared dataset directory. Output directories must be new. The default device is CPU; use `--device cuda:0` for GPU training.

| Backbone | Batch size | Epochs |
|---|---:|---:|
| MIRT | 256 | 20; 100 for ASSIST2009 |
| NeuralCD | 256 | 20 |
| KaNCD | 128 | 20 |
| KSCD | 16 | 20 |

The defaults use three seeds, learning rate 0.001, 20 sampled auxiliary exercises, and a candidate pool of 100. See [configs/defaults.json](configs/defaults.json) for the configuration reference; it is documentation, not an automatically loaded configuration.

Cross-partition student–exercise pair overlap is rejected by default. To reproduce a declared response-event protocol with overlapping pairs, use `--allow-pair-overlap` explicitly; do not use that flag for pair-disjoint evaluation. The main Junyi response-event protocol and the additional pair-disjoint protocol are different evaluations.

Validation students are split into selection, fitting, and tuning roles (50:25:25). Checkpoints and committee members use only selection students. Thresholds are frozen before test evaluation; test scoring requires `--evaluate-test`.

Run outputs include configurations, data hashes, validation roles, selected checkpoints, committee sequences, threshold rules, predictions, and metrics. These are generated outputs and are not committed here.

## Verification

```bash
python -m unittest discover -s scripts -p test_adop.py -v
python -m unittest discover -s multibackbone -p test_multibackbone.py -v
```

The tests cover backbone interactions, training gradients, data conventions, validation-role separation, and a synthetic end-to-end run. Passing them does not establish reproduction of the paper's dataset results.

## Results

[results/](results/) contains the final six-candidate controls and per-run metrics. These are aggregate reference results, not raw predictions. They include both favorable and unfavorable comparisons. The augmented and Original six-candidate selected committees yield eight wins and seven losses, with a descriptive mean AUC difference of approximately −0.320 percentage points across the 15 settings. The paper's 13/15 comparison uses the unaugmented single-model mean as its comparator.

## Attribution

See [THIRD_PARTY.md](THIRD_PARTY.md) for backbone references and implementation boundaries. No redistribution license is granted by this repository; licensing must be confirmed by the rights holders.

Source snapshot: `caiyusheng650/pace-uecd-optimization-paper@da1a15a9c9bb33a93dfcf6674591841320950598`. Core training, model, data, and ADOP files are retained without algorithm changes. This repository contains the current implementation and final result tables only.

