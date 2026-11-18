# scaling-surgical-ai

Code for "A Comparative Study in Surgical AI: Potential and Limitations of Data, Compute, and Scaling":
[paper](https://arxiv.org/abs/2603.27341),
[project page](https://skobelev.me/scaling-surgical-ai),
[models](https://huggingface.co/collections/skblv/surgical-video-understanding),
[leaderboard](https://eval.surgicalvideo.io).

The experiments are surgical tool detection on SDSC-EEA, CholecT50, PitVis-2023 and SurgVU: zero-shot vision-language models, LoRA fine-tuning of Gemma 3 27B and a LoRA rank sweep, YOLOv12-m and ResNet-50 baselines, and frontier proprietary models through APIs.

## Scope

- Researchers may use CholecT50, PitVis-2023 and SurgVU datasets, which are available subject to their corresponding data agreements. SDSC-EEA is a private dataset and access is granted by the Surgical Data Science Collective under individual data-use agreements. Please reach out to SDSC for access to SDSC-EEA. 

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                        # figures, aggregation, tests
uv sync --extra gpu --extra yolo --extra api   # experiments
cp .env.example .env                           # then set the roots and keys
```

## Figures

```bash
uv run python figures/make_figures.py
```

writes Figures 2 to 4 and their exact-match counterparts to `figures/output/` from the tables in `figures/data/`. Figure 1 needs ten SDSC-EEA frames: add `--frames-dir <directory>`.

## Experiments

Run the scripts in `experiments/<dataset>/` from the repository root with the variables of `.env` loaded:

```bash
set -a; . ./.env; set +a
uv run python experiments/cholect50/create_dataset.py
uv run python experiments/cholect50/zero_shot_vlm.py
uv run torchrun --nproc_per_node=<gpus> experiments/cholect50/lora_classification_head.py
```

Per dataset, `create_dataset` runs first. The fine-tuning scripts (`lora_classification_head`, `lora_json_head`, `lora_rank_sweep` and `resnet50`) are distributed and start through `torchrun`; the others start with `python`. The scripts in `experiments/cross_dataset/` aggregate the predictions into micro-F1, exact match and bootstrap confidence intervals. Outputs go to `$OUTPUT_ROOT/<dataset>_<script>`.

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Layout

- `src/scaling_surgical_ai/`: paths, metrics, bootstrap, prediction loaders, and the pipelines shared by PitVis-2023 and SurgVU.
- `experiments/`: one script per experiment, grouped by dataset.
- `figures/`: figure scripts and the numbers behind them.
- `assets/`: the list of evaluated models.

## Citation

```bibtex
@misc{skobelev2026comparativestudysurgicalai,
      title={A Comparative Study in Surgical AI: Potential and Limitations of Data, Compute, and Scaling},
      author={Kirill Skobelev and Eric Fithian and Yegor Baranovski and Jack Cook and Sandeep Angara and Shauna Otto and Zhuang-Fang Yi and John Zhu and Neeraj Mainkar and Margaux Masson-Forsythe and Daniel A. Donoho and X. Y. Han},
      year={2026},
      eprint={2603.27341},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.27341},
}
```
