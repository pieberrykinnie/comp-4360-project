# comp-4360-project

Repository for the Winter 2026 COMP 4360 - Machine Learning project.

## Instructions

### Setup Environment

1. Install `uv`:

  ```bash
  # macOS and Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (not recommended)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

2. Set up the environment:

  ```bash
  uv sync
  ```

### Download Data

1. Run the data fetching script:

  ```bash
  uv run fetch_data.py
  ```

2. Move the downloaded dataset to the expected folder:

  ```bash
  mkdir -p data/
  mv ~/.cache/kagglehub/datasets/ashery/chexpert/versions/1 data/CheXpert-1.0-small/
  ```

### Run Training Loop

Example configuration files are provided in `configs/`. Good base files to modify from are `configs/simmim_pretrain__vit_base_img224__100ep.yaml` and `configs/simmim_finetune__vit_base_img224__100ep.yaml`, for each of the pretraining and finetuning pipelines.

1. Run the SimMIM pretraining script:

  ```bash
  uv run torchrun --nproc-per-node <num_of_gpus> src/main_pretrain.py --cfg <config_file> --resume <optional_ckpt_pth>
  ```

After the training process completes, the model should be located at `output/config_name/config_tag/ckpt_epoch_FINAL_EPOCH.pth`.

2. Run the finetuning script:

  ```bash
  uv run torchrun --nproc-per-node <num_of_gpus> src/main_finetune.py --cfg <config_file> --resume <optional_ckpt_pth> --local-rank 0
  ```

Similarly, after the training process completes, the model should be located at `output/config_name/config_tag/ckpt_epoch_FINAL_EPOCH.pth`.

See any `log_rank_*.txt` in the `output/` directory for examples of what a training log should look like.

### Evaluate Model

By default, the finetuning script runs the evaluation for AUROC after every epoch. You can find this in the training log of the finetuning runs:

```
[2026-04-01 07:39:55 simmim_finetune] (main_finetune.py 394): INFO Test: [0/1]	Time 0.804 (avg 0.804)	Loss 0.4852 (avg 0.4852)	Mem 18757MB
[2026-04-01 07:39:55 simmim_finetune] (main_finetune.py 427): INFO Mean AUROC 0.6257
[2026-04-01 07:39:55 simmim_finetune] (main_finetune.py 428): INFO Per-class AUROC [0.7736013986013985, 0.36818851251840945, 0.6862745098039215, 0.7212669683257918, 0.49751243781094523, 0.71875, 0.677389705882353, 0.4027061855670103, 0.6739107611548557, 0.5413919413919415, 0.7432065217391306, 0.7462686567164178, nan, 0.5836030204962244]
[2026-04-01 07:39:55 simmim_finetune] (main_finetune.py 229): INFO Validation - Epoch 99: mean_auc: 0.6257 | loss: 0.4852
[2026-04-01 07:39:55 simmim_finetune] (main_finetune.py 236): INFO Best validation label accuracy so far: 1.00%
```

If you want to run the evaluation on its own on an existing model checkpoint, add the `--eval` flag:

```bash
uv run src/main_finetune.py --cfg <config_file> --resume <ckpt_pth> --eval --local-rank 0
```
