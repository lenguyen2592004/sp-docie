# Environment Setup and Running Instructions

## Environment Setup

1. **Create a Conda Environment**:
   Create a new Conda environment named `moe` with Python 3.10:

   ```bash
   conda create -n moe python=3.10 -y
   ```

2. **Activate the Conda Environment**:
   Activate the newly created environment:

   ```bash
   conda activate moe
   ```

3. **Run Environment Setup Script**:
   Execute the setup script to configure dependencies:

   ```bash
   bash moe.bash
   ```

4. **Build DGL with CUDA Support**:
   If using CUDA 12.8, build DGL with the provided script:

   ```bash
   bash build_dgl_cuda128.sh
   ```

5. **Optional Runtime Overrides (recommended for stable runs)**:
   ```bash
   source runtime_env_overrides.sh
   ```

### Debugging with Small Sample Sizes

To debug the system with a very small number of samples (e.g., to check for code errors without full training):

1. **Overfit on 1 Sample**:

   ```bash
   python moe.py \
     --stage overfit \
     --overfit-docs 1 \
     --epochs 10 \
     --device auto
   ```

2. **Custom Sample Limit**:
   ```bash
   python moe.py \
     --stage train \
     --train-limit 10 \
     --eval-limit 5 \
     --epochs 2 \
     --device auto
   ```

## Notes

- Ensure Conda is installed on your system before running the commands.
- The `moe.bash` script should contain all necessary dependency installations.
- The `build_dgl_cuda128.sh` script is specific to CUDA 12.8. Adjust if using a different CUDA version.

## Running the Pipeline

### Full Data Run

To run the full pipeline with all data, execute the following commands:

1. **Distant Supervision Cleaning**:

   ```bash
   python moe.py \
     --stage ds \
     --distant-file train_distant.json \
     --eval-file dev.json \
     --distant-clean-file train_distant_clean.json \
     --build-distant-clean \
     --distant-topk 2 \
     --device auto
   ```

2. **Supervised Training**:

   ```bash
   python -u moe.py \
   --stage train \
   --epochs 20 \
   --full-train \
   --full-eval \
   --full-test \
   --pretrained-gnn-checkpoint "<wandb_artifact_url_or_ref>" \
   --no-pair-markers \
   --result-dir inference_results \
   --result-file result.json
   ```

### Smoke Test

To quickly verify the pipeline with a small subset of data, use the `--debug` flag:

1. **Distant Supervision Cleaning (Smoke Test)**:

   ```bash
   python moe.py \
     --stage ds \
     --distant-file train_distant.json \
     --eval-file dev.json \
     --distant-clean-file train_distant_clean.json \
     --build-distant-clean \
     --distant-topk 2 \
     --device auto \
     --debug
   ```

2. **Supervised Training (Smoke Test)**:

   ```bash
   python moe.py \
     --stage train \
     --train-file train_annotated.json \
     --distant-clean-file train_distant_clean.json \
     --eval-file dev.json \
     --test-file dev.json \
     --epochs 2 \
     --num-experts 2 \
     --capacity-factor 1.0 \
     --lambda-moe 0.1 \
     --lambda-scl 0.05 \
     --scl-temp 0.1 \
     --candidate-keep-ratio 0.3 \
     --adaptive-threshold-scale 1.0 \
       --max-pairs-per-doc 25 \
       --max-seq-length 1024 \
       --result-dir inference_results \
       --result-file result.json \
     --device auto \
     --debug
   ```

3. **Overfit Smoke Test (fastest end-to-end check)**:
   ```bash
   python moe.py \
      --stage overfit \
      --overfit-docs 1 \
      --epochs 2 \
      --model-id sshleifer/tiny-gpt2 \
      --result-dir inference_results \
      --result-file result.json \
      --device auto \
      --debug
   ```

## Command Validity Notes

- The commands above are aligned with current `moe.py` arguments (including `--max-seq-length` and multi-label pipeline settings).
- If you want logs on Weights & Biases, keep default behavior (do not pass `--no-wandb`).
- W&B now defaults to `--wandb-mode online`. If your shell/directory was previously set to offline, the run still forces online unless you explicitly pass `--wandb-mode offline`.
- To avoid silent fallback to offline when cloud init fails, add `--wandb-no-offline-fallback`.
- If you hit `403 Forbidden`, avoid stale hardcoded credentials and run with your own account context, for example:
  `wandb login --relogin` then add `--wandb-project <your_project>` (optionally `--wandb-entity <your_team>`).
- For CPU-only debugging, use `--model-id sshleifer/tiny-gpt2` to avoid loading a very large backbone.
- With W&B enabled, the run uploads artifacts for:
  - best checkpoint (`best_model.pt`)
  - inference result JSON and files from `--result-dir` (for example `inference_results/*.json`)
  - full workspace snapshot at run start (all folders and child files under workspace)
- Server-side checkpoint retention keeps only the latest run folder under `checkpoints/`.
- To warm-start from a previous run, pass `--pretrained-gnn-checkpoint` with one of:
   - local file path (for example `checkpoints/<run_name>/best_model.pt`)
   - W&B artifact reference (for example `<entity>/<project>/<artifact_name>:v0`)
   - W&B artifact URL (for example `https://wandb.ai/<entity>/<project>/artifacts/model/<artifact_name>/v0`)
- Add `--pretrained-gnn-strict` if you want strict key matching when loading the checkpoint.

## Command full data

```bash
conda create -n moe python=3.10 -y
conda activate moe
bash moe.bash
bash fix_datapipe.sh
python moe.py \
--stage ds \
--distant-file train_distant.json \
--eval-file dev.json \
--distant-clean-file train_distant_clean.json \
--build-distant-clean \
--distant-topk 2 \
--device auto
python -u moe.py \
--stage train \
--epochs 20 \
--full-train \
--full-eval \
--full-test \
--pretrained-gnn-checkpoint "<wandb_artifact_url_or_ref>" \
--no-pair-markers \
--result-dir inference_results \
--result-file result.json
```
