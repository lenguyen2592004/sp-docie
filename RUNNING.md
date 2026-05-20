# Hướng dẫn chạy repo DocRE (MoE + FGW)

Tệp này mô tả các bước thiết lập và các lệnh hay dùng để chạy repo trên máy có GPU hoặc CPU. Nội dung ngắn gọn, copy-paste được.

**Prerequisites**

- Python 3.10
- Conda (khuyến nghị) hoặc virtualenv
- GPU + CUDA driver (nếu chạy trên GPU)

**1) Tạo môi trường & cài phụ thuộc**

Dùng script có sẵn để cài các package chính (POT, DGL, PyTorch, Transformers, bitsandbytes, PEFT, WandB...):

```bash
# Tạo conda env (nếu muốn)
bash /workspace/setup_moe_env.sh
conda activate moe

# Cài dependencies (sẽ cài POT, DGL wheel, torch, transformers, peft, bitsandbytes, sklearn, wandb, lightning...)
bash /workspace/moe.bash
```

Nếu không dùng conda, chạy tương tự bằng `pip`:

```bash
pip install pot dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install torch torchvision torchdata peft bitsandbytes accelerate transformers scikit-learn wandb lightning
```

**2) Kiểm tra import nhanh**

```bash
python /workspace/test_imports.py
```

Nếu có package thiếu, cài theo thông báo.

**3) Các script hữu ích**

- `bash /workspace/start_train.sh` — start training background (1 epoch trong script)
- `bash /workspace/run_test.sh` — run + monitor short test
- `bash /workspace/train_1000_small.sh` — training with small subsets (helper wrapper)
- `python /workspace/launch_training.py` — launch + monitor
- `python /workspace/debug_pipeline.py` — helper debug script

**4) Chạy nhanh (khuyến nghị theo thứ tự)**

a) Data sanity (KHÔNG load LLM):

```bash
python /workspace/moe.py --stage sanity --eval-limit 100
```

b) Candidate stats (KHÔNG load LLM):

```bash
python /workspace/moe.py --stage candidates --candidate-limit 500
```

c) Overfit test (kiểm tra pipeline có thể học):

```bash
python /workspace/moe.py --stage overfit --train-limit 20 --eval-limit 20 --epochs 200 --no-wandb
```

d) Train nhanh 1 epoch (GPU + W&B bật):

```bash
# GPU + W&B (mặc định code sẽ dùng HF cache trong /tmp nếu gặp lỗi NFS)
# NOTE: `--batch-size` now controls the per-document pair mini-batch (default=4).
python /workspace/moe.py --stage train --epochs 1 --device cuda --batch-size 4 --train-limit 2 --eval-limit 2
```

Nếu muốn tắt Wolrd of WandB (local run, no sync):

```bash
python /workspace/moe.py --stage train --epochs 1 --device cuda --train-limit 2 --eval-limit 2 --no-wandb
```

e) Chạy debug nhẹ trên CPU (model nhỏ):

```bash
python /workspace/moe.py --debug --device cpu --cpu-debug-model sshleifer/tiny-gpt2 --epochs 1 --no-wandb
```

**5) Chạy full (chỉ khi đã chắc chắn)**

```bash
python /workspace/moe.py --stage train --full-train --full-eval --epochs 10 --device cuda
```

Lưu ý: mặc định repo đặt `train` subset (500 train, 200 eval) để tránh tốn tài nguyên. Bật `--full-train`/`--full-eval` khi cần chạy toàn bộ.

**6) Vấn đề đã vá và lưu ý vận hành**

- Vấn đề tải model từ HuggingFace trong môi trường chia sẻ (NFS) có thể gây lỗi `OSError: [Errno 116] Stale file handle`. Mã đã thêm fallback: nếu gặp lỗi này thì script đặt HF cache sang `/workspace/tmp/hf_home` hoặc retry với `cache_dir` tạm thời. Vì vậy nếu bạn thấy lỗi cũ, hãy thử chạy lại.
- Nếu DGL không có backend CUDA thì `GraphExpert` sẽ chạy trên CPU (mã in cảnh báo). Để chạy DGL trên CUDA, cài đúng wheel DGL tương thích CUDA (xem https://www.dgl.ai/ và chọn wheel phù hợp với torch/cuda).
- Mặc định W&B sẽ đồng bộ (Config chứa `wandb_key`). Nếu muốn tắt sync: thêm `--no-wandb`.

- **Thay đổi quan trọng (batching):**
	- `--batch-size` điều khiển "pair mini-batch" trên mỗi tài liệu (tức `pair_batch_size` trong code). Giá trị mặc định là `4`.
	- `DataLoader` vẫn sử dụng `batch_size=1` ở cấp tài liệu; tăng `--batch-size` sẽ tăng số cặp (h,t) xử lý cùng lúc và tăng tải GPU.
	- Lời khuyên: bắt đầu với `--batch-size 4`, sau đó tăng dần (6 → 8 → 12) và theo dõi `nvidia-smi` để tránh OOM.

Examples:

```bash
# safe test (no W&B, small data)
python /workspace/moe.py --stage train --device cuda --batch-size 4 --train-limit 2 --eval-limit 2 --no-wandb

# push more throughput (watch GPU memory)
python /workspace/moe.py --stage train --device cuda --batch-size 8 --train-limit 50 --eval-limit 20
```

**7) Logs**

- W&B run logs: `/workspace/wandb/run-*/logs/`
- Starter scripts đặt log tạm thời: `/workspace/tmp/moe_train.log` (script `start_train.sh`)

**8) Free VRAM quick commands**
- In-Python safe cleanup (best inside the same Python session):

```python
import gc, torch
del model, logits, inputs  # các biến bạn giữ
gc.collect()
torch.cuda.empty_cache()
try:
	torch.cuda.ipc_collect()
except AttributeError:
	pass
```

- From terminal: find and kill processes then optionally reset GPU (root):

```bash
nvidia-smi             # xem sử dụng VRAM
# kill PID từ cột PID
kill <PID> || kill -9 <PID>
# reset GPU if supported (root)
sudo nvidia-smi --gpu-reset -i 0
```

**8) Troubleshooting nhanh**

- Nếu gặp `Stale file handle` khi tải model/tokenizer: xóa `/workspace/.hf_home` tạm thời hoặc dùng `--debug` để ép dùng model nhỏ.
- Nếu thiếu RAM/GPU OOM: giảm `--batch-size`, giảm `--max-pairs-per-doc` (default=50), hoặc chạy `--debug`.

---

Nếu muốn mình có thể:
- Thêm một `requirements.txt` chính xác từ môi trường hiện tại.
- Thêm script `run_gpu.sh` sẵn lệnh copyable (GPU + W&B) hoặc `run_cpu_debug.sh`.


