# Debug & Stage Usage Guide

File `moe.py` được thiết kế theo tư duy pipeline chuẩn (sanity → candidates → overfit → train → final eval).

Quan trọng: **mặc định KHÔNG chạy full data** (kể cả train full / inference full) để tránh tốn tài nguyên và kết luận sai khi pipeline chưa đúng.
Chỉ chạy full khi bạn **chủ động** bật `--full-train` và/hoặc `--full-eval`.

## Chế độ chạy theo stage (`--stage`)

`--stage` giúp tách các giai đoạn debug/analysis để không phải load LLM/GPU khi không cần.

Các stage hiện có:
- `sanity`: Data sanity & label/span validation (**không load LLM**)
- `candidates`: Candidate generation/pruning stats + gold recall (**không load LLM**)
- `overfit`: Overfit test (pipeline có học được không?)
- `train`: Training bình thường (nhưng vẫn subset mặc định)

## Cách dùng nhanh (khuyến nghị)

### 1) Data sanity (không cần LLM)
Chạy 50–100 docs để bắt bug span/label nhanh:
```bash
python moe.py --stage sanity --eval-limit 100
```

### 2) Candidate generation stats (không cần LLM)
Chạy 100–500 docs để xem candidate explosion + gold recall sau pruning:
```bash
python moe.py --stage candidates --candidate-limit 500
```
Output sẽ in:
- `Avg candidates/doc`
- `Gold kept/Gold total => recall=%`

### 3) Overfit test (pipeline có học được không?)
Chạy subset nhỏ (ví dụ 20 docs), tăng `--epochs` để model memorization được:
```bash
python moe.py --stage overfit --train-limit 20 --eval-limit 20 --epochs 200 --no-wandb
```

### 4) Train (subset mặc định, không chạy full)
Mặc định stage `train` dùng subset để debug tốc độ/logic:
- train: 500 docs
- eval: 200 docs

Chạy nhanh 1 epoch:
```bash
python moe.py --stage train --epochs 1 --no-wandb
```

## Debug mode (`--debug`)

`--debug` sẽ ép các limit xuống nhỏ (để chạy rất nhanh), độc lập với stage.

Ví dụ chạy train cực nhỏ:
```bash
python moe.py --debug
```

Lưu ý:
- `--debug-train-samples` và `--debug-samples` vẫn có tác dụng.
- `--debug-prototype-samples` hiện **chưa được dùng** trong code (giữ lại để tương thích, có thể bỏ sau).

### Tùy chỉnh số lượng docs (khuyến nghị dùng limit flags)
```bash
python moe.py --stage train --train-limit 50 --eval-limit 50 --epochs 3 --no-wandb
```

Các flag chính:
- `--train-limit`: giới hạn số doc train (mặc định theo stage)
- `--eval-limit`: giới hạn số doc eval (mặc định theo stage)
- `--candidate-limit`: giới hạn số doc cho stage `candidates`

### Tùy chỉnh số epochs:
```bash
python moe.py --debug --epochs 2
```

Mặc định `--epochs` là **10**.

### 5. Kết hợp các tùy chọn:
```bash
python moe.py --debug --debug-samples 3 --debug-train-samples 5 --epochs 1
```
- Test cực nhanh với chỉ 3 eval samples, 5 train samples, và 1 epoch

## Chạy full (chỉ khi bạn thật sự muốn)

Mặc định **không full**. Nếu muốn chạy full data:
```bash
python moe.py --stage train --full-train --full-eval --epochs 10
```

Khuyến nghị: chỉ bật full sau khi:
- `sanity` OK
- candidate recall OK
- overfit có dấu hiệu học được

## Tùy chọn khả dụng

| Tùy chọn | Mô tả | Giá trị mặc định |
|----------|-------|------------------|
| `--stage` | `sanity/candidates/overfit/train` | `train` |
| `--debug` | Ép limit xuống nhỏ để chạy nhanh | False |
| `--train-limit` | Giới hạn train docs | -1 (theo stage) |
| `--eval-limit` | Giới hạn eval docs | -1 (theo stage) |
| `--candidate-limit` | Giới hạn docs cho candidates | -1 (theo stage) |
| `--full-train` | Chạy full training set | False |
| `--full-eval` | Chạy full evaluation set | False |
| `--epochs` | Số epochs training | 10 |
| `--model-id` | HF model id | `Qwen/Qwen3-4B-Thinking-2507` |
| `--cpu-debug-model` | Model nhỏ khi CPU + debug/overfit | `sshleifer/tiny-gpt2` |
| `--allow-cpu-large-model` | Cho phép load model lớn trên CPU | False |

## Lưu ý

- Khi chạy stage `sanity`/`candidates` sẽ không load LLM, chạy nhanh để bắt lỗi logic.
- Khi bạn thấy “đang chạy full mẫu” ngoài ý muốn: kiểm tra xem có bật `--full-train/--full-eval` không.
- Nếu chạy CPU và model quá lớn, hãy dùng `--model-id` nhỏ hơn hoặc bật `--debug`.
