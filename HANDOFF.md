# HANDOFF — Adaptive-Depth MoE + PU Learning cho DocRE (cho agent code tiếp)

> Tài liệu bàn giao. Đọc kèm [RESEARCH_DESIGN_PU_MoE.md](RESEARCH_DESIGN_PU_MoE.md) để hiểu lý thuyết/lý do.
> File code chính: [moe.py](moe.py). Quy ước: **không chạy training**; chỉ sửa code + `py -m py_compile moe.py` để check cú pháp.

---

## 0. Mục tiêu nghiên cứu (1 câu)
Giải **false-negative / incomplete-labeling** trong document-level RE (DocRED/Re-DocRED) bằng kiến trúc
**MoE với adaptive-depth experts + difficulty-aware routing** (pillar kiến trúc) **+ PU learning** (pillar học máy),
backbone **RoBERTa-large**, có **shared/residual expert** làm thành phần ổn định.

Bar phải vượt: **TTM-RE = 84.01 F1** (Re-DocRED, Human+Distant).

---

## 1. Pipeline tổng thể (mục tiêu cuối)

```
DocRED doc
 └─ Stage DS (offline): clean_distant_supervision_data — MIL top-k + type-constraint  → train_distant_clean.json   [đã có]
 └─ Stage train:
     1. RoBERTa-large encoder (LoRA q/v) → token embeddings                            [ĐÃ ĐỔI BACKBONE]
     2. Candidate generation + prefilter (giảm số cặp)                                  [đã có]
     3. Per-pair k-hop entity subgraph (≤15 node, co-occurrence edges)                  [đã có]
     4. MoE:
          - DifficultyAwareRouter (pair_feats ⊕ difficulty_feats từ subgraph)          [ĐÃ LÀM]
          - Shared/residual expert (luôn áp) + N routed experts dị-độ-sâu [1,2,4]       [ĐÃ LÀM]
          - pair_emb = shared_out + routed_out → classifier                             [ĐÃ LÀM]
          - Expert nội bộ = AGGCN (soft-adjacency + dense GCN)                          [CHƯA LÀM — hiện là GraphExpert placeholder]
          - (mục tiêu) Expert-Choice routing trên pool theo document                    [CHƯA LÀM]
     5. Loss:
          - (hiện tại) focal + switch-LB + structural-contrastive                       [đã có]
          - (mục tiêu) PU loss: nnPU + SSR-PU squared-ranking, per-relation             [CHƯA LÀM]
          - FP handled bởi MIL weight; FN handled bởi PU (nhiễu bất đối xứng)           [CHƯA LÀM]
 └─ Eval: Re-DocRED test sạch, metric F1 + Ign F1 + RE_ignore_distant
```

---

## 2. ĐÃ LÀM ĐƯỢC (đã commit vào moe.py, compile OK)

### 2.1 Đổi backbone Qwen3-8B → RoBERTa-large (backbone-aware, vẫn chạy được Qwen)
| Thay đổi | Vị trí |
|---|---|
| Import `AutoModel`, `AutoConfig` | [moe.py:143](moe.py#L143) |
| `_is_encoder_backbone()` + nạp encoder bằng `AutoModel`, **fp32, không quant** | trong `_load_base_model` (~moe.py:2600) |
| Guard nới: bitsandbytes chỉ cần cho causal 4-bit | [moe.py:2174](moe.py#L2174) |
| `pad_token` chỉ set khi `None` (2 chỗ) — RoBERTa đã có `<pad>` | ~moe.py:2569, ~2629 |
| LoRA target `["query","value"]` cho RoBERTa/BERT | ~moe.py:2660 |
| Default `--model-id roberta-large`, `--max-seq-length 512` | ~moe.py:1924, ~2019 |

**Không cần đổi:** `hidden_size` tự đọc từ config (4096→1024), trích `hidden_states[-1]` dùng chung,
`add_prefix_space=True`, `resize_token_embeddings` cho `[E1][E2]`.

### 2.2 Adaptive-depth experts + difficulty router + shared/residual
| Thành phần | Vị trí |
|---|---|
| `DifficultyAwareRouter` (input = pair_feats ⊕ diff_feats) | [moe.py:1201](moe.py#L1201) |
| `MoEGraphRE._difficulty_features` (BFS hop(h,t), node/edge/direct, [0,1]) | ~moe.py:1409 |
| Heterogeneous-depth experts `[1,2,4]` (**dùng `GraphExpert` làm placeholder — sẽ thay bằng AGGCN, §3.0**) | `MoEGraphRE.__init__` ~moe.py:1353 |
| Shared/residual expert (luôn áp) | ~moe.py:1387 |
| Forward residual `shared_out + routed_out` | `MoEGraphRE.forward` ~moe.py:1456 |
| CLI `--expert-depths`, `--no-shared-expert` | ~moe.py:2020 |
| Truyền vào constructor + move shared_expert lên graph_device | ~moe.py:2836, ~2858 |

---

## 3. PHẢI LÀM (chưa hiện thực)

### 3.0 AGGCN expert (ƯU TIÊN CAO — pillar kiến trúc) — xem RESEARCH_DESIGN §10
Hiện expert là [`GraphExpert`](moe.py#L1269) (graph-masked transformer) làm **placeholder**.
Design yêu cầu **AGGCN** (soft-adjacency + densely-connected GCN). AGGCN ≠ GraphExpert.

**Việc cụ thể:**
1. Viết `AGGCNExpert(in_dim, out_dim, num_blocks, num_heads=2, dense_sublayers=2)` (xem RESEARCH_DESIGN §10.2-10.4):
   - mỗi block = Attention-Guided Layer (M soft-adjacency) + M Densely-Connected GCN + Linear Combination (+residual/LayerNorm).
   - `num_blocks` = "depth" → map từ `--expert-depths` (1/2/4 block).
   - gắn adjacency gốc A (từ `g.edges()`) vào head 0 (tùy chọn, ablation).
2. **Thay `GraphExpert` → `AGGCNExpert`** ở 2 chỗ trong `MoEGraphRE.__init__`:
   routed experts ([moe.py:1382](moe.py#L1382)) và shared_expert ([moe.py:1390](moe.py#L1390)).
   `num_layers=d` → `num_blocks=d`.
3. **Giữ interface y nguyên:** `forward(g, h, pair_repr) -> (B, out_dim*2)` qua gather `is_ht`;
   xử lý per-subgraph (unbatch); fallback graph rỗng như `GraphExpert._fallback`.
4. **KHÔNG đụng** router / difficulty / shared-residual / dispatch — chỉ thay nội bộ expert.
5. Giữ M=2, L=2 (chống overfit, data ~3k doc).

### 3.1 PU loss — pillar 2 (ƯU TIÊN CAO) — xem RESEARCH_DESIGN §1,2,3
Thay closed-world negatives bằng PU-consistent loss.

**Việc cụ thể:**
1. **Bỏ closed-world negatives:** ở training loop, cặp không-gold hiện bị coi là **negative**
   ([moe.py:3075-3079](moe.py#L3075)) và `build_multi_hot_targets` ([moe.py:899](moe.py#L899)) gán 0 cho mọi cặp không-gold.
   → coi cặp không-gold là **unlabeled** (U), không phải negative (N).
2. **Viết `nnpu_squared_ranking_loss(logits, targets, pi_per_rel, f0_none_score)`**:
   - nnPU non-negative clamp (Kiryo NeurIPS 2017): `R = π·E_P[ℓ⁺] + max(0, E_U[ℓ⁻] − π·E_P[ℓ⁻])`.
   - squared-ranking với none-class threshold f₀ (SSR-PU): `ℓ_SR = ¼(y(f−f₀) − margin)²`.
   - **clamp tính GLOBAL trên cả pool**, KHÔNG per-expert (xem §2 OQ2).
3. **FP handling (nhiễu bất đối xứng):** dùng trường `weight` trong train_distant_clean.json
   làm trọng số cho `E_P` (`Σ w·ℓ⁺ / Σ w`) → giảm ảnh hưởng positive nghi-FP.
4. **Ước lượng class-prior π_r per-relation** (mục mở — xem §H open question). Bắt đầu đơn giản:
   π_r = tần suất quan hệ r trên train (hoặc SSR-PU prior-shift). Cần ablation.
5. Thay/song song với `focal_loss_with_logits` cho doc DS; giữ focal cho doc gold nếu muốn.

### 3.2 EC routing + mở pool theo document (ƯU TIÊN TRUNG BÌNH) — xem RESEARCH_DESIGN §2.5, mục "mở pool"
Hiện tại routing là **top-1 với capacity**, pool = `--batch-size = 4` cặp → quá nhỏ cho Expert-Choice.

**Việc cụ thể:**
1. **Mở pool:** ở training loop ([moe.py:3090-3093](moe.py#L3090)) đang chunk cặp thành mini-batch 4.
   Gom **toàn bộ cặp của 1 document** vào một lần gọi `model(...)` (hoặc multi-doc) → n≈12–25.
   Tương tự ở `evaluate_model` ([moe.py:1688](moe.py#L1688), `pair_batch_size=10`).
2. **Đổi router/dispatch sang Expert-Choice:** mỗi expert chọn top-`k = ceil(c·n/e)` cặp
   (thay vì mỗi cặp chọn 1 expert). Giữ shared/residual expert như cũ.
3. **Bỏ `switch_load_balance_loss`** ([moe.py:3313](moe.py#L3313)) khi đã EC (EC tự cân bằng); thêm entropy reg nhỏ.
4. **Bảo vệ positive hiếm** khỏi bị capacity drop (must-keep, giống `limit_candidates_preserve_must_keep`).

### 3.3 (tùy chọn, nâng cao) — xem RESEARCH_DESIGN §3, §H
- Symmetric loss robust (Charoenphakdee ICML 2019) cho residual FP.
- Mô hình hóa tường minh FP rate η_r (PUbN / noisy-PU).
- Expert-agreement FN-mining (Hướng 1) nếu muốn đẩy trần.

---

## 4. CẦN TEST KỸ (dễ sai, kiểm trước khi tin kết quả)

### 4.1 Backbone RoBERTa
- [ ] **Truncation 512:** doc DocRED dài >512 subword → thực thể đuôi mất mention → fallback `doc_context`
  ([moe.py:1052](moe.py#L1052)). Đo % mention bị rơi; cân nhắc sliding-window nếu cao.
- [ ] **LoRA thật sự bật:** in ra `target_modules` và số tensor LoRA trainable (~moe.py:2783) — phải > 0.
  Nếu = 0 nghĩa là dò module sai → backbone đông cứng hoàn toàn.
- [ ] **pad/attention mask đúng:** xác nhận `pad_token != eos` cho RoBERTa; attention_mask có 0 ở pad.
- [ ] **Edge-case fallback:** nhánh "unsupported architecture" ([moe.py:2617](moe.py#L2617)) tính lại `is_encoder` chưa được cập nhật — chỉ ảnh hưởng khi model lạ, không phải roberta-large.

### 4.2 Difficulty features + router
- [ ] **Thứ tự khớp:** `graph_list = dgl.unbatch(subgraphs)` PHẢI cùng thứ tự với `pair_features` rows.
  Verify bằng cách so `is_ht` node feats với pair_features tương ứng.
- [ ] **diff_feats hợp lệ:** in phân phối hop_norm — phải có cả cặp dễ (hop nhỏ) lẫn khó (hop=1.0).
  Nếu tất cả = 1.0 → BFS sai hoặc graph toàn disconnected.
- [ ] **Router học độ khó:** heatmap `expert_depth × hop_distance` — kỳ vọng cặp hop xa → expert sâu.
  Nếu không tương quan → tăng trọng số diff_feats hoặc kiểm noise_eps.

### 4.3 Shared/residual + heterogeneous experts
- [ ] **Dims khớp:** `shared_out`, `routed_out`, classifier input đều `expert_dim*2`. Đã verify logic, test runtime.
- [ ] **Gradient path:** mọi cặp (kể cả bị capacity drop) vẫn có đường gradient (qua shared_out). Kiểm `loss.requires_grad`.
- [ ] **Tỉ trọng shared vs routed:** log `||shared_out||` vs `||routed_out||` — nếu routed≈0 thì router lười / shared nuốt hết.
- [ ] **Capacity với pool nhỏ:** hiện batch=4, capacity≈2 → nhiều cặp drop. Sau khi mở pool (§3.2) mới hợp lý.

### 4.3b AGGCN expert (khi làm §3.0)
- [ ] **Output dim:** `AGGCNExpert` trả `out_dim*2` (= `expert_dim*2`) khớp classifier + residual add.
- [ ] **Per-subgraph đúng thứ tự:** unbatch giữ thứ tự khớp `pair_features` (như GraphExpert).
- [ ] **Soft-adjacency hợp lệ:** in 1 ma trận `Ã` mẫu — phải là phân phối (softmax theo hàng), không NaN.
- [ ] **KHÔNG hard-prune:** xác nhận AGGCN nhận **full graph**; hop-distance chỉ dùng cho router (mềm), không cắt graph.
- [ ] **Param/overfit:** đếm tham số AGGCN×(experts+shared); theo dõi dev F1 sớm; giữ M=2,L=2.
- [ ] **Fallback graph rỗng/1-node:** không crash (giống `GraphExpert._fallback`).
- [ ] **Ablation:** GraphExpert vs AGGCNExpert có chênh F1 (chứng minh soft-pruning giúp).

### 4.4 PU loss (khi làm §3.1)
- [ ] **nnPU clamp:** log tần suất term âm bị clamp; nếu ~100% → π_r sai hoặc pool nhỏ.
- [ ] **Global vs per-expert clamp:** PHẢI global. Ablation chứng minh.
- [ ] **Stop-grad router:** term ascent KHÔNG được backprop vào router gate (xem RESEARCH_DESIGN §2).
- [ ] **π_r ổn định cho quan hệ hiếm:** quan hệ long-tail có thể 0 positive trong batch → tránh chia 0.
- [ ] **Eval trên Re-DocRED sạch** (không phải DocRED nhiễu) + Ign F1.

### 4.5 Checkpoint / tương thích
- [ ] Checkpoint Qwen cũ **KHÔNG tương thích** (hidden 4096→1024, router/expert đổi shape). Train lại từ đầu.
  Code dùng `strict=False` nên không crash, nhưng đừng kỳ vọng load được weight cũ.

---

## 5. Lệnh chạy mẫu (tham khảo, KHÔNG chạy trong handoff)
```
python moe.py --stage train --model-id roberta-large --max-seq-length 512 \
  --expert-depths "1,2,4" --num-experts 3 --capacity-factor 1.25 \
  --distant-mix-ratio 0.3 --distant-topk 2
```

## 6. Định nghĩa "done" cho từng pillar
- Backbone: train chạy không lỗi với roberta-large, LoRA trainable > 0, F1 dev hợp lý.
- AGGCN expert: `AGGCNExpert` thay `GraphExpert`, interface khớp; ablation AGGCN vs GraphExpert có chênh.
- Adaptive-depth: heatmap depth×hop cho thấy tương quan; ablation đồng nhất vs dị-độ-sâu có chênh.
- PU: vượt baseline closed-world trên Re-DocRED; ablation PU on/off rõ ràng.
- EC: pool theo doc; bỏ LB-loss; cân bằng tải tự động; không sập recall quan hệ hiếm.

## 7. Tham chiếu lý thuyết
Tất cả trong [RESEARCH_DESIGN_PU_MoE.md](RESEARCH_DESIGN_PU_MoE.md): §1 lõi PU, §2 OQ2 (PU×EC),
§3 OQ3 (nhiễu FP+FN), §8 hướng đã chốt, §9 thiết kế adaptive-depth, §10 AGGCN expert, §6 nguồn (đã verify).
