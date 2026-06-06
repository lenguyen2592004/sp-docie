# Research Design — PU Learning × Expert-Choice MoE × AGGCN cho DocRE dưới Distant Supervision

> Tài liệu thiết kế nghiên cứu. Mục tiêu: giải bài toán **false-negative / incomplete-labeling** ở
> document-level relation extraction (DocRED / Re-DocRED) trong kiến trúc Mixture-of-Experts dùng
> Expert-Choice routing với AGGCN experts và MIL top-k distant-supervision denoising.
>
> Trạng thái: đề xuất (chưa hiện thực hóa). Chưa chạm code production tới khi được duyệt.

---

## 0. Bối cảnh & gap

- **Gap lớn nhất:** nhiễu trong DocRE chủ yếu là **false-negative (FN) / nhãn không đầy đủ**, không phải false-positive.
  - Re-DocRED (EMNLP 2022): re-annotate 4.053 doc → **+~13 F1**.
  - Positive cực hiếm: ~3.18% cặp (DocRED) / 7.09% (Re-DocRED) — do số cặp tăng bậc hai theo số entity.
- **Pipeline hiện tại xử lý sai loại nhiễu:**
  - `clean_distant_supervision_data` (moe.py:823) — MIL top-k chỉ loại **FP**, không khôi phục FN.
  - Negative sampling (moe.py:3075-3079) + `build_multi_hot_targets` (moe.py:899) coi mọi cặp không-gold = **âm** (closed-world) → **bơm FN vào loss**.
- **Novelty (đã verify, độ tin cậy cao):** tổ hợp **PU × Expert-Choice MoE × AGGCN × DS-DocRE chưa ai làm**.
  Hàng xóm gần nhất: GMoE (NeurIPS 2023, GCN-expert nhưng token-choice, không PU/RE); SSR-PU/P³M/TTM-RE (PU cho DocRE nhưng không MoE/graph).
- **Bar SOTA phải vượt:** TTM-RE (ACL 2024) — **84.01 F1** trên Re-DocRED (Human+Distant).

---

## 1. Lõi phương pháp (có nền toán)

Thay closed-world negatives bằng **nnPU + SSR-PU squared-ranking loss, áp per-relation**.

### 1.1 nnPU (bắt buộc — non-negative)
Risk PU per-relation r:

```
R_r(f) = π_r · E_P[ℓ⁺(f)] + max(0,  E_U[ℓ⁻(f)] − π_r · E_P[ℓ⁻(f)])
```

- Unbiased PU (du Plessis, ICML 2015): term âm kéo empirical risk < 0 → overfit nặng với model sâu.
- nnPU (Kiryo, NeurIPS 2017): kẹp `max(0,·)` → cho phép dùng AGGCN/deep experts với ít positive.

### 1.2 SSR-PU (EMNLP 2022) — thích nghi multi-label + prior-shift
- **Shift:** hiệu chỉnh class-prior shift per-relation (DocRED recommend-revise → quan hệ phổ biến đã gán).
- **Squared ranking loss** với điểm none-class f₀ làm ngưỡng thích nghi:

```
ℓ_SR(f_i, y_i) = ¼ · ( y_i·(f_i − f_0) − margin )²
```

  đã chứng minh **Bayesian-consistent** với multi-label ranking metric.

### 1.3 Tích hợp vào code (additive, không đụng core EC/AGGCN)
1. Bỏ closed-world neg sampling (moe.py:3075-3079); cặp không-gold → **unlabeled**, không phải negative.
2. Thay focal loss cho doc DS bằng **nnPU + squared-ranking** per-relation, f₀ = điểm none-class.
3. Giữ MIL top-k (moe.py:823) làm tầng khử **FP** → kết hợp PU (FN) = khử nhiễu **bất đối xứng**.
4. EC routing + AGGCN giữ nguyên.

---

## 2. OQ2 — PU learning tương tác với Expert-Choice routing

> ⚠️ Suy luận từ nguyên lý (chưa có tiền lệ) → *giả thuyết thiết kế cần thực nghiệm* = đóng góp mới.

**Vấn đề:** `max(0,·)` của nnPU cần ước lượng `E_U` trên tập đại diện; nhưng EC chia batch thành
**bucket thiên lệch** (expert chọn cặp giống nhau) → ước lượng risk bị méo.

### Quyết định thiết kế
1. **Clamp nnPU tính GLOBAL trên toàn routing pool, không per-expert.**
   - Lý do: bucket EC là mẫu con thiên lệch → `E_U` per-bucket sai; quan hệ hiếm có thể 0-positive trong bucket.
   - Expert chỉ quyết định *biểu diễn*; risk lắp ráp ở **đầu ra classifier trên toàn pool**.
2. **Tách gradient: term risk-âm (ascent) KHÔNG backprop vào router gate.**
   - Lý do: gradient ascent của nnPU có thể phá routing. Dùng `stop-gradient` lên router trong nhánh ascent.
   - Term âm chỉ cập nhật expert + classifier.
3. **Bỏ `switch_load_balance_loss` (moe.py:3133)** — EC tự cân bằng. Thêm entropy reg nhỏ trên router làm phao chống collapse.
4. **Bảo vệ positive hiếm khỏi capacity drop** — must-keep positive ở tầng routing (giống `limit_candidates_preserve_must_keep`, moe.py:907).
5. **Pool lớn (route theo doc / multi-doc)** — cộng hưởng: tốt cho cả EC (có gì để chọn) lẫn nnPU (ước lượng E_U ổn định).

### Kiểm chứng
- Tần suất term-âm bị clamp; entropy router theo epoch; ma trận expert×relation.
- Ablation: clamp global vs per-expert; có/không stop-grad router.

---

## 3. OQ3 — Nhiễu bất đối xứng FP + FN đồng thời

> ✅ Có nền lý thuyết. Bản chất: PU chuẩn giả định positive sạch, nhưng DS làm positive nhiễm FP → "PU under noisy positives".

### Ba hướng (kết hợp)
**A. Decouple theo loại nhiễu (chính, khớp code):**
- FP trong P → MIL top-k (moe.py:823) + **confidence weight** (trường `weight` trong train_distant_clean.json).
  Thay `E_P[ℓ⁺]` bằng kỳ vọng có trọng số: `Σ w_i·ℓ⁺(f_i) / Σ w_i`.
- FN trong U → nnPU/SSR-PU.
- **Lý thuyết:** sai lệch của "PU nhiễm positive" so với PU sạch bị chặn **tuyến tính theo tỉ lệ FP η** (bias = O(η)).
  → MIL giảm η ⇒ thắt chặt bound. (Mệnh đề cần chứng minh chính thức — xem §5.)

**B. Lớp robust — symmetric loss:**
- Charoenphakdee et al. (ICML 2019): loss đối xứng `ℓ(z)+ℓ(−z)=const` robust với nhãn nhiễu,
  có classification-calibration + excess-risk bound + AUC-consistency.
- Dùng surrogate đối xứng/bị chặn trong PU risk để residual FP không lấn át.

**C. Mô hình hóa tường minh (stretch, novelty toán cao nhất):**
- Noisy-positive PU / PUbN: mô hình tỉ lệ FP η_r⁺, FN η_r⁻ per-relation như kênh nhiễu class-conditional → debias.
- Refs: arXiv 1606.08561, PUbN (ICML 2019), arXiv 2103.04685.

### Bất đối xứng
- π_r (FN) và η_r (FP) đặt riêng per-relation → điều trị bất đối xứng + long-tail.
- Ngưỡng none-class f₀ vốn đã tạo quyết định bất đối xứng.

### Kiểm chứng
- Eval trên **Re-DocRED test sạch** + **Ign F1**.
- Ablation tách: chỉ-MIL (FP) / chỉ-PU (FN) / MIL+PU / +symmetric / +noise-modeling.
- Quét độ nhạy theo η để kiểm bound O(η).

---

## 4. Kế hoạch thí nghiệm ưu tiên

1. Reproduce baseline (focal + closed-world) trên Re-DocRED → mốc.
2. +nnPU/SSR-PU loss (giữ kiến trúc) → đo mức tăng do PU.
3. +EC routing (mở pool theo doc).
4. +AGGCN expert.
5. Ablation: PU on/off × EC vs token-choice × AGGCN vs transformer-expert × MIL on/off × symmetric on/off.
6. So bar TTM-RE 84.01; đối chứng COMM (AAAI 2025), FM-RKD (IPM Q1).

---

## 5. Việc toán cần làm trước khi viết paper
- Chứng minh bound bias = O(η) cho "PU nhiễm positive" + điều kiện MIL giảm η.
- Kiểm consistency của squared-ranking dưới global-clamp trong setting MoE.

---

## 6. Nguồn (đã verify qua deep-research, 3-vote adversarial)
- SSR-PU — EMNLP 2022: https://aclanthology.org/2022.emnlp-main.276/ · https://arxiv.org/abs/2210.08709
- P³M — AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/29888/31550 · https://arxiv.org/abs/2306.14806
- TTM-RE — ACL 2024: https://aclanthology.org/2024.acl-long.26/
- Re-DocRED — EMNLP 2022: https://aclanthology.org/2022.emnlp-main.580/
- nnPU — Kiryo NeurIPS 2017: https://arxiv.org/pdf/1703.00593
- GMoE — NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/file/9f4064d145bad5e361206c3303bda7b8-Paper-Conference.pdf
- AGGCN — ACL 2019: https://arxiv.org/abs/1906.07510
- Symmetric losses — Charoenphakdee ICML 2019: https://arxiv.org/abs/1901.09314
- PUbN: https://openreview.net/pdf?id=rJzLciCqKm
- PU via noisy labels: https://arxiv.org/abs/2103.04685
- Class prior from noisy positives: https://arxiv.org/pdf/1606.08561
- COMM — AAAI 2025: https://arxiv.org/pdf/2503.13885
- FM-RKD — IPM Q1: https://www.sciencedirect.com/science/article/abs/pii/S0306457323002704

## 8. HƯỚNG ĐÃ CHỐT — Adaptive-Depth Experts + PU + Residual (trục bài A*)

> Quyết định: trục chính = **heterogeneous-depth experts (adaptive computation theo độ khó suy luận)**;
> pillar 2 = **PU learning** (xử lý false-negative); stabilizer = **shared/residual expert**.
> Residual/shared-expert là *thành phần ổn định*, KHÔNG phải đóng góp tiêu đề.

### 8.1 Ba trụ & vì sao
1. **Adaptive-depth experts (trục):** expert KHÔNG đồng nhất — vài expert nông (intra-sentence, 1-hop),
   vài expert sâu (cross-sentence, multi-hop AGGCN). EC routing gửi cặp "khó" tới expert sâu.
   - *Gap:* khoảng cách F1 intra- vs inter-sentence (điểm yếu kinh điển DocRE).
   - *Vì sao cũ hỏng:* 3 GraphExpert giống hệt (moe.py:1324-1327) phí compute cho cặp dễ, thiếu chiều sâu cho cặp khó.
   - *Vì sao sửa được:* EC cho mỗi cặp số expert biến thiên; ghép expert dị thể → adaptive computation theo độ khó.
2. **PU learning (pillar 2):** thay closed-world negatives bằng nnPU + SSR-PU squared-ranking (xem §1-3).
   Lấp FN — gap #1.
3. **Shared/residual expert (stabilizer):** một expert "shared" luôn bật (dense path) giữ pattern quan hệ
   head/phổ biến; routed experts lo phần đặc thù/tail → hỗ trợ long-tail + ổn định gradient.
   Tham chiếu: DeepSeekMoE shared+routed; residual MoE.

### 8.2 Thiết kế cụ thể
- **Tập expert:** 1 shared (luôn áp, residual) + N routed experts với **độ sâu khác nhau**
  (vd N=3: nông=1 lớp, vừa=2 lớp, sâu=4 lớp AGGCN). Output = shared_out + Σ routed_out (residual add).
- **Tín hiệu router độ khó:** thêm đặc trưng phụ trợ vào đầu vào router để học "độ khó":
  số hop giữa h,t trên entity-graph, số câu evidence, khoảng cách token nhỏ nhất
  (tái dùng `_pair_fast_score`, moe.py:635). → router dễ học gửi cặp khó tới expert sâu.
- **EC routing:** route theo pool document (mở pool, §2.5); bỏ switch-LB-loss; entropy reg nhỏ phòng collapse.
- **PU loss:** nnPU global-clamp (§2.1) + squared-ranking per-relation; MIL top-k + weight cho FP (§3A).

### 8.3 Ablation (để đủ tầm A*)
- Expert đồng nhất vs adaptive-depth (cô lập đóng góp trục).
- Có/không tín hiệu độ khó vào router.
- Có/không shared/residual expert.
- PU on/off; MIL on/off; symmetric loss on/off.
- Phân tích: F1 theo độ khó (intra vs 1-hop vs ≥2-hop) — chứng minh expert sâu thắng ở cặp khó.
- Phân tích routing: cặp khó có thực sự được gửi tới expert sâu không (heatmap depth × hop-distance).

### 8.4 Lịch huấn luyện đề xuất
1. Warm-up: train với loss chuẩn (focal) để expert/ router ổn định.
2. Bật PU loss (nnPU + squared-ranking) + MIL/weight cho FP.
3. Bật adaptive-depth routing đầy đủ + entropy reg.
4. Eval trên Re-DocRED sạch (Ign F1); so bar TTM-RE 84.01.

### 8.5 Rủi ro & kiểm soát
- Router không học được "độ khó" → đã tiêm tín hiệu phụ trợ (8.2).
- Expert sâu nuốt hết cặp (mất cân bằng) → EC capacity + must-keep + entropy reg.
- Residual path lấn át routed (router lười) → theo dõi tỉ trọng đóng góp shared vs routed.

---

## 9. THIẾT KẾ KỸ THUẬT (1) — Adaptive-Depth Experts + Difficulty-Aware Router

### 9.1 Nguyên tắc thiết kế khớp data
- DocRED: graph entity nhỏ (≤15 node), cặp **intra-sentence dễ** vs **cross-sentence/multi-hop khó**.
- → Tín hiệu độ khó **rút trực tiếp từ subgraph** (không cần đổi call-site): hop-distance giữa h,t,
  số node, số cạnh, cờ nối-trực-tiếp. Cặp khó (hop xa) → expert sâu.

### 9.2 Các thành phần
1. **GraphExpert dị-độ-sâu:** tái dùng `GraphExpert(num_layers=d)`; tạo N expert với depth khác nhau,
   ví dụ `expert_depths=[1,2,4]` (nông→sâu).
2. **Shared/residual expert:** một `GraphExpert` (depth vừa, vd 2) **luôn áp cho mọi cặp**;
   output cuối = `shared_out + routed_out` (residual). Giữ pattern quan hệ head + ổn định gradient.
3. **DifficultyAwareRouter:** input = `[pair_features (hidden*2)] ⊕ [difficulty_feats]`;
   top-1 (giai đoạn đầu) → sau nâng lên Expert-Choice khi mở pool theo doc.

### 9.3 Difficulty features (tính trong forward, từ mỗi subgraph)
- `hop_norm` = shortest-path-hops(h,t) / HOP_MAX (disconnected → 1.0)
- `nodes_norm` = num_nodes / 15
- `edges_norm` = num_edges / (15*14)
- `direct` = 1.0 nếu hop==1
→ vector 4 chiều, chuẩn hóa [0,1]. (Đánh đúng "độ khó suy luận" mà không cần evidence — vốn chỉ có lúc train.)

### 9.4 Forward (residual + heterogeneous routing)
```
diff = difficulty_feats(subgraphs)            # (B,4)
logits, top1 = router(pair_features, diff)
shared_out = shared_expert(all_graphs)        # (B, expert_dim*2), luôn áp
routed_out = 0; per expert e: routed_out[idx_e] = expert_e(graphs_e) * gate_e
pair_emb = shared_out + routed_out            # residual
logits_rel = classifier(pair_emb)
```

### 9.5 Tham số mới
- `--expert-depths "1,2,4"` (nếu set → num_experts = len); else sinh từ `--num-experts`.
- `--use-shared-expert` (mặc định bật).

### 9.6 Vì sao hợp data & đủ A*
- Adaptive computation theo độ khó suy luận = chủ đề được trọng vọng; tín hiệu độ khó lấy từ chính
  cấu trúc graph DocRED → rẻ, không thêm call-site.
- Shared+residual hỗ trợ long-tail (head ở shared, tail ở routed) + ổn định.
- Ablation: đồng nhất vs dị-độ-sâu; có/không diff-feats; có/không shared; heatmap depth×hop.

---

## 7. Caveat
- Số F1 là tác giả tự báo, chưa tái lập độc lập.
- "SOTA" là tương đối thời điểm; mốc 2026 có thể có hệ mạnh hơn.
- OQ2 chưa có tiền lệ → rủi ro thực nghiệm cao; OQ3 có lý thuyết → rủi ro trung bình.
