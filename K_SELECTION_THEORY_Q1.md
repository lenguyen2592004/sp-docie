# Q1-ready: Chọn `k_MIL`, `k_hop`, và Threshold **không Grid Search** (DocRED / DocIE)

Tài liệu này là **SOP cho coding-agent** + **phần viết-up có thể đưa vào paper**.
Mục tiêu: chọn tham số bằng **lý thuyết + đo lường tĩnh (static measurement)**, tránh “thử sai” trên lưới `(k_MIL, k_hop, τ)`.

> **Quan trọng (tính tổng quát):** Không tồn tại quy tắc chung có thể **đồng thời** làm _Precision, Recall, Micro-F1, Macro-F1, Accuracy_ đều đạt cực đại trừ khi mô hình phân loại là hoàn hảo. Vì vậy ta cần đặt bài toán như **tối ưu đa mục tiêu** (Pareto-optimal) hoặc **tối ưu một utility tổng hợp** có chứng minh.

---

## 0) Những chỗ “hook” trong code hiện tại

- MIL top-k denoising DS: `clean_distant_supervision_data(..., top_k=2)` ở [moe.py#L576](moe.py#L576).
  - Logic hiện tại: gom nhãn DS theo bag `(h,t)` rồi `bag[:max(1, int(top_k))]` ([moe.py#L621](moe.py#L621)).
- Dự đoán multi-label threshold: `predict_multi_label_relations(logits, threshold=0.5)` ở [moe.py#L803](moe.py#L803).
- Micro Precision/Recall/F1 (fact-level) đang dùng: `compute_fact_f1` ở [moe.py#L810](moe.py#L810).
- K-hop subgraph: `DocREDGraphBuilder.build_pair_subgraph(..., k_hop=1)` ở [moe.py#L966](moe.py#L966).
  - Có **budget cứng**: nếu số node > 15 thì cắt xuống 15 ([moe.py#L1007](moe.py#L1007)). Điều này phải được đưa vào lý thuyết như một ràng buộc.
- Adaptive threshold hiện tại (evaluation): `decision_threshold = clamp(0.5 * threshold_scale)` ở [moe.py#L1689](moe.py#L1689).

---

## 1) Vì sao “adaptive threshold” **không làm mất tính tổng quát**?

### 1.1. Phân biệt 2 thứ

- **Model parameters** (trọng số mạng): học từ train.
- **Decision rule** (ngưỡng τ, top-k, budget k-hop): là **bước hậu xử lý** để map score → dự đoán.

Trong lý thuyết thống kê, chọn decision rule để tối ưu một hàm mục tiêu đo lường (_F1, accuracy, cost-sensitive loss…_) là hợp lệ và chuẩn.

### 1.2. Khi nào bị reviewer chê “overfitting”?

Nếu bạn “tối ưu ngưỡng” bằng cách thử hàng trăm giá trị và báo cáo điểm tốt nhất **trên chính test** hoặc **dùng dev như test** mà không tách phần calibration.

### 1.3. Cách trình bày Q1

- Nêu rõ đây là **utility-driven decision rule**.
- Chứng minh (phần 3) rằng τ\* được suy ra từ điều kiện tối ưu của **kỳ vọng metric** dưới giả định score được hiệu chuẩn (calibrated).
- Thực nghiệm chỉ đóng vai trò **xác nhận** (validation), không phải “grid search”.

---

## 2) Tối ưu đa mục tiêu: không thể “maximize all” nếu không định nghĩa utility

### 2.1. Mệnh đề (trực giác + chứng minh ngắn)

Với score liên tục, tăng τ thường làm:

- Precision ↑
- Recall ↓

Do đó không thể đồng thời tối đa hóa cả Precision và Recall (và do đó cũng không thể đồng thời tối đa hóa mọi dạng F1/accuracy) trừ trường hợp phân tách hoàn hảo.

### 2.2. Cách làm đúng để vẫn “đạt hiệu suất cao nhất”

Chọn một trong hai khuôn sau (đều tổng quát và publishable):

**(A) Utility tổng hợp (khuyến nghị khi paper muốn “theoretical optimality”)**

Ta định nghĩa một utility scalar:

\[
U(\theta) = w*{\mu} \cdot F1*{micro}(\theta) + w*{M} \cdot F1*{macro}(\theta) + w*{A} \cdot Acc(\theta) - \lambda*{C} \cdot Cost(\theta) - \lambda\_{N} \cdot Noise(\theta)
\]

Trong đó \(\theta = (k*{MIL}, k*{hop}, \tau)\) hoặc mở rộng per-relation \(\tau_r\).

**(B) Ràng buộc (khuyến nghị khi muốn “an toàn metric”)**

\[
\max\ F1*{micro}(\theta)\ \ \text{s.t.}\ \ Acc(\theta)\ge A_0,\ \ F1*{macro}(\theta)\ge M_0,\ \ Cost(\theta)\le C_0
\]

Giải bằng Lagrangian → điều kiện tối ưu có dạng “marginal gain = marginal cost”.

### 2.3. Định nghĩa metric (để tránh reviewer bắt bẻ)

**Fact-level (đúng với code hiện tại):** dự đoán là một tập facts \((doc,h,t,r)\).

- Precision \(P\) = \(\frac{TP}{TP+FP}\)
- Recall \(R\) = \(\frac{TP}{TP+FN}\)
- Micro-F1 = \(\frac{2PR}{P+R}\)

**Macro-F1:** trung bình F1 theo relation (thường cần \(\tau_r\) theo relation để tối ưu hợp lý, vì long-tail).

**Accuracy:** cần nói rõ _universe_ dùng để đếm TN.

- Nếu lấy _mọi_ \((h,t,r)\) có thể có trong doc làm negative → TN cực lớn ⇒ accuracy gần 1 và **không có ý nghĩa**.
- Khuyến nghị: nếu muốn báo cáo accuracy, hãy định nghĩa rõ candidate universe (ví dụ: tập candidate pairs từ `CandidateGenerator` + giới hạn `max_pairs_per_doc`, và trên đó xét mọi relation). Khi đó accuracy là hợp lệ nhưng **phụ thuộc candidate-generation**, cần mô tả trong paper.

---

## 3) Chọn threshold \(\tau\) bằng **lý thuyết** (không grid search)

Phần này cho phép thay thế `--adaptive-threshold-scale` bằng **\(\tau^\*\) suy ra từ dữ liệu**.

### 3.1. Thiết lập xác suất (tổng quát)

Mỗi fact \((doc, h, t, r)\) là một biến nhị phân \(Y\in\{0,1\}\).
Mô hình cho score/probability \(S\in[0,1]\) (ví dụ `sigmoid(logit)`).

Gọi:

- \(\pi = P(Y=1)\)
- \(f\_+(s)\) là mật độ của \(S\) khi \(Y=1\)
- \(f\_-(s)\) là mật độ của \(S\) khi \(Y=0\)

Với ngưỡng \(\tau\):

\[
\begin{aligned}
TP(\tau) &= \pi N \int*{\tau}^{1} f*+(s)\,ds\\
FP(\tau) &= (1-\pi) N \int*{\tau}^{1} f*-(s)\,ds\\
FN(\tau) &= \pi N \int*{0}^{\tau} f*+(s)\,ds
\end{aligned}
\]

\[
F1(\tau)=\frac{2TP(\tau)}{2TP(\tau)+FP(\tau)+FN(\tau)}
\]

Từ đây suy ra luôn kỳ vọng Precision/Recall:
\[
P(\tau)=\frac{TP(\tau)}{TP(\tau)+FP(\tau)},\qquad R(\tau)=\frac{TP(\tau)}{TP(\tau)+FN(\tau)}.
\]

### 3.2. Theorem (F1-optimal threshold condition)

**Giả định A (calibration):** \(S\) là xác suất hậu nghiệm: \(S=P(Y=1\mid X)\).

**Định lý 1 (điều kiện tối ưu cho F1 kỳ vọng):**
Một ngưỡng tối ưu \(\tau^\*\) thỏa điều kiện cố định (fixed-point):

\[
P(Y=1\mid S=\tau^_) = \frac{F1(\tau^_)}{2}
\]

**Proof sketch (ý tưởng):** Viết F1 theo \(TP,FP,FN\); đạo hàm theo \(\tau\) dùng quy tắc Leibniz (
\(\frac{d}{d\tau}\int*{\tau}^1 f(s)ds=-f(\tau)\)
); đặt \(\frac{dF1}{d\tau}=0\) và thay Bayes:
\(P(Y=1\mid S=\tau)=\frac{\pi f*+(\tau)}{\pi f*+(\tau)+(1-\pi)f*-(\tau)}\).

> Ý nghĩa: τ\* không cần thử lưới; chỉ cần **ước lượng \(f*+,f*-\)** và giải phương trình.

#### Mở rộng: tối ưu \(F\_\beta\) (tổng quát cho “ưu tiên Recall” hay “ưu tiên Precision”)

Định nghĩa:
\[
F*\beta(\tau)=\frac{(1+\beta^2)\,TP(\tau)}{\beta^2 P*{pos}+TP(\tau)+FP(\tau)}
\]
với \(P\_{pos}=\pi N\) là số positive kỳ vọng.

**Định lý 1b (điều kiện tối ưu cho \(F\_\beta\) kỳ vọng):**
\[
P(Y=1\mid S=\tau^_) = \frac{F\_\beta(\tau^_)}{1+\beta^2}.
\]

Vì vậy muốn “đẩy Recall” bạn dùng \(\beta>1\); muốn “đẩy Precision” dùng \(\beta<1\). Tất cả vẫn là cùng một khuôn: **giải fixed-point**.

### 3.3. Cách đo lường (1-pass) để giải τ\*

**Dữ liệu:** `dev.json` hoặc `train_annotated.json`.

**Bước đo lường:**

1. Chạy model **một lần** để lấy `probs` cho tất cả facts ứng viên.
2. Tạo 2 tập score:
   - `S_pos`: score của facts thật (gold facts)
   - `S_neg`: score của facts sai (non-gold)
3. Fit phân phối tham số (khuyến nghị): Beta(\(\alpha,\beta\)) cho mỗi tập:
   - \(f*+(s)=\text{Beta}(\alpha*+,\beta\_+)\)
   - \(f*-(s)=\text{Beta}(\alpha*-,\beta\_-)\)
4. Tính \(F1(\tau)\) theo công thức tích phân của Beta (hàm Beta bất toàn).
5. Giải \(g(\tau)=P(Y=1\mid S=\tau)-F1(\tau)/2=0\) bằng bisection (đảm bảo hội tụ).

### 3.4. Macro-F1 và per-relation threshold

- **Micro-F1**: một \(\tau\) chung cho mọi relation.
- **Macro-F1**: thường tốt hơn nếu dùng \(\tau_r\) riêng cho mỗi relation \(r\) (đặc biệt long-tail).

Tổng quát: lặp lại Mục 3.1–3.3 cho từng relation để lấy \(\tau_r^\*\).

### 3.5. Accuracy (và cost-sensitive)

Accuracy tối ưu phụ thuộc vào **tỉ lệ lớp** và **chi phí sai**.

- Nếu chi phí FP và FN đối xứng và score calibrated → \(\tau=0.5\) tối ưu Bayes.
- Nếu FP đắt hơn FN → \(\tau\) tăng theo tỉ số chi phí.

### 3.6. Tối ưu đồng thời Precision/Recall (không grid)

Nếu paper muốn nhấn mạnh “tối ưu nhiều metric”, hãy trình bày như bài toán ràng buộc:

**Bài toán:**
\[
\max\ R(\tau)\ \ \text{s.t.}\ \ P(\tau)\ge P_0
\]

**Lý thuyết:** theo Neyman–Pearson, nghiệm tối ưu là threshold theo likelihood ratio (tương đương threshold theo posterior nếu score calibrated). Trong thực thi, vì \(P(\tau)\) (kỳ vọng) thường tăng theo \(\tau\), bạn chỉ cần:

1. giải \(P(\tau)=P*0\) bằng bisection để tìm \(\tau*{min}\)
2. chọn \(\tau^\*=\tau\_{min}\) (vì nó cho Recall lớn nhất trong tập feasible)

Tương tự, nếu muốn \(R(\tau)\ge R_0\) và tối đa Precision thì giải \(R(\tau)=R_0\).

---

## 4) Chọn \(k\_{MIL}\) cho DS cleaning bằng “marginal utility” (không thử cặp)

Trong code hiện tại, DS cleaning giữ `top_k` nhãn trong mỗi bag `(h,t)`.
Điều ta cần: chọn \(k\_{MIL}\) **tối ưu kỳ vọng** theo utility đã chọn.

### 4.1. Mô hình hóa MIL top-k

Với mỗi bag \(b\), các instance được sắp theo score giảm dần:
\(s*{b,1}\ge s*{b,2}\ge\dots\ge s\_{b,n_b}\).

Gọi \(q(s)=P(\text{instance đúng}\mid s)\) (hàm calibration trên DS-score).

Khi giữ top-k:
\[
\mathbb{E}[\text{TrueKept}(k)] = \sum*b \sum*{r=1}^{\min(k,n*b)} q(s*{b,r})
\]
\[
\mathbb{E}[\text{Kept}(k)] = \sum_b \min(k,n_b)
\]
\[
\mathbb{E}[\text{NoiseKept}(k)] = \mathbb{E}[\text{Kept}(k)]-\mathbb{E}[\text{TrueKept}(k)]
\]

### 4.2. Utility và điều kiện tối ưu biên

Chọn utility đơn giản (tổng quát):
\[
U(k)=\alpha\,\mathbb{E}[\text{TrueKept}(k)]-\beta\,\mathbb{E}[\text{NoiseKept}(k)]-\gamma\,\mathbb{E}[\text{Cost}(k)]
\]

**Quy tắc tối ưu biên (mang tính “chứng minh được”):**
Tăng từ k → k+1 có lợi khi
\[
\Delta U(k) \ge 0\ \Longleftrightarrow\ \mathbb{E}[q(s_{b,k+1})] \ge \frac{\beta+\gamma\,\Delta Cost}{\alpha+\beta}
\]

Nói cách khác: **chỉ thêm instance hạng (k+1) nếu xác suất đúng kỳ vọng đủ lớn**.

### 4.3. Cách đo lường q(s) (không cần train lại nhiều lần)

Có 2 lựa chọn:

**(A) Semi-supervised (khuyến nghị nếu có overlap doc/title)**

- Dùng `train_annotated.json` làm ground-truth.
- Với các doc trùng title trong DS (`train_distant.json`), đánh dấu nhãn DS nào “đúng” khi nó xuất hiện trong annotated labels.
- Fit \(q(s)\) bằng logistic / isotonic regression.

**(B) Unsupervised mixture (khi không có overlap)**

- Giả định score DS là mixture của signal/noise: \(f(s)=\pi f*+(s)+(1-\pi)f*-(s)\).
- Fit mixture (ví dụ 2-Beta) bằng EM.
- Tính posterior \(q(s)=P(signal\mid s)\) qua Bayes.

Sau khi có \(q(s)\), \(k\_{MIL}^\*\) suy ra trực tiếp từ điều kiện biên (4.2), không cần grid search.

> Gợi ý thực hành: thay “k cố định” bằng **ngưỡng DS** \(\tau*{DS}\) rồi đặt \(k_b=\#\{r:q(s*{b,r})\ge \tau*{DS}\}\). Nếu cần k cố định để đơn giản, lấy \(k*{MIL}^\*=\text{median}(k_b)\) hoặc quantile để ổn định.

---

## 5) Chọn \(k\_{hop}\) bằng xác suất bao phủ bằng chứng + ràng buộc bùng nổ đồ thị

### 5.1. Định nghĩa “coverage” (tổng quát)

DocRED có `evidence` ở mức sentence-id. Graph builder của repo xây đồ thị entity–entity bằng **co-occurrence trong cùng câu** (`build_pair_subgraph`).

Với một cặp entity (h,t) và một tập câu bằng chứng \(E\), định nghĩa tập “bridge entities” \(B(E)\): mọi entity xuất hiện trong các câu evidence.

Đặt:
\[
D = \max\_{u\in B(E)} \min\{dist(h,u), dist(t,u)\}
\]

Khi dùng k-hop neighborhood quanh {h,t}, ta “cover evidence” nếu mọi bridge entity cần thiết nằm trong subgraph.

Khi đó:
\[
C(k)=P(D\le k)
\]

### 5.2. Mô hình chi phí: bùng nổ theo branching factor

Với đồ thị có branching factor trung bình \(b>1\):
\[
N(k) \approx 2 + \sum\_{i=1}^{k} b^i = 2 + \frac{b^{k+1}-b}{b-1}
\]

Trong code hiện tại có budget cứng \(N(k)\le 15\) (cắt node khi >15), nên hiệu dụng:
\[
N\_{eff}(k)=\min(N(k), 15)
\]

### 5.3. Hai quy tắc chọn k_hop (đều “theory-first”)

**(Rule 1 — Quantile guarantee, sạch và dễ viết paper):**
Chọn
\[
k*{hop}^\* = \min\{\text{Quantile}*\alpha(D),\ k*{budget}\}
\]
trong đó \(\alpha\in[0.90,0.99]\) là mức đảm bảo coverage, và \(k*{budget}\) suy ra từ bất đẳng thức \(N(k)\le 15\).

**(Rule 2 — Marginal gain vs marginal cost):**
Chọn k tối đa hóa
\[
J(k)= C(k)-\lambda\log N*{eff}(k)
\]
Điều kiện tối ưu xấp xỉ:
\[
\Delta C(k) \approx \lambda\,\Delta \log N*{eff}(k)
\]

Rule 1 phù hợp khi bạn ưu tiên “guarantee coverage”. Rule 2 phù hợp khi bạn muốn “tối ưu utility tổng hợp”.

### 5.4. Đo lường D và b (static)

- Từ `train_annotated.json` hoặc `dev.json`:
  1. Build entity adjacency theo logic trong `build_pair_subgraph`.
  2. Với mỗi gold relation có evidence, tạo bridge set và tính D.
  3. Lấy phân phối của D → quantile.
- Tính b bằng trung bình degree hoặc trung bình số hàng xóm mới mỗi BFS layer.

---

## 6) Quy trình end-to-end (không grid search)

### Input

- `train_distant.json`: DS data (cho MIL cleaning)
- `train_annotated.json`: annotated train (cho đo lường graph/evidence và calibration)
- `dev.json`: validation (cho calibration threshold nếu muốn tách)

### Output

- \(k\_{MIL}^\*\)
- \(k\_{hop}^\*\)
- \(\tau^_\) hoặc \(\{\tau_r^_\}\)

### Pipeline

1. **Graph static analysis** → \(D\) distribution + \(b\) → \(k\_{hop}^\*\)
2. **DS score modeling** (semi/unsupervised) → \(q(s)\) → \(k*{MIL}^\*\) (hoặc \(\tau*{DS}\))
3. **Score calibration + threshold theory** → fit \(f*+,f*-\) → solve fixed-point → \(\tau^\*\)
4. Freeze \((k*{MIL}^\*, k*{hop}^_, \tau^_)\), train/infer **một lần** và báo cáo.

---

## 7) Phần “paper-ready” (English) — có thể copy vào Method/Theory

### Theorem: F1-optimal posterior threshold (sketch)

Assume a calibrated score \(S=P(Y=1\mid X)\) and a threshold decision rule \(\hat{Y}=\mathbb{1}[S\ge\tau]\). Let \(TP(\tau), FP(\tau), FN(\tau)\) denote expected counts. Then the expected F1 is

\[
F1(\tau)=\frac{2TP(\tau)}{2TP(\tau)+FP(\tau)+FN(\tau)}.
\]

Differentiating w.r.t. \(\tau\) and using Bayes’ rule yields a fixed-point condition for an optimal \(\tau^\*\):

\[
P(Y=1\mid S=\tau^_) = \frac{F1(\tau^_)}{2}.
\]

Hence, \(\tau^\*\) can be obtained without grid search by estimating class-conditional score densities \(f*+, f*-\) and solving the induced root equation.

### Principle: Marginal coverage vs. graph expansion

Let \(C(k)\) be evidence coverage probability and \(N(k)\) the expected subgraph size under a k-hop neighborhood. Under mild regularity, selecting \(k\) by maximizing \(C(k)-\lambda\log N(k)\) yields the marginal optimality condition \(\Delta C(k)\approx \lambda\,\Delta\log N(k)\), providing a data-measurable and general parameter selection rule.

---

## 8) Checklist cho coding-agent

- [ ] Implement static analyzer for \(D\) quantiles and branching factor \(b\) using the exact adjacency logic in `DocREDGraphBuilder`.
- [ ] Implement DS score posterior \(q(s)\) (prefer semi-supervised; fallback to 2-component mixture).
- [ ] Implement calibrated threshold solver (Beta fit + bisection) for micro-F1; optional per-relation for macro-F1.
- [ ] Add a `report.json` artifact containing: \(k*{MIL}^\*, k*{hop}^_, \tau^_\), quantiles, fitted parameters, and reproducibility seeds.

---

## 9) Gợi ý thực tế (không phá tính tổng quát)

- Nếu reviewer hỏi “tại sao Beta?”, câu trả lời: Beta là family tự nhiên trên \([0,1]\), đủ linh hoạt; có thể thay bằng KDE nhưng Beta giúp có công thức tích phân.
- Nếu reviewer hỏi “đây vẫn là tuning”, câu trả lời: không tuning lưới; đây là **closed-form optimality condition** + **one-shot estimation**.

---

## 10) Ghi chú tương thích repo

- `k_hop` hiện đang được truyền khi gọi `build_pair_subgraph` nhưng lời gọi trong `evaluate_model` đang dùng mặc định `k_hop=1` (không truyền tham số). Nếu bạn muốn áp dụng \(k\_{hop}^\*\), cần đảm bảo đường gọi có tham số.
- `decision_threshold` hiện là `0.5 * threshold_scale`; nếu chuyển sang \(\tau^_\), có thể thay `threshold_scale` bằng `2_\tau^\*` hoặc đổi hẳn tham số CLI.
