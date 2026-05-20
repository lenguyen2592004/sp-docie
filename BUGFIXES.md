## Bug Fixes Summary

### Fixed Issues

1. **NameError: max_pairs not defined (line 1081)**
   - Added `--max-pairs-per-doc` argument (default=50)
   - Defined `max_pairs = args.max_pairs_per_doc` before training loop

2. **DGLError: dtype mismatch (float32 vs bfloat16)**
   - Changed `torch.zeros()` calls to explicitly use `llm_embeddings.dtype`
   - Ensures all node features have consistent bfloat16 dtype
   - Fixed in `build_pair_subgraph()` method (lines 380, 382)

3. **CUDA OutOfMemoryError during pair encoding**
   - Reduced `pair_batch_size` from 4 to 2
   - Added aggressive memory clearing after LLM forward pass
   - Added `del` statements and `torch.cuda.empty_cache()` calls
   - Reduced `--max-pairs-per-doc` default from 100 to 50

### Changes Made

#### /workspace/moe.py

**Line 825:** Added argparse argument
```python
parser.add_argument('--max-pairs-per-doc', type=int, default=50, help='Maximum number of (h, t) pairs per document to process.')
```

**Line 1065:** Variable definition before training loop
```python
max_pairs = args.max_pairs_per_doc
```

**Lines 380, 382:** Fixed dtype consistency
```python
node_feats.append(torch.zeros(llm_embeddings.shape[-1], dtype=llm_embeddings.dtype, device=llm_embeddings.device))
node_feats = torch.stack(node_feats) if node_feats else torch.zeros((1, llm_embeddings.shape[-1]), dtype=llm_embeddings.dtype, device=llm_embeddings.device)
```

**Line 1104:** Reduced pair batch size
```python
pair_batch_size = 2  # Small batch to avoid OOM with per-pair markers
```

**Lines 1118-1120:** Memory cleanup after LLM forward
```python
del pair_out, pair_enc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Lines 1172-1175:** Memory cleanup after backward pass
```python
del batch_loss, ce_loss, switch_loss, loss_fgw, fgw_term, logits, router_logits, sgs, adjs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Verification

All syntax errors cleared:
- ✓ No Python compilation errors
- ✓ Model loads successfully
- ✓ Data loads (500 train, 200 eval docs)
- ✓ First loss computed (21.7486)

### Next Steps

Training should now run without errors. The OOM issue has been addressed with:
- Smaller batch sizes
- Aggressive memory clearing
- Reduced max pairs per document

Run training with:
```bash
python moe.py --stage train --epochs 1
```
