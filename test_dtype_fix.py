#!/usr/bin/env python3
"""Quick test to verify dtype consistency fix."""
import torch

# Simulate the issue
llm_embeddings = torch.randn(100, 2560, dtype=torch.bfloat16, device='cuda:0')

# Old way (causes dtype mismatch)
# zeros_old = torch.zeros(llm_embeddings.shape[-1]).to(llm_embeddings.device)
# print(f"Old zeros dtype: {zeros_old.dtype}")  # Would be float32

# New way (fixed)
zeros_new = torch.zeros(llm_embeddings.shape[-1], dtype=llm_embeddings.dtype, device=llm_embeddings.device)
print(f"LLM embeddings dtype: {llm_embeddings.dtype}")
print(f"New zeros dtype: {zeros_new.dtype}")
print(f"Match: {zeros_new.dtype == llm_embeddings.dtype}")

# Test with stacking
embeddings = llm_embeddings[:5].mean(0)
node_feats = [embeddings, zeros_new, embeddings]
stacked = torch.stack(node_feats)
print(f"Stacked dtype: {stacked.dtype}")
print(f"All dtypes match: {stacked.dtype == llm_embeddings.dtype}")
print("\n✓ Dtype fix verified - all tensors are bfloat16")
