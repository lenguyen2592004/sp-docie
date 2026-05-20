#!/usr/bin/env python3
"""Test device assignment for model."""

import torch
from transformers import AutoModelForCausalLM

print("Testing device assignment...")
print(f"CUDA available: {torch.cuda.is_available()}")

DEVICE = "cpu"
print(f"Target DEVICE: {DEVICE}")

print("Loading model...")
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

print(f"Model loaded. Default device: {next(base_model.parameters()).device}")

base_model = base_model.to(DEVICE)
print(f"After .to(DEVICE): {next(base_model.parameters()).device}")

# Test that model stays on CPU
dummy_input = torch.randint(0, 1000, (1, 10)).to(DEVICE)
output = base_model(dummy_input)
print(f"After forward pass: {next(base_model.parameters()).device}")
print(f"Output device: {output.logits.device}")

print("✓ Test passed!")
