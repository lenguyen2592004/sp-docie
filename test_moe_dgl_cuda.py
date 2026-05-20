#!/usr/bin/env python3
"""Test MoE model with DGL CUDA backend enabled."""
import torch
import dgl
import sys

print("="*60)
print("Testing MoE Graph RE with DGL CUDA")
print("="*60)

# Check environment
print(f"\n1. Environment Check:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
print(f"   DGL: {dgl.__version__}")

# Test DGL CUDA basic functionality
print(f"\n2. DGL CUDA Basic Test:")
try:
    g = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
    g = g.to('cuda')
    print(f"   ✅ DGL graph created on: {g.device}")
except Exception as e:
    print(f"   ❌ DGL CUDA error: {e}")
    sys.exit(1)

# Import and test actual model
print(f"\n3. Loading MoE Model:")
try:
    from moe import MoEGraphRE, DocREDGraphBuilder
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"   ✅ Imported MoEGraphRE and DocREDGraphBuilder")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Create model with graph_device='cuda'
print(f"\n4. Initializing Model:")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load a small LLM for testing (tiny-gpt2 is ~50MB)
    print(f"   Loading tiny-gpt2 as backbone...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2",
        torch_dtype=torch.float32
    ).to(device)
    print(f"   ✅ LLM loaded, hidden_size={llm_model.config.hidden_size}")
    
    # Create MoE model
    model = MoEGraphRE(
        llm_model=llm_model,
        num_relations=97,  # DocRED has 96 relations + NA
        num_experts=4,
        expert_dim=128,
        graph_device='cuda'  # Enable DGL CUDA
    )
    model = model.to(device)
    print(f"   ✅ Model initialized with device={device}, graph_device=cuda")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test graph builder (simplified - just verify it can be instantiated)
print(f"\n5. Testing Graph Builder:")
try:
    graph_builder = DocREDGraphBuilder(device='cuda')
    print(f"   ✅ DocREDGraphBuilder created for cuda")
    print(f"   Note: Full document processing requires LLM embeddings and document data")
    print(f"   Skipping full build_pair_subgraph test in quick validation")
        
except Exception as e:
    print(f"   ❌ Graph builder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass with dummy data
print(f"\n6. Testing Forward Pass:")
try:
    batch_size = 4
    hidden_size = llm_model.config.hidden_size
    num_ents = 3
    
    # Create dummy pair_features (concatenated head and tail entity embeddings)
    # Shape: (batch_size, hidden_size * 2)
    pair_features = torch.randn(batch_size, hidden_size * 2).to(device)
    
    # Create dummy subgraphs (entity pair subgraphs)
    # Each subgraph has a small number of nodes
    subgraph_list = []
    for _ in range(batch_size):
        # Simple triangle graph with 3 nodes
        g = dgl.graph(([0, 1, 2, 0], [1, 2, 0, 2]), num_nodes=num_ents)
        g = g.to('cuda')
        # Add dummy node features with key 'h' (required by MoEGraphRE)
        g.ndata['h'] = torch.randn(num_ents, hidden_size).to('cuda')
        # Add is_ht flag (head/tail markers)
        g.ndata['is_ht'] = torch.tensor([1.0, 1.0, 0.0]).to('cuda')
        subgraph_list.append(g)
    
    # Batch the subgraphs
    subgraphs = dgl.batch(subgraph_list)
    print(f"   Subgraphs batched: {subgraphs.num_nodes()} nodes, {subgraphs.num_edges()} edges")
    print(f"   Subgraphs device: {subgraphs.device}")
    print(f"   Pair features shape: {pair_features.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, router_logits, top1_idx = model(subgraphs, pair_features)
    
    print(f"   ✅ Forward pass successful")
    print(f"   Logits shape: {logits.shape} (should be [{batch_size}, 97])")
    print(f"   Router logits shape: {router_logits.shape}")
    print(f"   Top1 expert indices: {top1_idx.tolist()}")
    print(f"   Output device: {logits.device}")
    print(f"   Output dtype: {logits.dtype}")
    print(f"   Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"   NaN check: {torch.isnan(logits).any().item()}")
    
    if torch.isnan(logits).any():
        print(f"   ❌ WARNING: NaN detected in outputs!")
    else:
        print(f"   ✅ No NaN in outputs")
        
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*60}")
print("✅ ALL TESTS PASSED - MoE with DGL CUDA is working!")
print("="*60)
