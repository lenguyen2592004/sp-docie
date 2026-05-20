#!/usr/bin/env python3
import torch
import dgl

print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)

g = dgl.graph(([0], [0]), num_nodes=1)
try:
    g = g.to('cuda')
    print('dgl graph device:', g.device)
    print('DGL CUDA OK')
except Exception as exc:
    print('DGL CUDA FAIL:', type(exc).__name__, str(exc))
    raise
