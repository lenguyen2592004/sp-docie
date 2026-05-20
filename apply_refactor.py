import re

with open('/workspace/moe.py', 'r') as f:
    content = f.read()

original = content

# 1. Replace GraphBuilder class
old_graph_builder = '''class DocREDGraphBuilder:
    """
    Construct k-hop entity pair subgraphs from document.
    
    Graph nodes: entities
    Graph edges: co-occurrence in same sentence
    Subgraph: k-hop neighborhood around (h, t) pair
    """
    def __init__(self, device):
        self.device = str(device)

    def build_pair_subgraph(self, item, llm_embeddings, word_ids, h_id, t_id, k_hop=1):
        """
        Build subgraph for entity pair (h_id, t_id).
        
        Args:
            item: DocRED sample
            llm_embeddings: (seq_len, hidden_dim) from LLM
            word_ids: token-to-word alignment
            h_id, t_id: head and tail entity indices
            k_hop: neighborhood size
            
        Returns:
            g: DGL graph with node features
            adj: adjacency matrix (for FGW)
            h_idx, t_idx: indices of h,t in subgraph
        """
        vertex_set = item.get('vertex_set', item.get('vertexSet', []))
        num_entities = len(vertex_set)
        
        doc_adj = {}
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                sents_i = {m['sent_id'] for m in vertex_set[i]}
                sents_j = {m['sent_id'] for m in vertex_set[j]}
                if sents_i.intersection(sents_j):
                    if i not in doc_adj: doc_adj[i] = set()
                    if j not in doc_adj: doc_adj[j] = set()
                    doc_adj[i].add(j)
                    doc_adj[j].add(i)

        subgraph_nodes = {h_id, t_id}
        current_layer = {h_id, t_id}
        for _ in range(k_hop):
            next_layer = set()
            for node in current_layer:
                if node in doc_adj:
                    next_layer.update(doc_adj[node])
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
        
        sorted_nodes = sorted(list(subgraph_nodes))
        if len(sorted_nodes) > 15:
            sorted_nodes = [h_id, t_id] + [n for n in sorted_nodes if n not in [h_id, t_id]][:13]
            sorted_nodes = sorted(list(set(sorted_nodes)))

        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        num_sub_nodes = len(sorted_nodes)
        
        # Convert sentence-local mention spans to global doc word indices
        sents = item.get('sents', [])
        sent_offsets = [0]
        total = 0
        for s in sents:
            total += len(s)
            sent_offsets.append(total)

        def to_global_span(m):
            sid = int(m.get('sent_id', 0))
            start, end = int(m['pos'][0]), int(m['pos'][1])
            if sid < 0 or sid >= len(sent_offsets) - 1:
                return start, end
            base = sent_offsets[sid]
            return base + start, base + end

        word_to_tokens = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in word_to_tokens: word_to_tokens[word_idx] = []
                word_to_tokens[word_idx].append(token_idx)

        node_feats = []
        doc_context = llm_embeddings.mean(0)
        for entity_idx in sorted_nodes:
            entity_mentions = vertex_set[entity_idx]
            mention_embeds = []
            for m in entity_mentions:
                start, end = to_global_span(m)
                mention_token_indices = []
                for w_idx in range(start, end):
                    if w_idx in word_to_tokens: mention_token_indices.extend(word_to_tokens[w_idx])
                if mention_token_indices:
                    indices_tensor = torch.tensor(mention_token_indices).to(llm_embeddings.device)
                    mention_embeds.append(torch.index_select(llm_embeddings, 0, indices_tensor).mean(0))
            if mention_embeds:
                node_feats.append(torch.stack(mention_embeds).mean(0))
            else:
                # Mention can be truncated out of the token window; use doc context instead of all-zero vectors.
                node_feats.append(doc_context)
        
        node_feats = torch.stack(node_feats) if node_feats else torch.zeros((1, llm_embeddings.shape[-1]), dtype=llm_embeddings.dtype, device=llm_embeddings.device)

        u_list, v_list = [], []
        adj = torch.zeros((num_sub_nodes, num_sub_nodes))
        for i, u_node in enumerate(sorted_nodes):
            for j, v_node in enumerate(sorted_nodes):
                if i >= j: continue
                if u_node in doc_adj and v_node in doc_adj[u_node]:
                    u_list.extend([i, j])
                    v_list.extend([j, i])
                    adj[i, j] = adj[j, i] = 1.0
        
        if len(u_list) == 0:
            g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=num_sub_nodes)
        else:
            g = dgl.graph((torch.tensor(u_list), torch.tensor(v_list)), num_nodes=num_sub_nodes)
        
        # In hybrid mode, LLM may be on CUDA while DGL stays on CPU.
        # Always place graph + node features on graph builder device.
        graph_torch_device = torch.device(self.device)
        g = g.to(graph_torch_device)
        g = dgl.add_self_loop(g)
        g.ndata['h'] = node_feats.to(graph_torch_device)
        g.ndata['is_ht'] = torch.tensor(
            [1.0 if n in [h_id, t_id] else 0.0 for n in sorted_nodes],
            device=graph_torch_device,
        )
        return g, adj, node_to_idx[h_id], node_to_idx[t_id]'''

new_graph_builder = '''class DocREDGraphBuilder:
    """
    Construct k-hop entity pair subgraphs from document.
    
    Graph nodes: entities
    Graph edges: co-occurrence in same sentence with direction and type
    Subgraph: k-hop neighborhood around (h, t) pair
    """
    def __init__(self, device):
        self.device = str(device)

    def build_pair_subgraph(self, item, llm_embeddings, word_ids, h_id, t_id, k_hop=1):
        import collections
        vertex_set = item.get('vertex_set', item.get('vertexSet', []))
        num_entities = len(vertex_set)
        
        doc_adj = {}
        edge_meta = {}
        for i in range(num_entities):
            for j in range(num_entities):
                if i == j:
                    continue
                sents_i = {m['sent_id'] for m in vertex_set[i]}
                sents_j = {m['sent_id'] for m in vertex_set[j]}
                if sents_i.intersection(sents_j):
                    if i not in doc_adj: doc_adj[i] = set()
                    doc_adj[i].add(j)
                    dir_id = 1 if i < j else 0
                    edge_meta[(i, j)] = (0, dir_id)

        subgraph_nodes = {h_id, t_id}
        current_layer = {h_id, t_id}
        for _ in range(k_hop):
            next_layer = set()
            for node in current_layer:
                if node in doc_adj:
                    next_layer.update(doc_adj[node])
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
        
        sorted_nodes = sorted(list(subgraph_nodes))
        if len(sorted_nodes) > 15:
            sorted_nodes = [h_id, t_id] + [n for n in sorted_nodes if n not in [h_id, t_id]][:13]
            sorted_nodes = sorted(list(set(sorted_nodes)))

        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        num_sub_nodes = len(sorted_nodes)
        
        sents = item.get('sents', [])
        sent_offsets = [0]
        total = 0
        for s in sents:
            total += len(s)
            sent_offsets.append(total)

        def to_global_span(m):
            sid = int(m.get('sent_id', 0))
            start, end = int(m['pos'][0]), int(m['pos'][1])
            if sid < 0 or sid >= len(sent_offsets) - 1:
                return start, end
            base = sent_offsets[sid]
            return base + start, base + end

        word_to_tokens = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in word_to_tokens: word_to_tokens[word_idx] = []
                word_to_tokens[word_idx].append(token_idx)

        node_feats = []
        doc_context = llm_embeddings.mean(0)
        for entity_idx in sorted_nodes:
            entity_mentions = vertex_set[entity_idx]
            mention_embeds = []
            for m in entity_mentions:
                start, end = to_global_span(m)
                mention_token_indices = []
                for w_idx in range(start, end):
                    if w_idx in word_to_tokens: mention_token_indices.extend(word_to_tokens[w_idx])
                if mention_token_indices:
                    indices_tensor = torch.tensor(mention_token_indices).to(llm_embeddings.device)
                    mention_embeds.append(torch.index_select(llm_embeddings, 0, indices_tensor).mean(0))
            if mention_embeds:
                node_feats.append(torch.stack(mention_embeds).mean(0))
            else:
                node_feats.append(doc_context)
        
        node_feats = torch.stack(node_feats) if node_feats else torch.zeros((1, llm_embeddings.shape[-1]), dtype=llm_embeddings.dtype, device=llm_embeddings.device)

        u_list, v_list = [], []
        etype_list, edir_list = [], []
        adj_local = {i: set() for i in range(num_sub_nodes)}
        edge_index_map = {}
        
        for i, u_node in enumerate(sorted_nodes):
            for j, v_node in enumerate(sorted_nodes):
                if i == j:
                    continue
                if u_node in doc_adj and v_node in doc_adj[u_node]:
                    u_list.append(i)
                    v_list.append(j)
                    et, ed = edge_meta.get((u_node, v_node), (0, 1))
                    etype_list.append(et)
                    edir_list.append(ed)
                    adj_local[i].add(j)
                    edge_index_map[(i, j)] = len(u_list) - 1
        
        for i in range(num_sub_nodes):
            u_list.append(i)
            v_list.append(i)
            etype_list.append(1)
            edir_list.append(2)
        
        if len(u_list) == 0:
            g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=num_sub_nodes)
        else:
            g = dgl.graph((torch.tensor(u_list), torch.tensor(v_list)), num_nodes=num_sub_nodes)
        
        graph_torch_device = torch.device(self.device)
        g = g.to(graph_torch_device)
        g.ndata['h'] = node_feats.to(graph_torch_device)
        g.ndata['is_ht'] = torch.tensor(
            [1.0 if n in [h_id, t_id] else 0.0 for n in sorted_nodes],
            device=graph_torch_device,
        )
        g.edata['type'] = torch.tensor(etype_list, dtype=torch.long, device=graph_torch_device)
        g.edata['dir'] = torch.tensor(edir_list, dtype=torch.long, device=graph_torch_device)
        
        start_idx = node_to_idx[h_id]
        end_idx = node_to_idx[t_id]
        queue = collections.deque([(start_idx, [])])
        visited = {start_idx}
        path_edges = []
        while queue:
            node, path = queue.popleft()
            if node == end_idx:
                path_edges = path
                break
            for neighbor in adj_local.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(node, neighbor)]))
        
        path_types = []
        path_dirs = []
        for u, v in path_edges:
            idx = edge_index_map[(u, v)]
            path_types.append(int(etype_list[idx]))
            path_dirs.append(int(edir_list[idx]))
        
        return g, node_to_idx[h_id], node_to_idx[t_id], (path_types, path_dirs)'''

if old_graph_builder in content:
    content = content.replace(old_graph_builder, new_graph_builder)
    print("Replaced GraphBuilder")
else:
    print("WARNING: GraphBuilder block not found exactly")

# 2. Replace old model classes with import
old_models = '''# ==========================================
# 2. FGW UTILITIES (Entropic OT)
# ==========================================
class RelationPrototype(nn.Module):
    """
    Learnable relation prototypes for structural contrastive alignment.
    """
    def __init__(self, num_relations, dim):
        super().__init__()
        self.num_relations = num_relations
        self.proto = nn.Parameter(torch.randn(num_relations, dim) * 0.02)

    def get(self, rel_id):
        return self.proto[rel_id]

    def get_all(self):
        return self.proto


def gcompute_fgw_distance(g1_nodes, g1_adj, g2_nodes, g2_adj, alpha=0.5, reg=0.05):
    """
    Entropic Fused Gromov-Wasserstein Distance (differentiable via POT Sinkhorn).
    
    Math: min_T (1-\u03b1)\u27e8T,M\u27e9 + \u03b1\u00b7GW(A,B,T) - \u03b5\u00b7H(T)
    
    Args:
        g1_nodes, g2_nodes: node embeddings (N1, D), (N2, D)
        g1_adj, g2_adj: adjacency (N1, N1), (N2, N2)
        alpha: balance between feature (1-\u03b1) and structure (\u03b1)
        reg: entropy regularization \u03b5
    
    Returns:
        FGW distance (scalar tensor, differentiable)
    """
    try:
        # POT/FGW is numerically more stable in float32 than low-precision dtypes.
        g1_nodes = torch.nan_to_num(g1_nodes.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        g2_nodes = torch.nan_to_num(g2_nodes.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        g1_adj = torch.nan_to_num(g1_adj.float(), nan=0.0, posinf=1.0, neginf=0.0)
        g2_adj = torch.nan_to_num(g2_adj.float(), nan=0.0, posinf=1.0, neginf=0.0)

        if (not torch.isfinite(g1_nodes).all()) or (not torch.isfinite(g2_nodes).all()):
            return torch.tensor(1e-3, device=g1_nodes.device, dtype=torch.float32)

        # 1. Feature cost matrix M
        M = ot.dist(g1_nodes, g2_nodes, metric='sqeuclidean')
        M = M / (M.max() + 1e-9)  # normalize

        # 2. Uniform marginals
        n1, n2 = g1_nodes.shape[0], g2_nodes.shape[0]
        p1 = torch.ones(n1, device=g1_nodes.device, dtype=g1_nodes.dtype) / n1
        p2 = torch.ones(n2, device=g2_nodes.device, dtype=g2_nodes.dtype) / n2

        # 3. Entropic FGW (POT backend handles gradient)
        fgw_dist = ot.gromov.entropic_fused_gromov_wasserstein2(
            M, g1_adj, g2_adj, p1, p2,
            alpha=alpha,
            epsilon=reg,
            loss_fun='square_loss',
            symmetric=True,
            max_iter=300,
            tol=1e-6,
            verbose=False
        )

        # Ensure tensor output
        if not torch.is_tensor(fgw_dist):
            fgw_dist = torch.tensor(fgw_dist, device=g1_nodes.device, dtype=torch.float32)

        fgw_dist = torch.nan_to_num(fgw_dist, nan=1e-3, posinf=10.0, neginf=0.0)
        if not torch.isfinite(fgw_dist):
            return torch.tensor(1e-3, device=g1_nodes.device, dtype=torch.float32)
        
        return fgw_dist
    except Exception as e:
        # Fallback: return small penalty to avoid gradient issues
        return torch.tensor(1e-3, device=g1_nodes.device, dtype=torch.float32)


class SparseRouter(nn.Module):
    """
    Noisy Top-1 Router (Switch Transformer style).
    
    Math: p(e|x) = softmax(W\u00b7x + noise)
    Top-1: dispatch x to expert e* = argmax p(e|x)
    
    Noise prevents router collapse during training.
    """
    def __init__(self, in_dim, num_experts, noise_eps=1e-2):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_experts)
        self.noise_eps = noise_eps
        self.num_experts = num_experts

    def forward(self, x, training=True):
        """
        Args:
            x: input features (B, in_dim)
            training: add noise only during training
        Returns:
            logits: raw scores (B, num_experts)
            top_idx: selected expert indices (B,)
        """
        logits = self.linear(x)
        
        # Add Gaussian noise during training (exploration)
        if training:
            noise = torch.randn_like(logits) * self.noise_eps
            logits = logits + noise
        
        # Top-1 hard routing
        _, top_idx = logits.topk(1, dim=-1)
        return logits, top_idx.squeeze(-1)


# ==========================================
# 3. MOE-GRAPH MODEL (Core Architecture)
# ==========================================
class GraphTransformerLayer(nn.Module):
    """Transformer block with graph-constrained self-attention."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask):
        y, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + y)
        x = self.norm2(x + self.ffn(x))
        return x


class GraphExpert(nn.Module):
    """
    Graph Expert: Graph Attention Transformer for encoding entity pair subgraphs.

    Each expert learns to process a specific cluster of graph patterns,
    while attention is constrained by graph adjacency.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        del hidden_dim
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(out_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def _fallback(self, pair_repr_i, dtype, device):
        out_dim2 = self.out_dim * 2
        if pair_repr_i.shape[-1] >= out_dim2:
            return pair_repr_i[:out_dim2].to(device=device, dtype=dtype)
        pad = torch.zeros(out_dim2 - pair_repr_i.shape[-1], device=device, dtype=dtype)
        return torch.cat([pair_repr_i.to(device=device, dtype=dtype), pad], dim=-1)
        
    def forward(self, g, h, pair_repr):
        """
        Args:
            g: batched DGL graph
            h: node features (num_nodes, in_dim)
            pair_repr: context features (for potential skip connection)
        Returns:
            pair_embedding: (batch_size, out_dim*2) [h_final, t_final]
        """
        h = self.in_proj(h)
        graphs = dgl.unbatch(g)
        sizes = g.batch_num_nodes().tolist()
        out = []
        offset = 0

        for i, (sg, n) in enumerate(zip(graphs, sizes)):
            if n <= 0:
                out.append(self._fallback(pair_repr[i], h.dtype, pair_repr.device))
                continue

            x = h[offset:offset + n].unsqueeze(0)
            offset += n

            src, dst = sg.edges()
            attn_mask = torch.ones((n, n), device=x.device, dtype=torch.bool)
            if src.numel() > 0:
                attn_mask[src.long(), dst.long()] = False
            else:
                attn_mask.fill_(False)

            for layer in self.layers:
                x = layer(x, attn_mask)
            x = x.squeeze(0)

            is_ht = sg.ndata['is_ht'].bool()
            ht_nodes = x[is_ht]
            if ht_nodes.shape[0] >= 2:
                out.append(torch.cat([ht_nodes[0], ht_nodes[1]], dim=-1))
            elif ht_nodes.shape[0] == 1:
                out.append(torch.cat([ht_nodes[0], ht_nodes[0]], dim=-1))
            else:
                out.append(self._fallback(pair_repr[i], x.dtype, pair_repr.device))

        if len(out) != pair_repr.shape[0]:
            return torch.stack([
                self._fallback(pair_repr[i], h.dtype, pair_repr.device)
                for i in range(pair_repr.shape[0])
            ], dim=0)
        return torch.stack(out, dim=0).to(pair_repr.device)

class MoEGraphRE(nn.Module):
    """
    Mixture-of-Experts Graph Relation Extraction Model.
    
    Pipeline: LLM embeddings \u2192 Entity pairs \u2192 Graph construction \u2192 Sparse MoE \u2192 Relation classifier
    
    MoE learns: p(r|h,t,D) = \u03a3_e p(e|h,t,D) \u00b7 p(r|h,t,D,e)
             Router \u2248 posterior expert selection
             Expert \u2248 conditional graph encoder
    """
    def __init__(self, llm_model, num_relations, num_experts=4, expert_dim=128, noise_scale=1e-2, capacity_factor=1.25, graph_device='cpu'):
        super().__init__()
        self.llm = llm_model
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.noise_scale = noise_scale
        self.capacity_factor = capacity_factor
        self.graph_device = graph_device
        
        hidden_size = llm_model.config.hidden_size
        
        # Experts: specialized graph encoders
        self.experts = nn.ModuleList([
            GraphExpert(hidden_size, expert_dim, expert_dim) 
            for _ in range(num_experts)
        ])
        
        # Router: learns p(expert | entity_pair_context)
        self.router = SparseRouter(hidden_size * 2, num_experts, noise_eps=noise_scale)
        
        # Classifier: final relation predictor
        self.classifier = nn.Sequential(
            nn.Linear(expert_dim * 2, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim, num_relations)
        )
        
    def forward(self, subgraphs, pair_features):
        """
        Sparse MoE Forward Pass (Switch-style conditional computation).
        
        Args:
            subgraphs: batched DGL graph (B entity pair subgraphs)
            pair_features: (B, hidden_size*2) concatenated [h_embed, t_embed]
        
        Returns:
            logits: relation logits (B, num_relations)
            router_logits: routing scores (B, num_experts)
            top1_idx: selected expert per sample (B,)
            pair_emb: expert-composed pair representation (B, expert_dim*2)
        """
        # Keep MoE path in fp32 for stable autograd under DDP + quantized LLM outputs.
        dtype = torch.float32
        device = pair_features.device
        pair_features = pair_features.to(dtype)
        batch_size = pair_features.shape[0]
        
        # Ensure correct dtype
        self.experts.to(dtype)
        self.router.to(dtype)
        self.classifier.to(dtype)
        
        # Step 1: Router - Select top-1 expert per sample
        router_logits, top1_idx = self.router(pair_features, training=self.training)

        # Step 2: Compute gating weights (for gradient flow)
        probs = F.softmax(router_logits, dim=-1)
        top1_prob = probs.gather(1, top1_idx.unsqueeze(-1)).squeeze(-1)

        # Step 3: Dispatch & Execute experts (capacity-limited)
        graph_list = dgl.unbatch(subgraphs)
        # DGL may be CPU-only even when the rest of the model runs on CUDA.
        graph_list = [g.to(self.graph_device) for g in graph_list]
        # Gradient-safe default path: even if a sample is dropped by capacity or hits
        # expert fallback, the loss still has a valid autograd path to pair_features.
        out_dim = self.expert_dim * 2
        if pair_features.shape[-1] >= out_dim:
            pair_emb = pair_features[:, :out_dim].clone()
        else:
            pad = torch.zeros(
                (batch_size, out_dim - pair_features.shape[-1]),
                device=device,
                dtype=dtype,
            )
            pair_emb = torch.cat([pair_features, pad], dim=-1)
        
        # Capacity: max tokens per expert (prevents overload)
        capacity = max(1, int(math.ceil(self.capacity_factor * batch_size / self.num_experts)))
        
        for e_idx in range(self.num_experts):
            # Find samples routed to expert e_idx
            mask = (top1_idx == e_idx)
            selected_indices = mask.nonzero(as_tuple=True)[0]
            
            if selected_indices.numel() == 0:
                continue  # Skip unused expert

            # Apply capacity limit (drop lowest priority if overflow)
            if selected_indices.numel() > capacity:
                sel_probs = top1_prob[selected_indices]
                _, top_k = torch.topk(sel_probs, k=capacity)
                selected_indices = selected_indices[top_k]
                
            # Prepare sub-batch for this expert
            e_pair_feats = pair_features[selected_indices]
            e_graphs = dgl.batch([graph_list[i] for i in selected_indices.tolist()])
            
            # Ensure graph is on selected graph device (DGL batch may not preserve device)
            e_graphs = e_graphs.to(self.graph_device)
            
            # Execute ONLY this expert (conditional computation)
            # Experts consume graph features on graph device; output is moved back to pair device.
            e_out = self.experts[e_idx](
                e_graphs,
                e_graphs.ndata['h'].to(device=self.graph_device, dtype=dtype),
                e_pair_feats.to(device=self.graph_device, dtype=dtype),
            ).to(device)
            
            # Combine back (weighted by router confidence)
            pair_emb[selected_indices] = e_out * top1_prob[selected_indices].unsqueeze(-1)
            
        # Step 4: Final classification
        logits = self.classifier(pair_emb)
        return logits, router_logits, top1_idx, pair_emb'''

new_models = '''# ==========================================
# 2. SP-GAT MODEL (Core Architecture)
# ==========================================
from sp_gat import SPGATRE, sparsity_loss'''

if old_models in content:
    content = content.replace(old_models, new_models)
    print("Replaced model classes")
else:
    print("WARNING: Model block not found exactly")

# 3. Remove switch_load_balance_loss
old_switch = '''def switch_load_balance_loss(router_logits, top_idx, num_experts):
    """
    Switch Load Balancing Loss (variance-based).
    
    Ensures experts are utilized uniformly:
    - Importance: router's soft assignment (what it wants to use)
    - Load: actual hard assignment (what gets used)
    
    Loss = Var(importance) + Var(load)
    
    This prevents router collapse where all samples go to 1-2 experts.
    """
    # Soft assignment (differentiable)
    probs = F.softmax(router_logits.float(), dim=-1)  # (B, E)
    importance = probs.sum(0)  # (E,)
    
    # Hard assignment (actual dispatch)
    load = torch.bincount(top_idx, minlength=num_experts).float().to(router_logits.device)
    
    # Normalize and compute variance
    importance_norm = importance / (importance.sum() + 1e-9)
    load_norm = load / (load.sum() + 1e-9)
    
    importance_loss = torch.var(importance_norm)
    load_loss = torch.var(load_norm)
    
    out = importance_loss + load_loss
    if not torch.isfinite(out):
        return torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)
    return out.to(router_logits.dtype)'''

if old_switch in content:
    content = content.replace(old_switch, '')
    print("Removed switch_load_balance_loss")
else:
    print("WARNING: switch_load_balance_loss not found")

# 4. Update evaluate_model model call
old_eval = '''                for h_idx, t_idx in batch_pairs:
                    g, _, _, _ = graph_builder.build_pair_subgraph(item, doc_embeds, w_ids, h_idx, t_idx)
                    subgraphs.append(g)
                    ht = g.ndata['h'][g.ndata['is_ht'].bool()]
                    h_f, t_f = (ht[0], ht[1]) if ht.shape[0] >= 2 else (ht[0], ht[0])
                    pair_features.append(torch.cat([h_f, t_f]))

                if use_pair_markers and marker_feats is not None:
                    for idx in range(len(pair_features)):
                        if marker_mask[idx]:
                            marker_vec = marker_feats[idx].to(pair_features[idx].device)
                            pair_features[idx] = 0.5 * pair_features[idx] + 0.5 * marker_vec
                
                logits, _, _, _ = core_model(dgl.batch(subgraphs), torch.stack(pair_features).to(device))'''

new_eval = '''                path_infos = []
                for h_idx, t_idx in batch_pairs:
                    g, _, _, path_info = graph_builder.build_pair_subgraph(item, doc_embeds, w_ids, h_idx, t_idx)
                    subgraphs.append(g)
                    path_infos.append(path_info)
                    ht = g.ndata['h'][g.ndata['is_ht'].bool()]
                    h_f, t_f = (ht[0], ht[1]) if ht.shape[0] >= 2 else (ht[0], ht[0])
                    pair_features.append(torch.cat([h_f, t_f]))

                if use_pair_markers and marker_feats is not None:
                    for idx in range(len(pair_features)):
                        if marker_mask[idx]:
                            marker_vec = marker_feats[idx].to(pair_features[idx].device)
                            pair_features[idx] = 0.5 * pair_features[idx] + 0.5 * marker_vec
                
                logits = core_model(dgl.batch(subgraphs), torch.stack(pair_features).to(device), path_infos)[0]'''

if old_eval in content:
    content = content.replace(old_eval, new_eval)
    print("Replaced evaluate_model call")
else:
    print("WARNING: evaluate_model block not found")

# 5. Update main() model init
old_init = '''    # MoE Graph RE Model with Sparse Experts
    model = MoEGraphRE(
        lora_model,
        num_relations,
        num_experts=args.num_experts,
        capacity_factor=args.capacity_factor,
        graph_device=graph_device,
    ).to(DEVICE)

    if graph_device == 'cpu':
        model.experts = model.experts.to('cpu')
    
    # Learnable relation prototypes for structural contrastive alignment.
    prototype_dim = model.expert_dim * 2
    prototypes = RelationPrototype(num_relations, prototype_dim).to(DEVICE)'''

new_init = '''    # SP-GAT Graph RE Model
    model = SPGATRE(
        lora_model,
        num_relations,
        num_layers=2,
        num_heads=4,
        d_head=128,
        d_e=32,
        d_d=16,
        d_p=64,
        num_edge_types=10,
        dropout=0.1,
        graph_device=graph_device,
    ).to(DEVICE)'''

if old_init in content:
    content = content.replace(old_init, new_init)
    print("Replaced model init")
else:
    print("WARNING: model init block not found")

# 6. Remove prototype DDP
old_proto_ddp = '''    if distributed:
        prototypes = DDP(
            prototypes,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )'''

if old_proto_ddp in content:
    content = content.replace(old_proto_ddp, '')
    print("Removed prototype DDP")
else:
    print("WARNING: prototype DDP not found")

# 7. Update phase config
old_phase = '''        if phase_name == 'lora':
            # LoRA phase: train adapter weights only, freeze GNN + prototypes.
            lora_trainable = 0
            for n, p in core.llm.named_parameters():
                is_lora = ('lora_' in n)
                p.requires_grad = is_lora
                if is_lora:
                    lora_trainable += 1

            _set_module_requires_grad(core.experts, False)
            _set_module_requires_grad(core.router, False)
            _set_module_requires_grad(core.classifier, False)
            _set_module_requires_grad(proto_core, False)
            phase_lr = args.lr_lora
            if rank == 0:
                print(f"[PHASE] LoRA phase active | trainable_lora_tensors={lora_trainable} | lr={phase_lr}")
        else:
            # GNN phase: freeze LLM and train graph experts/router/classifier + prototypes.
            _set_module_requires_grad(core.llm, False)
            _set_module_requires_grad(core.experts, True)
            _set_module_requires_grad(core.router, True)
            _set_module_requires_grad(core.classifier, True)
            _set_module_requires_grad(proto_core, True)
            phase_lr = args.lr_gnn
            if rank == 0:
                print(f"[PHASE] GNN phase active | lr={phase_lr}")'''

new_phase = '''        if phase_name == 'lora':
            # LoRA phase: train adapter weights only, freeze GNN.
            lora_trainable = 0
            for n, p in core.llm.named_parameters():
                is_lora = ('lora_' in n)
                p.requires_grad = is_lora
                if is_lora:
                    lora_trainable += 1

            _set_module_requires_grad(core.gat_layers, False)
            _set_module_requires_grad(core.path_encoder, False)
            _set_module_requires_grad(core.classifier, False)
            phase_lr = args.lr_lora
            if rank == 0:
                print(f"[PHASE] LoRA phase active | trainable_lora_tensors={lora_trainable} | lr={phase_lr}")
        else:
            # GNN phase: freeze LLM and train SP-GAT + classifier.
            _set_module_requires_grad(core.llm, False)
            _set_module_requires_grad(core.gat_layers, True)
            _set_module_requires_grad(core.path_encoder, True)
            _set_module_requires_grad(core.classifier, True)
            phase_lr = args.lr_gnn
            if rank == 0:
                print(f"[PHASE] GNN phase active | lr={phase_lr}")'''

if old_phase in content:
    content = content.replace(old_phase, new_phase)
    print("Replaced phase config")
else:
    print("WARNING: phase config not found")

# 8. Update optimizer params
old_opt = "        trainable = [p for p in list(model.parameters()) + list(prototypes.parameters()) if p.requires_grad]"
new_opt = "        trainable = [p for p in model.parameters() if p.requires_grad]"
if old_opt in content:
    content = content.replace(old_opt, new_opt)
    print("Replaced optimizer params")
else:
    print("WARNING: optimizer params not found")

# 9. Update compact checkpoint
old_compact = '''    def _build_compact_checkpoint_payload(epoch, best_val_f1):
        core_model = _unwrap_model(model)
        return {
            "epoch": int(epoch),
            "best_val_f1": float(best_val_f1),
            "experts_state_dict": core_model.experts.state_dict(),
            "router_state_dict": core_model.router.state_dict(),
            "classifier_state_dict": core_model.classifier.state_dict(),
            "prototype_state_dict": _unwrap_model(prototypes).state_dict(),
            "args": vars(args),
            "run_name": str(run_name),
            "checkpoint_format": "compact_no_llm_backbone",
        }'''

new_compact = '''    def _build_compact_checkpoint_payload(epoch, best_val_f1):
        core_model = _unwrap_model(model)
        return {
            "epoch": int(epoch),
            "best_val_f1": float(best_val_f1),
            "model_state_dict": core_model.state_dict(),
            "args": vars(args),
            "run_name": str(run_name),
            "checkpoint_format": "spgat_full",
        }'''

if old_compact in content:
    content = content.replace(old_compact, new_compact)
    print("Replaced compact checkpoint")
else:
    print("WARNING: compact checkpoint not found")

# 10. Update best checkpoint save
old_best_ckpt = '''                ckpt_payload = {
                    "epoch": int(epoch),
                    "best_val_f1": float(best_f1),
                    "model_state_dict": _unwrap_model(model).state_dict(),
                    "prototype_state_dict": _unwrap_model(prototypes).state_dict(),
                    "args": vars(args),
                    "run_name": str(run_name),
                }'''

new_best_ckpt = '''                ckpt_payload = {
                    "epoch": int(epoch),
                    "best_val_f1": float(best_f1),
                    "model_state_dict": _unwrap_model(model).state_dict(),
                    "args": vars(args),
                    "run_name": str(run_name),
                }'''

if old_best_ckpt in content:
    content = content.replace(old_best_ckpt, new_best_ckpt)
    print("Replaced best checkpoint")
else:
    print("WARNING: best checkpoint not found")

# 11. Update checkpoint load
old_load_ckpt = '''    def _load_checkpoint_into_current_model(checkpoint_path):
        if not checkpoint_path or (not os.path.exists(checkpoint_path)):
            return False
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        core_model = _unwrap_model(model)
        proto_model = _unwrap_model(prototypes)

        if "model_state_dict" in ckpt:
            core_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            if "experts_state_dict" in ckpt:
                core_model.experts.load_state_dict(ckpt["experts_state_dict"], strict=False)
            if "router_state_dict" in ckpt:
                core_model.router.load_state_dict(ckpt["router_state_dict"], strict=False)
            if "classifier_state_dict" in ckpt:
                core_model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
        if "prototype_state_dict" in ckpt:
            proto_model.load_state_dict(ckpt["prototype_state_dict"], strict=False)
        return True'''

new_load_ckpt = '''    def _load_checkpoint_into_current_model(checkpoint_path):
        if not checkpoint_path or (not os.path.exists(checkpoint_path)):
            return False
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        core_model = _unwrap_model(model)
        if "model_state_dict" in ckpt:
            core_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return True'''

if old_load_ckpt in content:
    content = content.replace(old_load_ckpt, new_load_ckpt)
    print("Replaced checkpoint load")
else:
    print("WARNING: checkpoint load not found")

# 12. Update training loop
old_train = '''            # Layer 3: Pair subgraph construction (k-hop neighborhoods)
            # Doc-level loader stays at 1; this is the true compute batch on pair subgraphs.
            pair_batch_size = max(1, args.batch_size)
            for p_s in range(0, len(train_p), pair_batch_size):
                p_e = min(p_s + pair_batch_size, len(train_p))
                b_p = train_p[p_s:p_e]
                b_l = build_multi_hot_targets(b_p, pair_to_rels, num_relations, DEVICE)
                sgs, adjs, p_f = [], [], []
                marker_feats = None
                marker_mask = None
                if use_pair_markers:
                    pair_items = build_pair_batch_items(item, b_p)
                    pair_enc = collate_fn(pair_items, tokenizer, max_length=args.max_seq_length)['encodings'].to(DEVICE)
                    pair_out = core_model.llm(**pair_enc, output_hidden_states=True)
                    marker_feats, marker_mask = extract_marker_pair_features(pair_out, pair_enc, e1_id, e2_id)
                    # Free memory immediately after marker extraction
                    del pair_out, pair_enc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                for h, t in b_p:
                    g, adj, _, _ = graph_builder.build_pair_subgraph(item, doc_emb, w_ids, h, t)
                    sgs.append(g)
                    adjs.append(adj)
                    ht = g.ndata['h'][g.ndata['is_ht'].bool()]
                    h_f, t_f = (ht[0], ht[1]) if ht.shape[0]>=2 else (ht[0], ht[0])
                    p_f.append(torch.cat([h_f, t_f]))

                if use_pair_markers and marker_feats is not None:
                    for idx in range(len(p_f)):
                        if marker_mask[idx]:
                            marker_vec = marker_feats[idx].to(p_f[idx].device)
                            p_f[idx] = 0.5 * p_f[idx] + 0.5 * marker_vec

                logits, router_logits, top_idx, pair_repr = model(dgl.batch(sgs), torch.stack(p_f).to(DEVICE))

                # Loss 1: Focal multi-label loss for long-tail relations.
                cls_loss = focal_loss_with_logits(
                    logits,
                    b_l.float(),
                    gamma=args.focal_gamma,
                    alpha=args.focal_alpha,
                    reduction="mean",
                )

                # Loss 2: Switch Load Balancing (expert utilization)
                switch_loss = switch_load_balance_loss(router_logits, top_idx, core_model.num_experts)

                # Loss 3: Structural Contrastive Learning (InfoNCE) with relation prototypes
                scl_loss = structural_contrastive_loss(
                    pair_repr,
                    b_l,
                    _unwrap_model(prototypes),
                    temperature=args.scl_temp,
                )

                # Total Loss = L_CE + \u03bb_moe * L_balance + \u03bb_scl * L_scl
                batch_loss = cls_loss + args.lambda_moe * switch_loss + args.lambda_scl * scl_loss'''

new_train = '''            # Layer 3: Pair subgraph construction (k-hop neighborhoods)
            pair_batch_size = max(1, args.batch_size)
            for p_s in range(0, len(train_p), pair_batch_size):
                p_e = min(p_s + pair_batch_size, len(train_p))
                b_p = train_p[p_s:p_e]
                b_l = build_multi_hot_targets(b_p, pair_to_rels, num_relations, DEVICE)
                sgs, path_infos, p_f = [], [], []
                marker_feats = None
                marker_mask = None
                if use_pair_markers:
                    pair_items = build_pair_batch_items(item, b_p)
                    pair_enc = collate_fn(pair_items, tokenizer, max_length=args.max_seq_length)['encodings'].to(DEVICE)
                    pair_out = core_model.llm(**pair_enc, output_hidden_states=True)
                    marker_feats, marker_mask = extract_marker_pair_features(pair_out, pair_enc, e1_id, e2_id)
                    del pair_out, pair_enc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                for h, t in b_p:
                    g, _, _, path_info = graph_builder.build_pair_subgraph(item, doc_emb, w_ids, h, t)
                    sgs.append(g)
                    path_infos.append(path_info)
                    ht = g.ndata['h'][g.ndata['is_ht'].bool()]
                    h_f, t_f = (ht[0], ht[1]) if ht.shape[0]>=2 else (ht[0], ht[0])
                    p_f.append(torch.cat([h_f, t_f]))

                if use_pair_markers and marker_feats is not None:
                    for idx in range(len(p_f)):
                        if marker_mask[idx]:
                            marker_vec = marker_feats[idx].to(p_f[idx].device)
                            p_f[idx] = 0.5 * p_f[idx] + 0.5 * marker_vec

                logits, pi = model(dgl.batch(sgs), torch.stack(p_f).to(DEVICE), path_infos)

                # Loss 1: Focal multi-label loss for long-tail relations.
                cls_loss = focal_loss_with_logits(
                    logits,
                    b_l.float(),
                    gamma=args.focal_gamma,
                    alpha=args.focal_alpha,
                    reduction="mean",
                )

                # Loss 2: Sparsity regularization (Stochastic Path Pruning)
                sparsity = sparsity_loss(pi, p0=0.1)

                # Total Loss = L_CE + \u03b2 * L_sparsity
                batch_loss = cls_loss + 0.01 * sparsity'''

if old_train in content:
    content = content.replace(old_train, new_train)
    print("Replaced training loop")
else:
    print("WARNING: training loop not found")

# 13. Update memory cleanup
old_cleanup = "                del batch_loss, cls_loss, switch_loss, scl_loss, logits, router_logits, pair_repr, sgs, adjs"
new_cleanup = "                del batch_loss, cls_loss, sparsity, logits, pi, sgs"
if old_cleanup in content:
    content = content.replace(old_cleanup, new_cleanup)
    print("Replaced cleanup")
else:
    print("WARNING: cleanup not found")

# 14. Update argparser
old_args = '''    # MoE hyperparameters
    parser.add_argument('--num-experts', type=int, default=3, help='Number of graph experts')
    parser.add_argument('--capacity-factor', type=float, default=1.25, help='Expert capacity multiplier')
    parser.add_argument('--lambda-moe', type=float, default=0.1, help='Switch load balance loss weight')
    
    # Structural contrastive alignment hyperparameters
    parser.add_argument('--lambda-scl', type=float, default=0.05, help='Structural contrastive loss weight')
    parser.add_argument('--scl-temp', type=float, default=0.1, help='InfoNCE temperature for SCL')'''

new_args = '''    # SP-GAT hyperparameters
    parser.add_argument('--beta-sparsity', type=float, default=0.01, help='Sparsity loss weight')'''

if old_args in content:
    content = content.replace(old_args, new_args)
    print("Replaced argparser")
else:
    print("WARNING: argparser not found")

# 15. Update pretrained loading text
old_pre = '        print(f"[PRETRAIN] Loaded pretrained GNN checkpoint from: {resolved_path}")'
new_pre = '        print(f"[PRETRAIN] Loaded pretrained SP-GAT checkpoint from: {resolved_path}")'
if old_pre in content:
    content = content.replace(old_pre, new_pre)
    print("Replaced pretrained text")
else:
    print("WARNING: pretrained text not found")

# 16. Update _load_checkpoint_into_modules
old_load_mod = '''    def _load_checkpoint_into_modules(checkpoint_path, strict=False):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        loaded_any = False

        def _report(prefix, load_result):
            if rank != 0:
                return
            missing = []
            unexpected = []
            if hasattr(load_result, "missing_keys"):
                missing = list(getattr(load_result, "missing_keys", []))
                unexpected = list(getattr(load_result, "unexpected_keys", []))
            elif isinstance(load_result, (tuple, list)) and len(load_result) == 2:
                missing = list(load_result[0])
                unexpected = list(load_result[1])
            if missing or unexpected:
                print(
                    f"[PRETRAIN] {prefix} load report: missing={len(missing)} unexpected={len(unexpected)}"
                )

        if "model_state_dict" in ckpt:
            result = model.load_state_dict(ckpt["model_state_dict"], strict=bool(strict))
            _report("full model", result)
            loaded_any = True
        else:
            if "experts_state_dict" in ckpt:
                result = model.experts.load_state_dict(ckpt["experts_state_dict"], strict=bool(strict))
                _report("experts", result)
                loaded_any = True
            if "router_state_dict" in ckpt:
                result = model.router.load_state_dict(ckpt["router_state_dict"], strict=bool(strict))
                _report("router", result)
                loaded_any = True
            if "classifier_state_dict" in ckpt:
                result = model.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=bool(strict))
                _report("classifier", result)
                loaded_any = True

        if "prototype_state_dict" in ckpt:
            result = prototypes.load_state_dict(ckpt["prototype_state_dict"], strict=bool(strict))
            _report("prototypes", result)
            loaded_any = True

        return loaded_any'''

new_load_mod = '''    def _load_checkpoint_into_modules(checkpoint_path, strict=False):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        loaded_any = False

        def _report(prefix, load_result):
            if rank != 0:
                return
            missing = []
            unexpected = []
            if hasattr(load_result, "missing_keys"):
                missing = list(getattr(load_result, "missing_keys", []))
                unexpected = list(getattr(load_result, "unexpected_keys", []))
            elif isinstance(load_result, (tuple, list)) and len(load_result) == 2:
                missing = list(load_result[0])
                unexpected = list(load_result[1])
            if missing or unexpected:
                print(
                    f"[PRETRAIN] {prefix} load report: missing={len(missing)} unexpected={len(unexpected)}"
                )

        if "model_state_dict" in ckpt:
            result = model.load_state_dict(ckpt["model_state_dict"], strict=bool(strict))
            _report("full model", result)
            loaded_any = True

        return loaded_any'''

if old_load_mod in content:
    content = content.replace(old_load_mod, new_load_mod)
    print("Replaced load modules")
else:
    print("WARNING: load modules not found")

with open('/workspace/moe.py', 'w') as f:
    f.write(content)

if content != original:
    print("moe.py updated successfully")
else:
    print("No changes made")
