"""
SP-GAT: Stochastic Path-enhanced Graph Attention Transformer
Upgraded: 4-layer relational GAT + prenorm + residual + stochastic depth
+ Pair-State Initialization + Triangle Reasoning + Recycling Refinement
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class SDPBiLSTM(nn.Module):
    """Shortest Dependency Path BiLSTM encoder."""
    def __init__(self, num_edge_types, d_e=32, d_d=16, d_hidden=32):
        super().__init__()
        self.d_p = d_hidden * 2
        self.type_emb = nn.Embedding(num_edge_types, d_e)
        self.dir_emb = nn.Embedding(3, d_d)
        self.bilstm = nn.LSTM(d_e + d_d, d_hidden, batch_first=True, bidirectional=True)

    def forward(self, edge_types, edge_dirs):
        if edge_types.numel() == 0:
            return torch.zeros(self.d_p, device=edge_types.device, dtype=torch.float32)
        x = torch.cat([self.type_emb(edge_types), self.dir_emb(edge_dirs)], dim=-1)
        x = x.unsqueeze(0)
        _, (h_n, _) = self.bilstm(x)
        return h_n.view(-1)


class PairStateReasoning(nn.Module):
    """Pair-state initialization + sparse triangle reasoning + recycling refinement."""
    def __init__(self, hidden_dim, dropout=0.1, num_iters=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        # P_ij = MLP([h_i, h_j, h_i*h_j, h_i-h_j])
        self.pair_init = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Triangle message: f(P_ik, P_kj) approximated as MLP([P, m_k])
        self.tri_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Recycling gate
        self.recycle_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_head, h_tail, mediators=None):
        inter = h_head * h_tail
        diff = h_head - h_tail
        P = self.pair_init(torch.cat([h_head, h_tail, inter, diff], dim=-1))
        for _ in range(self.num_iters):
            if mediators is not None and mediators.shape[1] > 0:
                P_exp = P.unsqueeze(1).expand(-1, mediators.shape[1], -1)
                concat = torch.cat([P_exp, mediators], dim=-1)
                msg = self.tri_mlp(concat.view(-1, concat.shape[-1])).view_as(P_exp)
                tri_msg = msg.mean(dim=1)
                P = P + tri_msg
            gate = self.recycle_gate(torch.cat([P, h_head], dim=-1))
            P = gate * P + (1 - gate) * h_tail
        return P


class SPGATLayer(nn.Module):
    """Directional edge-aware GAT with Stochastic Path Pruning."""
    def __init__(self, in_dim, out_dim, num_heads, edge_dim, path_dim,
                 num_edge_types=10, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.W_q = nn.Linear(in_dim, num_heads * out_dim)
        self.W_k = nn.Linear(in_dim, num_heads * out_dim)
        self.W_v = nn.Linear(in_dim, num_heads * out_dim)
        self.W_p = nn.Linear(path_dim, num_heads * out_dim)

        self.E_type = nn.Embedding(num_edge_types, edge_dim // 2)
        self.d_dir = nn.Embedding(3, edge_dim // 2)

        feat_dim = 2 * out_dim + edge_dim + out_dim
        self.attn_vec = nn.Parameter(torch.ones(num_heads, feat_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, edge_types, edge_dirs, p_ht):
        q = self.W_q(x).view(-1, self.num_heads, self.out_dim)
        k = self.W_k(x).view(-1, self.num_heads, self.out_dim)
        v = self.W_v(x).view(-1, self.num_heads, self.out_dim)

        e_feat = torch.cat([self.E_type(edge_types),
                            self.d_dir(edge_dirs)], dim=-1)

        batch_sizes = g.batch_num_nodes().tolist()
        node_p_ht = []
        for i, n in enumerate(batch_sizes):
            node_p_ht.append(p_ht[i].unsqueeze(0).expand(n, -1))
        node_p_ht = torch.cat(node_p_ht, dim=0)
        p_proj = self.W_p(node_p_ht).view(-1, self.num_heads, self.out_dim)

        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v
        g.ndata['p_proj'] = p_proj
        g.edata['e_feat'] = e_feat

        def edge_attn(edges):
            q_k = torch.cat([edges.dst['q'], edges.src['k']], dim=-1)
            e = edges.data['e_feat'].unsqueeze(1).expand(-1, self.num_heads, -1)
            p = edges.src['p_proj']
            feat = torch.cat([q_k, e, p], dim=-1)
            e_score = torch.einsum('ekf,kf->ek', feat, self.attn_vec)
            return {'e_score': F.leaky_relu(e_score, negative_slope=0.2)}

        g.apply_edges(edge_attn)

        e_score = g.edata['e_score']
        pi = torch.sigmoid(e_score)
        g.edata['pi'] = pi

        if self.training:
            tau = 0.5
            u = torch.rand_like(pi)
            gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            logit = torch.log(pi + 1e-10) - torch.log(1 - pi + 1e-10)
            s_raw = torch.sigmoid((logit + gumbel) / tau)
        else:
            s_raw = (pi > 0.5).float()

        src, dst = g.edges()
        is_self_loop = (src == dst)
        is_sl = is_self_loop.unsqueeze(-1).expand_as(s_raw)
        s = torch.where(is_sl, torch.ones_like(s_raw), s_raw)
        g.edata['s'] = s

        alpha = edge_softmax(g, e_score.contiguous())
        g.edata['alpha'] = alpha
        alpha = self.dropout(alpha)

        g.edata['alpha_s'] = (alpha * s).unsqueeze(-1)
        g.update_all(fn.u_mul_e('v', 'alpha_s', 'm'),
                     fn.sum('m', 'h_out'))

        h_out = g.ndata['h_out'].view(-1, self.num_heads * self.out_dim)
        return h_out, pi


class SPGATRE(nn.Module):
    """Full SP-GAT Relation Extraction model with Pair-State Reasoning."""
    def __init__(self, llm_model, num_relations, num_layers=4, num_heads=4,
                 d_head=128, d_e=32, d_d=16, d_p=64, num_edge_types=10,
                 dropout=0.1, graph_device='cpu', stochastic_depth_rate=0.1):
        super().__init__()
        self.llm = llm_model
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.graph_device = graph_device

        hidden_size = llm_model.config.hidden_size

        self.path_encoder = SDPBiLSTM(num_edge_types, d_e=d_e, d_d=d_d,
                                      d_hidden=d_p // 2)

        self.gat_layers = nn.ModuleList([
            SPGATLayer(
                in_dim=hidden_size if i == 0 else num_heads * d_head,
                out_dim=d_head,
                num_heads=num_heads,
                edge_dim=d_e + d_d,
                path_dim=self.path_encoder.d_p,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size if i == 0 else num_heads * d_head)
            for i in range(num_layers)
        ])

        self.stochastic_depth_rate = stochastic_depth_rate

        self.pair_reasoning = PairStateReasoning(
            hidden_dim=num_heads * d_head,
            dropout=dropout,
            num_iters=2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_heads * d_head + self.path_encoder.d_p,
                      d_head * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_head * 2, num_relations)
        )
        # ATLOP-style adaptive threshold bias (B2)
        self.threshold_bias = nn.Parameter(torch.zeros(num_relations))

    def forward(self, g, pair_features, path_infos):
        dtype = torch.float32
        device = pair_features.device

        # Move graph and all graph tensors to the target device (GPU when DGL CUDA is available)
        g = g.to(self.graph_device)
        x = g.ndata['h'].to(dtype=dtype, device=self.graph_device)
        edge_types = g.edata['type'].to(device=self.graph_device)
        edge_dirs = g.edata['dir'].to(device=self.graph_device)

        p_hts = []
        for types, dirs in path_infos:
            types_t = torch.tensor(types, dtype=torch.long, device=self.graph_device) if types else torch.empty(0, dtype=torch.long, device=self.graph_device)
            dirs_t = torch.tensor(dirs, dtype=torch.long, device=self.graph_device) if dirs else torch.empty(0, dtype=torch.long, device=self.graph_device)
            if types_t.numel() > 0:
                p_ht = self.path_encoder(types_t, dirs_t)
            else:
                p_ht = torch.zeros(self.path_encoder.d_p, device=self.graph_device, dtype=dtype)
            p_hts.append(p_ht)
        p_ht = torch.stack(p_hts).to(dtype=dtype, device=self.graph_device)

        saved_pi = None
        for i, layer in enumerate(self.gat_layers):
            # Prenorm + residual (only when dims match) + stochastic depth
            x_norm = self.layer_norms[i](x)
            h_out, pi = layer(g, x_norm, edge_types, edge_dirs, p_ht)
            if self.training and self.stochastic_depth_rate > 0.0:
                # Drop path with increasing survival probability for deeper layers
                survival_prob = 1.0 - (self.stochastic_depth_rate * (i / max(1, self.num_layers - 1)))
                if torch.rand(1).item() > survival_prob:
                    h_out = torch.zeros_like(h_out)
            if x.shape == h_out.shape:
                x = x + h_out
            else:
                x = h_out
            if i == 0:
                saved_pi = pi
            if i < self.num_layers - 1:
                x = F.relu(x)

        graphs = dgl.unbatch(g)
        out = []
        offset = 0
        for i, sg in enumerate(graphs):
            n = sg.num_nodes()
            sg_x = x[offset:offset + n]
            offset += n

            is_ht = sg.ndata['is_ht'].bool()
            ht_nodes = sg_x[is_ht]
            mediator_nodes = sg_x[~is_ht]

            if ht_nodes.shape[0] >= 2:
                h_head = ht_nodes[0]
                h_tail = ht_nodes[1]
            elif ht_nodes.shape[0] == 1:
                h_head = ht_nodes[0]
                h_tail = ht_nodes[0]
            else:
                pf = pair_features[i]
                target = self.gat_layers[-1].num_heads * self.gat_layers[-1].out_dim
                if pf.shape[-1] >= target:
                    h_head = h_tail = pf[:target]
                else:
                    pad = torch.zeros(target - pf.shape[-1],
                                      device=pf.device, dtype=pf.dtype)
                    h_head = h_tail = torch.cat([pf, pad], dim=-1)

            # Sparse mediator routing: use up to 3 mediators
            if mediator_nodes.shape[0] > 0:
                k = min(3, mediator_nodes.shape[0])
                mediators = mediator_nodes[:k].unsqueeze(0)
            else:
                mediators = torch.zeros((1, 0, h_head.shape[-1]), device=h_head.device, dtype=h_head.dtype)

            # Pair-state reasoning
            P = self.pair_reasoning(h_head.unsqueeze(0), h_tail.unsqueeze(0), mediators=mediators)
            P = P.squeeze(0)

            p_enc = p_ht[i]
            z = torch.cat([P, p_enc], dim=-1)
            out.append(z)

        out = torch.stack(out, dim=0).to(device)
        logits = self.classifier(out) - self.threshold_bias
        if saved_pi is not None:
            saved_pi = saved_pi.to(device)
        return logits, saved_pi


def sparsity_loss(pi, p0=0.1):
    if pi is None:
        return torch.tensor(0.0)
    pi_mean = pi.mean(dim=-1)
    kl = pi_mean * torch.log((pi_mean + 1e-10) / p0) + \
         (1 - pi_mean) * torch.log((1 - pi_mean + 1e-10) / (1 - p0))
    return kl.mean()
