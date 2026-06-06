"""
========================================================================================================
DOCUMENT-LEVEL RELATION EXTRACTION WITH DS CLEANING + GRAPH ATTENTION TRANSFORMER MOE (MULTI-LABEL)
========================================================================================================

CURRENT PIPELINE (implemented):
    Stage 1 (DS cleaning)
    Raw DS
      └─► Type constraints
          └─► MIL top-k denoising per (h,t)
              └─► train_distant_clean.json

    Stage 2 (Supervised DocRE)
    DocRED (+ optional cleaned DS mix)
      └─► LLM encoder (LoRA phase + GNN phase schedule)
          └─► Candidate generation + prefilter
              └─► Pair subgraph construction
                  └─► Sparse MoE with Graph Attention Transformer experts
                      ├─► Multi-label training: Focal Loss
                      ├─► Router load-balance loss (Switch-style)
                      └─► Structural contrastive prototype alignment

INFERENCE / EVAL BEHAVIOR:
    - Multi-label prediction via sigmoid + threshold.
    - Result JSON is written per run under --result-dir, prefixed by run name.
    - Heuristic evidence sentence ids are emitted for predicted (h,t,r).
    - Includes DocRED official-style evaluate() helper and training-time fact-level evaluate_model().

TRACKING / ARTIFACTS:
    - W&B run logs train/val/test metrics.
    - Uploads artifacts for source code (moe.py), best checkpoint, and result JSON.

REFERENCES:
    - Switch Transformers (Fedus et al., 2022)
    - DocRED (Yao et al., 2019)
    - Structural Graph Contrastive Learning literature
========================================================================================================
"""

import json
import argparse
import math
import importlib
import os
from datetime import datetime
# Disable C++ level warnings from Torch (like lazyInitCUDA deprecation)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def _is_graphbolt_optional_dep_error(exc):
    msg = str(exc)
    return (
        ("torchdata.datapipes" in msg)
        or ("No module named 'pandas'" in msg)
        or ('No module named "pandas"' in msg)
    )


def _patch_dgl_graphbolt_optional_deps():
    """Disable GraphBolt imports when optional GraphBolt deps are unavailable."""
    candidate_dirs = []
    try:
        import site
        candidate_dirs.extend(site.getsitepackages())
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidate_dirs.append(user_site)
    except Exception:
        pass

    patched = False
    shim = (
        '"""Graphbolt compatibility shim for environments without optional deps."""\n'
        "import warnings\n"
        "try:\n"
        "    from .base import *\n"
        "    from .minibatch import *\n"
        "    from .dataloader import *\n"
        "    from .dataset import *\n"
        "    from .feature_fetcher import *\n"
        "    from .feature_store import *\n"
        "    from .impl import *\n"
        "    from .itemset import *\n"
        "    from .item_sampler import *\n"
        "    from .minibatch_transformer import *\n"
        "    from .negative_sampler import *\n"
        "    from .sampled_subgraph import *\n"
        "    from .subgraph_sampler import *\n"
        "except ModuleNotFoundError as exc:\n"
        "    msg = str(exc)\n"
        "    if ('torchdata.datapipes' in msg) or ('No module named \'pandas\'' in msg) or ('No module named \"pandas\"' in msg):\n"
        "        warnings.warn(f'GraphBolt disabled: missing optional dependency ({msg}).')\n"
        "    else:\n"
        "        raise\n"
    )

    for base in candidate_dirs:
        p = os.path.join(base, "dgl", "graphbolt", "__init__.py")
        if not os.path.isfile(p):
            continue
        try:
            txt = open(p, "r", encoding="utf-8").read()
            if (
                "compatibility shim for environments without optional deps" in txt
                or "compatibility shim for environments without torchdata.datapipes" in txt
            ):
                return True
            with open(p, "w", encoding="utf-8") as f:
                f.write(shim)
            patched = True
        except Exception:
            continue
    return patched


def _safe_import_dgl():
    try:
        return importlib.import_module("dgl"), None
    except Exception as exc:
        if _is_graphbolt_optional_dep_error(exc):
            patched = _patch_dgl_graphbolt_optional_deps()
            if patched:
                try:
                    return importlib.import_module("dgl"), None
                except Exception as exc2:
                    return None, exc2
        return None, exc


dgl, DGL_IMPORT_ERROR = _safe_import_dgl()

try:
    import ot  # POT: Python Optimal Transport
except ImportError:
    ot = None
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoConfig = None
    BitsAndBytesConfig = None

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    get_peft_model = None
    LoraConfig = None
    TaskType = None

import sys
import random
import shutil
from collections import Counter, defaultdict
import re
import errno
import uuid
import functools
from urllib.parse import urlparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from lightning.fabric import seed_everything
except ImportError:
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# To use it:
seed_everything(42) 


def _load_adopt_optimizer_class():
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    adopt_src = os.path.join(workspace_root, "adopt-main", "src")
    if os.path.isdir(adopt_src) and adopt_src not in sys.path:
        sys.path.insert(0, adopt_src)
    from adopt import ADOPT  # type: ignore
    return ADOPT


try:
    ADOPTOptimizer = _load_adopt_optimizer_class()
    ADOPT_IMPORT_ERROR = None
except Exception as _adopt_exc:
    ADOPTOptimizer = None
    ADOPT_IMPORT_ERROR = _adopt_exc

# Import wandb helpers with fallback for environments without wandb.
try:
    import wandb_utils as _wandb_utils_module
    from wandb_utils import init_wandb_run, log_metrics, log_system_metrics, save_model_artifact, save_evaluation_artifact, finish_run
except Exception:
    _wandb_utils_module = None

    def init_wandb_run(*args, **kwargs):
        return False

    def log_metrics(*args, **kwargs):
        return None

    def log_system_metrics(*args, **kwargs):
        return None

    def save_model_artifact(*args, **kwargs):
        return False

    def save_evaluation_artifact(*args, **kwargs):
        return False

    def finish_run(*args, **kwargs):
        return None

from config import Config
try:
    import wandb
except ImportError:
    wandb = None

# Monkey-patch set_submodule if missing
if not hasattr(nn.Module, "set_submodule"):
    def set_submodule(self, target: str, module: nn.Module) -> None:
        atoms = target.split(".")
        name = atoms.pop()
        mod = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no attribute `" + item + "`")
            mod = getattr(mod, item)
            if not isinstance(mod, nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")
        setattr(mod, name, module)
    nn.Module.set_submodule = set_submodule

# ==========================================
# 1. DATASET & CANDIDATE GENERATION
# ==========================================
class DocREDDataset(torch.utils.data.Dataset):
    """Simple wrapper for DocRED data samples."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def _sent_offsets(item):
    sents = item.get('sents', [])
    offsets = [0]
    total = 0
    for s in sents:
        total += len(s)
        offsets.append(total)
    return offsets

def _global_span(item, mention):
    sent_id = int(mention.get('sent_id', 0))
    start, end = int(mention['pos'][0]), int(mention['pos'][1])
    offsets = _sent_offsets(item)
    if sent_id < 0 or sent_id >= len(offsets) - 1:
        return start, end
    base = offsets[sent_id]
    return base + start, base + end

def _select_primary_mention(item, entity_mentions):
    # Choose earliest mention for deterministic markers.
    best = None
    best_start = None
    for m in entity_mentions:
        s, _ = _global_span(item, m)
        if best is None or s < best_start:
            best = m
            best_start = s
    return best

def build_marked_words(item, h_idx, t_idx):
    """Insert [E1]/[E2] markers around head/tail mentions in flattened words."""
    words = [w for s in item['sents'] for w in s]
    vertex_set = item.get('vertex_set', item.get('vertexSet', []))
    if not vertex_set:
        return words

    h_mentions = vertex_set[h_idx]
    t_mentions = vertex_set[t_idx]
    h_m = _select_primary_mention(item, h_mentions)
    t_m = _select_primary_mention(item, t_mentions)
    if h_m is None or t_m is None:
        return words

    h_s, h_e = _global_span(item, h_m)
    t_s, t_e = _global_span(item, t_m)

    start_markers = {}
    end_markers = {}

    start_markers.setdefault(h_s, []).append("[E1]")
    end_markers.setdefault(h_e, []).append("[/E1]")
    start_markers.setdefault(t_s, []).append("[E2]")
    end_markers.setdefault(t_e, []).append("[/E2]")

    for k in start_markers:
        start_markers[k] = sorted(start_markers[k])
    for k in end_markers:
        end_markers[k] = sorted(end_markers[k])

    marked = []
    for i, w in enumerate(words):
        if i in start_markers:
            marked.extend(start_markers[i])
        marked.append(w)
        if (i + 1) in end_markers:
            marked.extend(end_markers[i + 1])
    return marked

def build_pair_batch_items(item, pairs):
    """Create lightweight items for per-pair marker encoding."""
    batch_items = []
    for h_idx, t_idx in pairs:
        batch_items.append({
            '_marked_words': build_marked_words(item, h_idx, t_idx),
            'sents': item.get('sents', [])
        })
    return batch_items


def infer_pair_evidence_sent_ids(item, h_idx, t_idx, max_evidence=3):
    """Heuristic evidence sentence ids for a predicted (h,t) relation."""
    vertex_set = item.get('vertex_set', item.get('vertexSet', []))
    if not vertex_set or h_idx >= len(vertex_set) or t_idx >= len(vertex_set):
        return []

    h_mentions = vertex_set[h_idx]
    t_mentions = vertex_set[t_idx]
    h_sents = {int(m.get('sent_id', 0)) for m in h_mentions}
    t_sents = {int(m.get('sent_id', 0)) for m in t_mentions}

    cooccur = sorted(list(h_sents.intersection(t_sents)))
    if cooccur:
        return cooccur[:max(1, int(max_evidence))]

    merged = sorted(list(h_sents.union(t_sents)))
    return merged[:max(1, int(max_evidence))]

def extract_marker_pair_features(outputs, encodings, e1_id, e2_id):
    """Extract [E1] and [E2] token embeddings as pair features."""
    hidden = outputs.hidden_states[-1]
    input_ids = encodings['input_ids']
    batch_size = input_ids.shape[0]
    feats = []
    mask = []
    for i in range(batch_size):
        ids = input_ids[i]
        e1_pos = (ids == e1_id).nonzero(as_tuple=True)[0]
        e2_pos = (ids == e2_id).nonzero(as_tuple=True)[0]
        if e1_pos.numel() == 0 or e2_pos.numel() == 0:
            feats.append(torch.zeros(hidden.shape[-1] * 2, device=hidden.device, dtype=hidden.dtype))
            mask.append(False)
            continue
        h_vec = hidden[i, e1_pos[0]]
        t_vec = hidden[i, e2_pos[0]]
        feats.append(torch.cat([h_vec, t_vec]))
        mask.append(True)
    return torch.stack(feats), torch.tensor(mask, device=hidden.device)

def collate_fn(batch, tokenizer, max_length=1024):
    """
    Collate function: tokenizes document sentences for LLM.
    If an item includes `_marked_words`, those are used directly.
    Returns word-level tokenization info for entity alignment.
    """
    all_inputs = []
    if hasattr(tokenizer, 'add_prefix_space'):
        tokenizer.add_prefix_space = True
    for item in batch:
        if '_marked_words' in item:
            words = item['_marked_words']
        else:
            words = [w for s in item['sents'] for w in s]
        all_inputs.append(words)
    encodings = tokenizer(
        all_inputs,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {'items': batch, 'encodings': encodings}


def move_batch_to_device(batch, device):
    """Move batch tensors to target device in main process (not in DataLoader workers)."""
    batch['encodings'] = batch['encodings'].to(device)
    return batch

class CandidateGenerator:
    """
    Generate entity pair candidates for relation extraction.
    
    Strategy: Include pairs that are reachable within max_dist tokens or co-occur.
    This filters out distant pairs unlikely to have direct relations.
    """
    def __init__(self, max_dist=128):
        self.max_dist = max_dist

    def _pair_fast_score(self, item, vertex_set, h, t):
        """Fast EP-RSR-style score to filter obvious NA pairs before graph reasoning."""
        h_mentions = vertex_set[h]
        t_mentions = vertex_set[t]
        h_sents = {m['sent_id'] for m in h_mentions}
        t_sents = {m['sent_id'] for m in t_mentions}
        sent_overlap = 1.0 if h_sents.intersection(t_sents) else 0.0

        min_dist = float('inf')
        for hm in h_mentions:
            for tm in t_mentions:
                hm_s, _ = self._global_span(item, hm)
                tm_s, _ = self._global_span(item, tm)
                min_dist = min(min_dist, abs(hm_s - tm_s))
        dist_score = 1.0 / (1.0 + float(min_dist))
        return sent_overlap + dist_score

    def _sent_offsets(self, item):
        sents = item.get('sents', [])
        offsets = [0]
        total = 0
        for s in sents:
            total += len(s)
            offsets.append(total)
        return offsets

    def _global_span(self, item, mention):
        """Convert DocRED mention span (sentence-local) to global word indices."""
        sent_id = int(mention.get('sent_id', 0))
        start, end = int(mention['pos'][0]), int(mention['pos'][1])
        offsets = self._sent_offsets(item)
        if sent_id < 0 or sent_id >= len(offsets) - 1:
            return start, end
        base = offsets[sent_id]
        return base + start, base + end

    def generate_candidates(self, item, vertex_set):
        """
        Args:
            item: DocRE sample
            vertex_set: list of entity mentions
        Returns:
            candidates: list of (h_idx, t_idx) pairs
        """
        num_entities = len(vertex_set)
        candidates = []
        for h in range(num_entities):
            for t in range(num_entities):
                if h == t: continue
                h_mentions = vertex_set[h]
                t_mentions = vertex_set[t]
                
                reachable = False
                h_sents = {m['sent_id'] for m in h_mentions}
                t_sents = {m['sent_id'] for m in t_mentions}
                
                if h_sents.intersection(t_sents):
                    reachable = True
                else:
                    min_dist = float('inf')
                    for hm in h_mentions:
                        for tm in t_mentions:
                            hm_s, _ = self._global_span(item, hm)
                            tm_s, _ = self._global_span(item, tm)
                            d = abs(hm_s - tm_s)
                            if d < min_dist: min_dist = d
                    if min_dist <= self.max_dist:
                        reachable = True
                
                if reachable:
                    candidates.append((h, t))
        return candidates

    def prefilter_candidates(self, item, vertex_set, candidates, keep_ratio=0.3, must_keep=None):
        """Binary pre-filter (EP-RSR style): keep top-scoring pairs and always keep must_keep."""
        if not candidates:
            return []
        must_keep = set() if must_keep is None else set(must_keep)

        scored = []
        for p in candidates:
            scored.append((self._pair_fast_score(item, vertex_set, p[0], p[1]), p))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_n = max(1, int(len(scored) * float(keep_ratio)))
        kept = [p for _, p in scored[:top_n]]
        kept_set = set(kept)
        for p in must_keep:
            if p not in kept_set:
                kept.append(p)
                kept_set.add(p)
        return kept


TYPE_CONSTRAINTS = {
    "P17": ({"ORG", "PER", "MISC"}, {"LOC"}),
    "P131": ({"ORG", "LOC", "MISC"}, {"LOC"}),
    "P150": ({"LOC"}, {"LOC"}),
    "P19": ({"PER"}, {"LOC"}),
    "P569": ({"PER"}, {"TIME"}),
    "P570": ({"PER"}, {"TIME"}),
    "P175": ({"MISC", "WORK", "ORG"}, {"PER"}),
    "P161": ({"MISC", "WORK"}, {"PER"}),
    "P264": ({"MISC", "WORK"}, {"ORG"}),
}


def _entity_type_set(entity_mentions):
    return {str(m.get("type", "MISC")) for m in entity_mentions}


def _relation_type_allowed(rel, h_types, t_types):
    if rel not in TYPE_CONSTRAINTS:
        return True
    h_ok, t_ok = TYPE_CONSTRAINTS[rel]
    return (len(h_ok.intersection(h_types)) > 0) and (len(t_ok.intersection(t_types)) > 0)


def load_json_robust(path):
    """Load JSON with a fallback for truncated/minified files by clipping to the last closing bracket."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(path, "r") as f:
            text = f.read()
        last = text.rfind("]")
        if last < 0:
            last = len(text) - 1
        clipped = text[: last + 1]
        # Fix common malformed-json artifacts in huge generated files.
        clipped = re.sub(r",\s*\]", "]", clipped)
        clipped = re.sub(r",\s*\}", "}", clipped)
        try:
            return json.loads(clipped)
        except json.JSONDecodeError:
            # Last-resort recovery for truncated array files: decode valid prefix items.
            dec = json.JSONDecoder()
            data = []
            i = clipped.find("[")
            if i < 0:
                raise
            i += 1
            n = len(clipped)
            while i < n:
                while i < n and clipped[i] in " \t\r\n,":
                    i += 1
                if i >= n or clipped[i] == "]":
                    break
                try:
                    obj, j = dec.raw_decode(clipped, i)
                except json.JSONDecodeError:
                    break
                data.append(obj)
                i = j
            if not data:
                raise
            print(f"[WARN] Recovered partial JSON from {path}: {len(data)} items")
            return data


def clean_distant_supervision_data(distant_data, dev_data, rel2id, top_k=2, eps=1e-6):
    """
    DS cleaning:
    1) Type-constraint filtering
    2) MIL top-k denoising per (h,t) bag using evidence-aware score
    """
    cleaned = []
    stats = {
        "input_docs": len(distant_data),
        "kept_docs": 0,
        "dropped_bad_index": 0,
        "dropped_type": 0,
        "dropped_unknown_rel": 0,
        "dropped_by_mil": 0,
    }

    for item in distant_data:
        vertex_set = item.get("vertexSet", item.get("vertex_set", []))
        labels = item.get("labels", [])
        pair_bags = defaultdict(list)

        for lbl in labels:
            rel = lbl.get("r")
            h = int(lbl.get("h", -1))
            t = int(lbl.get("t", -1))
            if rel not in rel2id:
                stats["dropped_unknown_rel"] += 1
                continue
            if h < 0 or t < 0 or h >= len(vertex_set) or t >= len(vertex_set):
                stats["dropped_bad_index"] += 1
                continue

            h_types = _entity_type_set(vertex_set[h])
            t_types = _entity_type_set(vertex_set[t])
            if not _relation_type_allowed(rel, h_types, t_types):
                stats["dropped_type"] += 1
                continue

            ev = lbl.get("evidence", [])
            score = 1.0 + 0.1 * float(len(ev))
            pair_bags[(h, t)].append((score, dict(lbl)))

        new_labels = []
        for _, bag in pair_bags.items():
            bag.sort(key=lambda x: x[0], reverse=True)
            kept = bag[: max(1, int(top_k))]
            stats["dropped_by_mil"] += max(0, len(bag) - len(kept))
            for _, lbl in kept:
                new_labels.append(lbl)

        if not new_labels:
            continue

        out_item = dict(item)
        out_item["labels"] = new_labels
        cleaned.append(out_item)

    stats["kept_docs"] = len(cleaned)
    return cleaned, stats


def build_pair_relation_sets(labels, rel2id):
    """Build (h,t)->set(rel_id)."""
    pair_to_rels = defaultdict(set)
    for lbl in labels:
        rel = lbl.get('r')
        if rel not in rel2id:
            continue
        pair = (int(lbl.get('h', -1)), int(lbl.get('t', -1)))
        if pair[0] < 0 or pair[1] < 0:
            continue
        rel_id = rel2id[rel]
        pair_to_rels[pair].add(rel_id)
    return pair_to_rels


def build_multi_hot_targets(pairs, pair_to_rels, num_relations, device):
    targets = torch.zeros((len(pairs), num_relations), dtype=torch.float32, device=device)
    for i, pair in enumerate(pairs):
        for rel_id in pair_to_rels.get(pair, set()):
            targets[i, rel_id] = 1.0
    return targets


def limit_candidates_preserve_must_keep(candidates, must_keep, max_pairs):
    """Cap candidates without dropping required positives."""
    if max_pairs <= 0 or len(candidates) <= max_pairs:
        return candidates
    must_keep = set(must_keep)
    must = [p for p in candidates if p in must_keep]
    others = [p for p in candidates if p not in must_keep]
    eff_max = max(max_pairs, len(must))
    return must + others[: max(0, eff_max - len(must))]


def structural_contrastive_loss(pair_repr, label_multi_hot, prototypes, temperature=0.1):
    """Prototype alignment for multi-label pairs using normalized multi-positive targets."""
    valid_mask = label_multi_hot.sum(dim=-1) > 0
    if valid_mask.sum() == 0:
        return torch.zeros((), device=pair_repr.device, dtype=pair_repr.dtype)

    z = F.normalize(pair_repr[valid_mask].float(), dim=-1)
    targets = label_multi_hot[valid_mask].float()
    targets = targets / (targets.sum(dim=-1, keepdim=True) + 1e-9)
    proto = F.normalize(prototypes.get_all().float(), dim=-1)
    sim = z @ proto.t()
    logits = sim / max(1e-6, float(temperature))
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def focal_loss_with_logits(
    logits,
    targets,
    gamma=2.0,
    alpha=0.25,
    eps=1e-8,
    reduction="mean",
):
    """Multi-label focal loss on logits."""
    x = logits.float()
    y = targets.float()

    probs = torch.sigmoid(x).clamp(min=eps, max=1.0 - eps)
    pt = probs * y + (1.0 - probs) * (1.0 - y)

    if alpha is None:
        alpha_t = torch.ones_like(y)
    else:
        alpha = float(alpha)
        alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)

    focal_weight = alpha_t * torch.pow((1.0 - pt).clamp(min=0.0), float(gamma))
    bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    loss = focal_weight * bce

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def compute_relation_frequency(data, rel2id):
    """Count per-relation frequencies from training labels."""
    freq = np.zeros((len(rel2id),), dtype=np.int64)
    if not data:
        return freq
    for item in data:
        for lbl in item.get('labels', []):
            rel = lbl.get('r')
            if rel in rel2id:
                freq[rel2id[rel]] += 1
    return freq


def build_tail_buckets(rel_freq):
    """Split relations into head/medium/tail buckets by train frequency tertiles."""
    rel_freq = np.asarray(rel_freq, dtype=np.float64)
    non_zero = rel_freq[rel_freq > 0]
    if non_zero.size == 0:
        return {
            'head': set(),
            'medium': set(),
            'tail': set(range(rel_freq.shape[0])),
        }
    q1 = float(np.quantile(non_zero, 1.0 / 3.0))
    q2 = float(np.quantile(non_zero, 2.0 / 3.0))
    buckets = {'head': set(), 'medium': set(), 'tail': set()}
    for rel_id, c in enumerate(rel_freq.tolist()):
        if c <= 0 or c <= q1:
            buckets['tail'].add(rel_id)
        elif c <= q2:
            buckets['medium'].add(rel_id)
        else:
            buckets['head'].add(rel_id)
    return buckets


def compute_long_tail_metrics(pred_facts, gold_facts, num_relations, tail_buckets):
    """Compute Macro-F1, Tail bucket F1s, and relation coverage."""
    tp_by_rel = np.zeros((num_relations,), dtype=np.int64)
    fp_by_rel = np.zeros((num_relations,), dtype=np.int64)
    fn_by_rel = np.zeros((num_relations,), dtype=np.int64)

    inter = pred_facts.intersection(gold_facts)
    pred_only = pred_facts.difference(gold_facts)
    gold_only = gold_facts.difference(pred_facts)

    for _, _, _, rel_id in inter:
        tp_by_rel[int(rel_id)] += 1
    for _, _, _, rel_id in pred_only:
        fp_by_rel[int(rel_id)] += 1
    for _, _, _, rel_id in gold_only:
        fn_by_rel[int(rel_id)] += 1

    rel_f1 = np.zeros((num_relations,), dtype=np.float64)
    for rid in range(num_relations):
        tp = float(tp_by_rel[rid])
        fp = float(fp_by_rel[rid])
        fn = float(fn_by_rel[rid])
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        rel_f1[rid] = 0.0 if (p + r) <= 0.0 else (2.0 * p * r / (p + r))

    macro_f1 = float(rel_f1.mean()) if rel_f1.size > 0 else 0.0

    def _bucket_f1(name):
        ids = sorted(list(tail_buckets.get(name, set()))) if tail_buckets else []
        if not ids:
            return 0.0
        return float(np.mean(rel_f1[ids]))

    pred_rel_ids = {int(rel_id) for (_, _, _, rel_id) in pred_facts}
    coverage_count = len(pred_rel_ids)
    coverage_ratio = coverage_count / max(1, num_relations)

    return {
        'macro_f1': macro_f1,
        'tail_f1_head': _bucket_f1('head'),
        'tail_f1_medium': _bucket_f1('medium'),
        'tail_f1_tail': _bucket_f1('tail'),
        'coverage_relations_nonzero': float(coverage_count),
        'coverage_ratio': float(coverage_ratio),
    }


def predict_multi_label_relations(logits, threshold=0.5):
    logits_fp32 = logits.float()
    probs = torch.sigmoid(logits_fp32)
    pred_mask = probs >= float(threshold)
    return pred_mask, probs, logits_fp32


def compute_fact_f1(pred_facts, gold_facts):
    tp = len(pred_facts.intersection(gold_facts))
    fp = len(pred_facts) - tp
    fn = len(gold_facts) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall / (precision + recall))
    return precision, recall, f1


def _build_entity_relation_facts(data):
    """Build train facts (entity-name, entity-name, relation) for ignore-train metrics."""
    facts = set()
    if not data:
        return facts
    for item in data:
        vertex_set = item.get('vertex_set', item.get('vertexSet', []))
        labels = item.get('labels', [])
        for lbl in labels:
            try:
                h_idx = int(lbl['h'])
                t_idx = int(lbl['t'])
                rel = str(lbl['r'])
            except Exception:
                continue
            if h_idx < 0 or t_idx < 0 or h_idx >= len(vertex_set) or t_idx >= len(vertex_set):
                continue
            for h_m in vertex_set[h_idx]:
                for t_m in vertex_set[t_idx]:
                    h_name = str(h_m.get('name', ''))
                    t_name = str(t_m.get('name', ''))
                    if h_name and t_name:
                        facts.add((h_name, t_name, rel))
    return facts


def _compute_official_style_docred_metrics(infer_results, eval_data, train_facts_annotated=None, train_facts_distant=None):
    """Compute official-style DocRED metrics from in-memory predictions and eval set."""
    train_facts_annotated = train_facts_annotated or set()
    train_facts_distant = train_facts_distant or set()

    gold_map = {}
    total_gold_relations = 0
    total_gold_evidences = 0
    title_to_vertex = {}

    for item in eval_data:
        title = item.get('title', 'unknown')
        vertex_set = item.get('vertex_set', item.get('vertexSet', []))
        title_to_vertex[title] = vertex_set
        for lbl in item.get('labels', []):
            try:
                h_idx = int(lbl['h'])
                t_idx = int(lbl['t'])
                rel = str(lbl['r'])
            except Exception:
                continue
            evidence = set(int(e) for e in lbl.get('evidence', []))
            gold_map[(title, rel, h_idx, t_idx)] = evidence
            total_gold_relations += 1
            total_gold_evidences += len(evidence)

    pred_map = {}
    for pred in infer_results:
        try:
            key = (
                pred.get('title', 'unknown'),
                str(pred['r']),
                int(pred['h_idx']),
                int(pred['t_idx']),
            )
        except Exception:
            continue
        evidence = set(int(e) for e in pred.get('evidence', []))
        if key in pred_map:
            pred_map[key] = pred_map[key].union(evidence)
        else:
            pred_map[key] = evidence

    pred_keys = set(pred_map.keys())
    gold_keys = set(gold_map.keys())
    inter = pred_keys.intersection(gold_keys)

    correct_re = len(inter)
    pred_re = len(pred_keys)

    correct_evidence = 0
    pred_evi = 0
    for key, evi in pred_map.items():
        pred_evi += len(evi)
        if key in gold_map:
            correct_evidence += len(evi.intersection(gold_map[key]))

    re_p = correct_re / max(1, pred_re)
    re_r = correct_re / max(1, total_gold_relations)
    re_f1 = 0.0 if (re_p + re_r) <= 0 else (2.0 * re_p * re_r / (re_p + re_r))

    evi_p = correct_evidence / max(1, pred_evi)
    evi_r = correct_evidence / max(1, total_gold_evidences)
    evi_f1 = 0.0 if (evi_p + evi_r) <= 0 else (2.0 * evi_p * evi_r / (evi_p + evi_r))

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    for title, rel, h_idx, t_idx in inter:
        vertex_set = title_to_vertex.get(title, [])
        if h_idx < 0 or t_idx < 0 or h_idx >= len(vertex_set) or t_idx >= len(vertex_set):
            continue
        in_ann = False
        in_dist = False
        for h_m in vertex_set[h_idx]:
            for t_m in vertex_set[t_idx]:
                tup = (str(h_m.get('name', '')), str(t_m.get('name', '')), rel)
                if tup in train_facts_annotated:
                    in_ann = True
                if tup in train_facts_distant:
                    in_dist = True
            if in_ann and in_dist:
                break
        if in_ann:
            correct_in_train_annotated += 1
        if in_dist:
            correct_in_train_distant += 1

    denom_ann = max(1, pred_re - correct_in_train_annotated)
    denom_dist = max(1, pred_re - correct_in_train_distant)
    re_p_ignore_ann = (correct_re - correct_in_train_annotated) / denom_ann
    re_p_ignore_dist = (correct_re - correct_in_train_distant) / denom_dist
    re_f1_ignore_ann = 0.0 if (re_p_ignore_ann + re_r) <= 0 else (2.0 * re_p_ignore_ann * re_r / (re_p_ignore_ann + re_r))
    re_f1_ignore_dist = 0.0 if (re_p_ignore_dist + re_r) <= 0 else (2.0 * re_p_ignore_dist * re_r / (re_p_ignore_dist + re_r))

    return {
        'RE_P': float(re_p),
        'RE_R': float(re_r),
        'F1-RE': float(re_f1),
        'Evidence_P': float(evi_p),
        'Evidence_R': float(evi_r),
        'F1-Evidence': float(evi_f1),
        'RE_ignore_annotated_P': float(re_p_ignore_ann),
        'RE_ignore_annotated_R': float(re_r),
        'RE_ignore_annotated_F1': float(re_f1_ignore_ann),
        'RE_ignore_distant_P': float(re_p_ignore_dist),
        'RE_ignore_distant_R': float(re_r),
        'RE_ignore_distant_F1': float(re_f1_ignore_dist),
    }

class DocREDGraphBuilder:
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
            sorted_nodes = sorted(sorted_nodes)

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
        return g, adj, node_to_idx[h_id], node_to_idx[t_id]

# ==========================================
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
    
    Math: min_T (1-α)⟨T,M⟩ + α·GW(A,B,T) - ε·H(T)
    
    Args:
        g1_nodes, g2_nodes: node embeddings (N1, D), (N2, D)
        g1_adj, g2_adj: adjacency (N1, N1), (N2, N2)
        alpha: balance between feature (1-α) and structure (α)
        reg: entropy regularization ε
    
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
    
    Math: p(e|x) = softmax(W·x + noise)
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


class DifficultyAwareRouter(nn.Module):
    """
    Router for depth-heterogeneous experts.

    Routes each entity pair using BOTH its contextual pair features AND structural
    difficulty signals derived from the pair's subgraph (graph hop-distance between
    h and t, subgraph size/density). The intent: steer "hard" pairs (long reasoning
    paths, larger neighborhoods) toward deeper experts and "easy" intra-sentence pairs
    toward shallow experts — adaptive computation matched to reasoning difficulty.

    Math: logits = W·[pair_features ⊕ difficulty_feats] (+ noise during training);
    top-1 dispatch e* = argmax(logits). Difficulty features are appended so the router
    can condition routing on structure, not just content.
    """
    def __init__(self, in_dim, num_experts, num_diff_feats=4, noise_eps=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.num_diff_feats = num_diff_feats
        self.noise_eps = noise_eps
        self.linear = nn.Linear(in_dim + num_diff_feats, num_experts)

    def forward(self, x, diff_feats, training=True):
        """
        Args:
            x: pair features (B, in_dim)
            diff_feats: structural difficulty features (B, num_diff_feats), in [0,1]
            training: add exploration noise only during training
        Returns:
            logits: routing scores (B, num_experts)
            top_idx: selected expert indices (B,)
        """
        h = torch.cat([x, diff_feats.to(x.dtype)], dim=-1)
        logits = self.linear(h)
        if training:
            logits = logits + torch.randn_like(logits) * self.noise_eps
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
    
    Pipeline: LLM embeddings → Entity pairs → Graph construction → Sparse MoE → Relation classifier
    
    MoE learns: p(r|h,t,D) = Σ_e p(e|h,t,D) · p(r|h,t,D,e)
             Router ≈ posterior expert selection
             Expert ≈ conditional graph encoder
    """
    def __init__(self, llm_model, num_relations, num_experts=4, expert_dim=128, noise_scale=1e-2, capacity_factor=1.25, graph_device='cpu', expert_depths=None, use_shared_expert=True, num_diff_feats=4, hop_max=4):
        super().__init__()
        self.llm = llm_model
        self.expert_dim = expert_dim
        self.noise_scale = noise_scale
        self.capacity_factor = capacity_factor
        self.graph_device = graph_device
        self.num_diff_feats = num_diff_feats
        self.hop_max = hop_max

        hidden_size = llm_model.config.hidden_size

        # Heterogeneous-depth experts: each routed expert has a different number of
        # graph-transformer layers (shallow → deep). EC/difficulty routing steers hard
        # (long-reasoning) pairs to deep experts and easy intra-sentence pairs to shallow ones.
        if expert_depths is None:
            # Spread depths across num_experts, e.g. 3 → [1,2,4]; 4 → [1,2,3,4]
            if num_experts <= 1:
                expert_depths = [2]
            elif num_experts == 2:
                expert_depths = [1, 3]
            elif num_experts == 3:
                expert_depths = [1, 2, 4]
            else:
                expert_depths = [max(1, 1 + i) for i in range(num_experts)]
        self.expert_depths = list(expert_depths)
        self.num_experts = len(self.expert_depths)

        self.experts = nn.ModuleList([
            GraphExpert(hidden_size, expert_dim, expert_dim, num_layers=d)
            for d in self.expert_depths
        ])

        # Shared/residual expert: always applied to every pair (dense path). Captures
        # common (head-relation) patterns and stabilises gradients; routed experts add
        # specialisation residually. (DeepSeekMoE-style shared+routed.)
        self.use_shared_expert = use_shared_expert
        self.shared_expert = GraphExpert(hidden_size, expert_dim, expert_dim, num_layers=2) if use_shared_expert else None

        # Difficulty-aware router: conditions routing on pair context + graph structure.
        self.router = DifficultyAwareRouter(hidden_size * 2, self.num_experts, num_diff_feats=num_diff_feats, noise_eps=noise_scale)

        # Classifier: final relation predictor
        self.classifier = nn.Sequential(
            nn.Linear(expert_dim * 2, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim, num_relations)
        )

    def _difficulty_features(self, graph_list):
        """
        Per-pair structural difficulty signal, derived from each pair's subgraph.

        Returns (B, num_diff_feats) in [0,1]:
          [hop_norm, nodes_norm, edges_norm, direct_flag]
        where hop = shortest-path hops between the two is_ht nodes (h,t) on the
        (undirected) subgraph. Disconnected/missing → hardest (1.0). Computed on tiny
        (<=15-node) graphs, so BFS cost is negligible. No call-site changes needed.
        """
        feats = []
        for g in graph_list:
            try:
                n_nodes = int(g.num_nodes())
                n_edges = int(g.num_edges())
                ht = g.ndata['is_ht'].bool().nonzero(as_tuple=True)[0].tolist() if 'is_ht' in g.ndata else []
                hop = self.hop_max  # default: hardest
                if len(ht) >= 2:
                    src, dst = g.edges()
                    adj = defaultdict(set)
                    for u, v in zip(src.tolist(), dst.tolist()):
                        adj[u].add(v)
                        adj[v].add(u)
                    start, goal = ht[0], ht[1]
                    # BFS shortest path (capped at hop_max)
                    seen = {start}
                    frontier = [start]
                    d = 0
                    found = (start == goal)
                    while frontier and not found and d < self.hop_max:
                        d += 1
                        nxt = []
                        for u in frontier:
                            for w in adj[u]:
                                if w == goal:
                                    found = True
                                    break
                                if w not in seen:
                                    seen.add(w)
                                    nxt.append(w)
                            if found:
                                break
                        frontier = nxt
                    hop = d if found else self.hop_max
                hop_norm = min(1.0, hop / float(self.hop_max))
                nodes_norm = min(1.0, n_nodes / 15.0)
                edges_norm = min(1.0, n_edges / float(15 * 14))
                direct = 1.0 if hop == 1 else 0.0
            except Exception:
                hop_norm, nodes_norm, edges_norm, direct = 1.0, 0.0, 0.0, 0.0
            feats.append([hop_norm, nodes_norm, edges_norm, direct])
        return torch.tensor(feats, dtype=torch.float32)
        
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
        out_dim = self.expert_dim * 2

        # Ensure correct dtype
        self.experts.to(dtype)
        self.router.to(dtype)
        self.classifier.to(dtype)
        if self.shared_expert is not None:
            self.shared_expert.to(dtype)

        # Unbatch subgraphs first (needed for structural difficulty features).
        # DGL may be CPU-only even when the rest of the model runs on CUDA.
        graph_list = dgl.unbatch(subgraphs)
        graph_list = [g.to(self.graph_device) for g in graph_list]

        # Step 1: Difficulty-aware routing — condition on pair context + graph structure.
        diff_feats = self._difficulty_features(graph_list).to(device)
        router_logits, top1_idx = self.router(pair_features, diff_feats, training=self.training)

        # Step 2: Gating weights (for gradient flow)
        probs = F.softmax(router_logits, dim=-1)
        top1_prob = probs.gather(1, top1_idx.unsqueeze(-1)).squeeze(-1)

        # Step 3a: Shared/residual expert — always applied to every pair (dense path).
        # Provides the residual anchor; routed experts add specialisation on top.
        if self.use_shared_expert and self.shared_expert is not None:
            all_graphs = dgl.batch(graph_list).to(self.graph_device)
            pair_emb = self.shared_expert(
                all_graphs,
                all_graphs.ndata['h'].to(device=self.graph_device, dtype=dtype),
                pair_features.to(device=self.graph_device, dtype=dtype),
            ).to(device)
        else:
            # Gradient-safe fallback: valid autograd path to pair_features even with no shared expert.
            if pair_features.shape[-1] >= out_dim:
                pair_emb = pair_features[:, :out_dim].clone()
            else:
                pad = torch.zeros((batch_size, out_dim - pair_features.shape[-1]), device=device, dtype=dtype)
                pair_emb = torch.cat([pair_features, pad], dim=-1)

        # Step 3b: Routed heterogeneous-depth experts (capacity-limited), added residually.
        routed_out = torch.zeros((batch_size, out_dim), device=device, dtype=dtype)
        capacity = max(1, int(math.ceil(self.capacity_factor * batch_size / self.num_experts)))

        for e_idx in range(self.num_experts):
            mask = (top1_idx == e_idx)
            selected_indices = mask.nonzero(as_tuple=True)[0]
            if selected_indices.numel() == 0:
                continue

            # Capacity: drop lowest-confidence overflow
            if selected_indices.numel() > capacity:
                sel_probs = top1_prob[selected_indices]
                _, top_k = torch.topk(sel_probs, k=capacity)
                selected_indices = selected_indices[top_k]

            e_pair_feats = pair_features[selected_indices]
            e_graphs = dgl.batch([graph_list[i] for i in selected_indices.tolist()]).to(self.graph_device)

            # Execute ONLY this expert (conditional computation)
            e_out = self.experts[e_idx](
                e_graphs,
                e_graphs.ndata['h'].to(device=self.graph_device, dtype=dtype),
                e_pair_feats.to(device=self.graph_device, dtype=dtype),
            ).to(device)

            # Residual contribution, weighted by router confidence
            routed_out[selected_indices] = e_out * top1_prob[selected_indices].unsqueeze(-1)

        # Residual fusion: shared (dense) + routed (specialised)
        pair_emb = pair_emb + routed_out

        # Step 4: Final classification
        logits = self.classifier(pair_emb)
        return logits, router_logits, top1_idx, pair_emb


def _unwrap_model(m):
    return m.module if hasattr(m, 'module') else m


def _all_reduce_scalar(value, device):
    if not dist.is_available() or not dist.is_initialized():
        return value
    t = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.item())

def switch_load_balance_loss(router_logits, top_idx, num_experts):
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
    return out.to(router_logits.dtype)

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================
def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def evaluate(input_dir, output_dir):
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fact_in_train_annotated = gen_train_facts("../data/train_annotated.json", truth_dir)
        fact_in_train_distant = gen_train_facts("../data/train_distant.json", truth_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')
        output_file = open(output_filename, 'w')

        truth_file = os.path.join(truth_dir, "dev_test.json")
        truth = json.load(open(truth_file))

        std = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}

        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']

                h_idx = label['h']
                t_idx = label['t']
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

        tot_relations = len(std)

        submission_answer_file = os.path.join(submit_dir, "result.json")
        tmp = json.load(open(submission_answer_file))
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i-1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])

        correct_re = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1

        re_p = 1.0 * correct_re / len(submission_answer)
        re_r = 1.0 * correct_re / tot_relations
        if re_p+re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi>0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences
        if evi_p+evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)
        re_p_ignore_train = 1.0 * (correct_re-correct_in_train_distant) / (len(submission_answer)-correct_in_train_distant)

        if re_p_ignore_train_annotated+re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train+re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        print ('RE_F1:', re_f1)
        print ('Evi_F1:', evi_f1)
        print ('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)
        print ('RE_ignore_distant_F1:', re_f1_ignore_train)

        output_file.write("RE_F1: %f\n" % re_f1)
        output_file.write("Evi_F1: %f\n" % evi_f1)

        output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
        output_file.write("RE_ignore_distant_F1: %f\n" % re_f1_ignore_train)

        output_file.close()


def evaluate_model(model, tokenizer, dev_data, graph_builder, rel2id, device, id2rel, debug=False, debug_samples=10, batch_size=1, oracle=False, dump_path=None, candidate_gen=None, use_pair_markers=False, e1_id=None, e2_id=None, candidate_keep_ratio=0.3, threshold_scale=1.0, max_pairs_per_doc=50, args_max_length=1024, result_output_path=None, print_infer_sample=False, tail_buckets=None, train_facts_annotated=None, train_facts_distant=None):
    core_model = _unwrap_model(model)
    core_model.eval()
    pred_facts = set()
    gold_facts = set()
    dump_results = []
    infer_results = []
    printed_input_sample = False
    printed_output_sample = False
    
    eval_data = dev_data[:debug_samples] if debug else dev_data
    eval_collate = functools.partial(collate_fn, tokenizer=tokenizer, max_length=args_max_length)
    dataloader = DataLoader(
        DocREDDataset(eval_data),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=eval_collate,
    )
    
    print(f"\nEvaluating {'(ORACLE)' if oracle else ''}... {len(eval_data)} docs")
    with torch.no_grad():
        for doc_idx, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)
            item = batch['items'][0]
            encodings = batch['encodings']
            outputs = core_model.llm(**encodings, output_hidden_states=True)
            doc_embeds = outputs.hidden_states[-1][0]
            w_ids = encodings.word_ids(batch_index=0)
            
            vertex_set = item.get('vertex_set', item.get('vertexSet', []))
            pair_to_rels = build_pair_relation_sets(item.get('labels', []), rel2id)
            must_keep_pairs = set(pair_to_rels.keys())

            for (h_idx, t_idx), rel_ids in pair_to_rels.items():
                for rel_id in rel_ids:
                    gold_facts.add((doc_idx, int(h_idx), int(t_idx), int(rel_id)))

            entity_pairs = list(must_keep_pairs) if oracle else candidate_gen.generate_candidates(item, vertex_set)
            if not oracle:
                entity_pairs = candidate_gen.prefilter_candidates(
                    item,
                    vertex_set,
                    entity_pairs,
                    keep_ratio=candidate_keep_ratio,
                    must_keep=must_keep_pairs,
                )
                entity_pairs = limit_candidates_preserve_must_keep(entity_pairs, must_keep_pairs, max_pairs_per_doc)

            if not entity_pairs:
                continue

            doc_results = {"doc_id": item.get('title', 'unknown'), "entity_pairs": []}
            pair_batch_size = 10
            decision_threshold = min(0.95, max(0.05, 0.5 * float(threshold_scale)))
            for p_start in range(0, len(entity_pairs), pair_batch_size):
                p_end = min(p_start + pair_batch_size, len(entity_pairs))
                batch_pairs = entity_pairs[p_start:p_end]
                subgraphs, pair_features = [], []
                marker_feats = None
                marker_mask = None
                if use_pair_markers:
                    pair_items = build_pair_batch_items(item, batch_pairs)
                    pair_enc = collate_fn(pair_items, tokenizer, max_length=args_max_length)['encodings'].to(device)
                    pair_out = core_model.llm(**pair_enc, output_hidden_states=True)
                    marker_feats, marker_mask = extract_marker_pair_features(pair_out, pair_enc, e1_id, e2_id)
                for h_idx, t_idx in batch_pairs:
                    g, _, _, _ = graph_builder.build_pair_subgraph(item, doc_embeds, w_ids, h_idx, t_idx)
                    subgraphs.append(g)
                    ht_feats = g.ndata['h'][g.ndata['is_ht'].bool()]
                    h_f, t_f = (ht_feats[0], ht_feats[1]) if ht_feats.shape[0] >= 2 else (ht_feats[0], ht_feats[0])
                    pair_features.append(torch.cat([h_f, t_f]))

                if use_pair_markers and marker_feats is not None:
                    for idx in range(len(pair_features)):
                        if marker_mask[idx]:
                            marker_vec = marker_feats[idx].to(pair_features[idx].device)
                            pair_features[idx] = 0.5 * pair_features[idx] + 0.5 * marker_vec
                
                logits, _, _, _ = core_model(dgl.batch(subgraphs), torch.stack(pair_features).to(device))
                preds_tensor, probs_tensor, _ = predict_multi_label_relations(
                    logits,
                    threshold=decision_threshold,
                )
                probs_batch = probs_tensor.float().cpu().numpy()
                preds_batch = preds_tensor.cpu().numpy().astype(bool)
                
                for j, pair in enumerate(batch_pairs):
                    pred_rel_ids = np.where(preds_batch[j])[0].tolist()
                    for rel_id in pred_rel_ids:
                        pred_facts.add((doc_idx, int(pair[0]), int(pair[1]), int(rel_id)))
                        pred_item = {
                            "title": item.get('title', 'unknown'),
                            "h_idx": int(pair[0]),
                            "t_idx": int(pair[1]),
                            "r": id2rel[int(rel_id)],
                            "evidence": infer_pair_evidence_sent_ids(item, int(pair[0]), int(pair[1])),
                        }
                        infer_results.append(pred_item)
                        if print_infer_sample and not printed_output_sample:
                            print("[INFER-DEBUG] Output sample after inference:")
                            print(json.dumps(pred_item, ensure_ascii=False))
                            printed_output_sample = True

                    if print_infer_sample and not printed_input_sample:
                        sample_input = {
                            "title": item.get('title', 'unknown'),
                            "h_idx": int(pair[0]),
                            "t_idx": int(pair[1]),
                            "r": "NA",
                            "evidence": infer_pair_evidence_sent_ids(item, int(pair[0]), int(pair[1])),
                        }
                        print("[INFER-DEBUG] Input sample before inference:")
                        print(json.dumps(sample_input, ensure_ascii=False))
                        printed_input_sample = True

                    if len(dump_results) < 5:
                        gold_rel_ids = sorted(pair_to_rels.get(pair, set()))
                        doc_results["entity_pairs"].append({
                            "h": vertex_set[pair[0]][0]['name'],
                            "t": vertex_set[pair[1]][0]['name'],
                            "gold": [id2rel[g] for g in gold_rel_ids],
                            "pred": [id2rel[p] for p in pred_rel_ids],
                        })
            if doc_results["entity_pairs"]:
                dump_results.append(doc_results)

    if dump_path and dump_results:
        with open(dump_path, 'w') as f: json.dump(dump_results, f, indent=2)

    if result_output_path is not None:
        with open(result_output_path, 'w') as f:
            json.dump(infer_results, f, ensure_ascii=False, indent=2)
        print(f"[INFER-DEBUG] Saved inference output to {result_output_path} ({len(infer_results)} items)")

    if print_infer_sample and not printed_output_sample:
        print("[INFER-DEBUG] Output sample after inference:")
        print(json.dumps({"title": "N/A", "h_idx": -1, "t_idx": -1, "r": "NA", "evidence": []}, ensure_ascii=False))

    precision, recall, f1 = compute_fact_f1(pred_facts, gold_facts)
    long_tail_metrics = compute_long_tail_metrics(
        pred_facts,
        gold_facts,
        num_relations=len(rel2id),
        tail_buckets=tail_buckets,
    )
    out_metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }
    out_metrics.update(long_tail_metrics)
    out_metrics.update(
        _compute_official_style_docred_metrics(
            infer_results=infer_results,
            eval_data=eval_data,
            train_facts_annotated=train_facts_annotated,
            train_facts_distant=train_facts_distant,
        )
    )
    return out_metrics

# ==========================================
# 5. MAIN PIPELINE (6-Layer Architecture)
# ==========================================
def main():
    def _ensure_stable_hf_cache_dir():
        """Use a stable local cache to avoid stale NFS handles in shared workspaces."""
        user_hf_home = os.environ.get("HF_HOME", "").strip()
        workspace_root = os.path.dirname(os.path.abspath(__file__))
        default_hf_home = os.path.join(workspace_root, "tmp", "hf_home")
        if not user_hf_home or user_hf_home.startswith("/workspace/"):
            hf_home = default_hf_home
            os.environ["HF_HOME"] = hf_home
        else:
            hf_home = user_hf_home

        hub_cache = os.path.join(hf_home, "hub")
        transformers_cache = os.path.join(hf_home, "transformers")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)
        os.makedirs(hf_home, exist_ok=True)
        os.makedirs(hub_cache, exist_ok=True)
        os.makedirs(transformers_cache, exist_ok=True)
        return hf_home

    def _is_stale_handle_error(exc: Exception) -> bool:
        msg = str(exc)
        return (
            isinstance(exc, OSError)
            and (getattr(exc, "errno", None) == errno.ESTALE or "Stale file handle" in msg)
        )

    def _load_with_cache_retry(loader_cls, model_id, load_kwargs, artifact_name="artifact"):
        """Retry once with a unique workspace-local tmp cache dir when ESTALE occurs."""
        try:
            return loader_cls.from_pretrained(model_id, **load_kwargs)
        except Exception as e:
            if not _is_stale_handle_error(e):
                raise
            workspace_root = os.path.dirname(os.path.abspath(__file__))
            retry_cache = os.path.join(
                workspace_root,
                "tmp",
                f"hf_home_retry_{os.getpid()}_{uuid.uuid4().hex[:8]}",
            )
            os.makedirs(retry_cache, exist_ok=True)
            retry_kwargs = dict(load_kwargs)
            retry_kwargs["cache_dir"] = retry_cache
            print(f"[WARN] {artifact_name} load failed due to stale handle. Retrying with cache_dir={retry_cache}")
            return loader_cls.from_pretrained(model_id, **retry_kwargs)

    hf_home = _ensure_stable_hf_cache_dir()
    print(f"[HF] Using cache root: {hf_home}")

    parser = argparse.ArgumentParser(description="DocRE with Sparse MoE + DS + Structural Contrastive Learning")
    parser.add_argument(
        '--stage',
        type=str,
        default='train',
        choices=['sanity', 'candidates', 'ds', 'overfit', 'train'],
        help='Pipeline stage. Defaults to train (but still uses limited subsets unless --full-* is set).'
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode with reduced samples')
    parser.add_argument('--debug-samples', type=int, default=10)
    parser.add_argument('--debug-train-samples', type=int, default=20)
    parser.add_argument('--debug-prototype-samples', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lora-epochs', type=int, default=2, help='Epochs for LoRA-only phase (freeze GNN/prototypes). 0 disables this phase.')
    parser.add_argument('--gnn-epochs', type=int, default=-1, help='Epochs for GNN/prototype phase (freeze LLM). -1 falls back to --epochs.')
    parser.add_argument('--lr-lora', type=float, default=5e-5, help='Learning rate for LoRA phase.')
    parser.add_argument('--lr-gnn', type=float, default=5e-5, help='Learning rate for GNN phase.')
    parser.add_argument('--batch-size', type=int, default=4, help='Pair mini-batch size per document (default doubled from 2 to 4 for better GPU utilization).')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader worker processes per rank.')

    # Subset controls (default is NOT full, even in train/inference)
    parser.add_argument('--train-file', type=str, default='train_annotated.json', help='Path to training JSON (DocRED format).')
    parser.add_argument('--distant-file', type=str, default='train_distant.json', help='Path to distant-supervision JSON.')
    parser.add_argument('--distant-clean-file', type=str, default='train_distant_clean.json', help='Output/consumed cleaned DS file path.')
    parser.add_argument('--build-distant-clean', action='store_true', help='Build cleaned DS file before training.')
    parser.add_argument('--distant-topk', type=int, default=2, help='MIL top-k instances kept per (h,t) bag in DS cleaning.')
    parser.add_argument('--distant-mix-ratio', type=float, default=0.3, help='Ratio of cleaned DS docs mixed into supervised train set.')
    parser.add_argument('--eval-file', type=str, default='dev.json', help='Path to validation/eval JSON (DocRED format).')
    parser.add_argument('--test-file', type=str, default='dev.json', help='Optional path to test JSON. If provided, runs a final test evaluation after training.')
    parser.add_argument('--train-val-ratio', type=float, default=0.8, help='Train split ratio when splitting --train-file into new train/val sets (e.g., 0.8 => 8:2).')
    parser.add_argument('--split-seed', type=int, default=42, help='Random seed for deterministic train/val splitting.')
    parser.add_argument('--eval-from-train', action='store_true', help='Use a validation split taken from --train-file instead of loading --eval-file.')
    parser.add_argument('--eval-offset', type=int, default=-1, help='Start index for the validation split when using --eval-from-train. -1 => use train limit as offset.')
    parser.add_argument('--train-limit', type=int, default=-1, help='Max train docs to use. -1 = stage default. Ignored if --full-train.')
    parser.add_argument('--eval-limit', type=int, default=-1, help='Max eval docs to use. -1 = stage default. Ignored if --full-eval.')
    parser.add_argument('--test-limit', type=int, default=-1, help='Max test docs to use for final evaluation. -1 = use eval limit. Ignored if --full-test.')
    parser.add_argument('--candidate-limit', type=int, default=-1, help='Max docs for candidate stats. -1 = stage default.')
    parser.add_argument('--full-train', action='store_true', help='Use full training set (DISABLED by default).')
    parser.add_argument('--full-eval', action='store_true', help='Use full evaluation set (DISABLED by default).')
    parser.add_argument('--full-test', action='store_true', help='Use full test set for final evaluation.')

    parser.add_argument('--model-id', type=str, default="roberta-large", help='HF model id for the LLM backbone. Encoder backbones (roberta-large/bert) load via AutoModel without quantization; causal LMs (e.g. Qwen/Qwen3-8B) load 4-bit on CUDA.')
    parser.add_argument('--cpu-debug-model', type=str, default="sshleifer/tiny-gpt2", help='Fallback model id when running on CPU in debug/overfit.')
    parser.add_argument('--allow-cpu-large-model', action='store_true', help='Allow loading large models on CPU (may be extremely slow / OOM).')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection. auto uses CUDA only if both torch and DGL support it; otherwise falls back to CPU.')
    
    # MoE hyperparameters
    parser.add_argument('--num-experts', type=int, default=3, help='Number of routed graph experts (ignored if --expert-depths is set).')
    parser.add_argument('--expert-depths', type=str, default="1,2,4", help='Comma-separated graph-layer depths for the heterogeneous routed experts, shallow→deep (e.g. "1,2,4"). Overrides --num-experts. Empty string falls back to homogeneous experts from --num-experts.')
    parser.add_argument('--no-shared-expert', action='store_true', help='Disable the always-on shared/residual expert (default: enabled).')
    parser.add_argument('--capacity-factor', type=float, default=1.25, help='Expert capacity multiplier')
    parser.add_argument('--lambda-moe', type=float, default=0.1, help='Switch load balance loss weight')
    
    # Structural contrastive alignment hyperparameters
    parser.add_argument('--lambda-scl', type=float, default=0.05, help='Structural contrastive loss weight')
    parser.add_argument('--scl-temp', type=float, default=0.1, help='InfoNCE temperature for SCL')

    # Candidate filtering + multi-label threshold
    parser.add_argument('--candidate-keep-ratio', type=float, default=0.3, help='EP-RSR style keep ratio after fast pair filtering.')
    parser.add_argument('--adaptive-threshold-scale', type=float, default=0.6, help='Scale factor applied to base sigmoid threshold (base=0.5) during evaluation.')

    # Classification loss + optimizer
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma.')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha weight for positive labels.')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for ADOPT optimizer.')
    parser.add_argument('--adopt-beta1', type=float, default=0.9, help='ADOPT beta1.')
    parser.add_argument('--adopt-beta2', type=float, default=0.9999, help='ADOPT beta2.')
    parser.add_argument('--adopt-eps', type=float, default=1e-6, help='ADOPT epsilon.')
    parser.add_argument('--neg-pos-ratio', type=float, default=3.0, help='Target negative-to-positive ratio per document for pair training.')
    parser.add_argument('--neg-buffer', type=int, default=5, help='Extra negative pairs sampled per document.')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for F1 improvement before stopping.')
    
    # Memory optimization
    parser.add_argument('--max-pairs-per-doc', type=int, default=25, help='Maximum number of (h, t) pairs per document to process.')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Maximum tokenized sequence length per document/pair view. RoBERTa/BERT cap at 512 (max_position_embeddings=514); raise only for long-context causal LMs.')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0, help='Max grad norm for clipping to prevent exploding gradients.')

    parser.add_argument('--no-pair-markers', action='store_true', help='Disable per-pair inline markers ([E1]/[E2])')
    parser.add_argument('--result-file', type=str, default='result.json', help='Base result filename (suffix). Run name will be prepended.')
    parser.add_argument('--result-dir', type=str, default='inference_results', help='Directory under workspace to store per-run inference outputs.')
    parser.add_argument(
        '--pretrained-gnn-checkpoint',
        type=str,
        default='',
        help='Optional pretrained GNN checkpoint source. Supports local file path, W&B artifact ref (entity/project/name:version), or W&B artifact URL.',
    )
    parser.add_argument('--pretrained-gnn-strict', action='store_true', help='Load pretrained GNN checkpoint with strict key matching.')

    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='', help='Override W&B project name. Empty uses Config.wandb_project.')
    parser.add_argument('--wandb-entity', type=str, default='', help='Optional W&B entity/team. Empty uses default account from login.')
    parser.add_argument('--wandb-api-key', type=str, default='', help='Optional explicit W&B API key. Empty uses WANDB_API_KEY or existing wandb login session.')
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'auto'], help='W&B mode. Default online to sync to cloud unless explicitly set to offline.')
    parser.add_argument('--wandb-no-offline-fallback', action='store_true', help='If set, fail W&B init instead of silently falling back to offline mode when online init fails.')
    parser.add_argument('--skip-wandb-workspace-upload', action='store_true', help='Skip uploading workspace snapshot artifact to W&B.')
    parser.add_argument('--overfit', action='store_true', help='Overfit on 1 sample for debugging')
    parser.add_argument('--overfit-docs', type=int, default=1, help='Docs used for overfit stage/mode (default 1).')
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank used by torchrun/DDP.')
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "-1"))
    local_rank = local_rank_env if local_rank_env >= 0 else args.local_rank

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but torch.cuda is not available")
        if local_rank < 0:
            raise RuntimeError("DDP requires LOCAL_RANK to be set (use torchrun)")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    # Clear GPU memory at startup to avoid conflicts from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    if rank == 0 and (args.debug or args.stage in {'overfit', 'train'}):
        print(f"[ARGS] stage={args.stage} epochs={args.epochs} debug={args.debug} full_train={args.full_train} full_eval={args.full_eval}")

    # Stage defaults: deliberately NOT full by default.
    # Rationale: prevent misleading conclusions and save time/compute.
    STAGE_DEFAULTS = {
        'sanity': {'train': 0, 'eval': 100, 'candidates': 0},
        'candidates': {'train': 0, 'eval': 0, 'candidates': 500},
        'ds': {'train': 0, 'eval': 500, 'candidates': 0},
        'overfit': {'train': args.overfit_docs, 'eval': min(1, args.overfit_docs), 'candidates': 0},
        'train': {'train': 500, 'eval': 200, 'candidates': 0},
    }

    def _limit_or_default(val, default_val):
        return default_val if val is None or val < 0 else val

    def _subset(data, limit, full=False):
        if full:
            return data
        if limit is None:
            return data
        if limit <= 0:
            return []
        return data[:min(limit, len(data))]

    def _subset_range(data, offset, limit, full=False):
        if full:
            return data
        if data is None:
            return None
        if offset is None or offset < 0:
            offset = 0
        if limit is None:
            return data[offset:]
        if limit <= 0:
            return []
        if offset >= len(data):
            return []
        end = min(offset + limit, len(data))
        return data[offset:end]

    def _split_train_val(data, train_ratio=0.8, seed=42):
        """Deterministically split data into train/val while keeping both sides non-empty when possible."""
        n = len(data)
        if n <= 1:
            return data, data[:]

        ratio = float(train_ratio)
        ratio = 0.8 if not math.isfinite(ratio) else max(0.05, min(0.95, ratio))
        train_n = int(n * ratio)
        train_n = max(1, min(n - 1, train_n))

        idx = list(range(n))
        rng = random.Random(int(seed))
        rng.shuffle(idx)

        train_idx = set(idx[:train_n])
        train_split = [data[i] for i in range(n) if i in train_idx]
        val_split = [data[i] for i in range(n) if i not in train_idx]
        return train_split, val_split
    
    with open('rel_info.json', 'r') as f:
        rel_info = json.load(f)
    rel2id = {rel: i for i, rel in enumerate(rel_info.keys())}
    id2rel = {i: rel for rel, i in rel2id.items()}
    num_relations = len(rel2id)

    # Load only what we need per stage (avoid reading full train/dev unnecessarily)
    train_data = None
    distant_data = None
    dev_data = None
    if args.stage in {'sanity', 'candidates', 'ds'}:
        with open(args.eval_file, 'r') as f:
            dev_data = json.load(f)
        if args.stage == 'ds' or args.build_distant_clean:
            distant_data = load_json_robust(args.distant_file)
    else:
        with open(args.train_file, 'r') as f:
            train_data = json.load(f)
        if args.eval_from_train:
            dev_data = train_data
        else:
            with open(args.eval_file, 'r') as f:
                dev_data = json.load(f)
        if args.build_distant_clean:
            distant_data = load_json_robust(args.distant_file)

    test_data = None
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)

    # Stage-specific limits
    defaults = STAGE_DEFAULTS[args.stage]
    train_limit = _limit_or_default(args.train_limit, defaults['train'])
    eval_limit = _limit_or_default(args.eval_limit, defaults['eval'])
    cand_limit = _limit_or_default(args.candidate_limit, defaults['candidates'])
    test_limit = eval_limit if (args.test_limit is None or args.test_limit < 0) else args.test_limit

    eval_offset = args.eval_offset
    if eval_offset is None or eval_offset < 0:
        eval_offset = train_limit if (train_limit is not None and train_limit > 0) else 0

    if args.debug:
        # Debug flag always forces small sizes
        train_limit = min(train_limit if train_limit > 0 else args.debug_train_samples, args.debug_train_samples)
        eval_limit = min(eval_limit if eval_limit > 0 else args.debug_samples, args.debug_samples)
        cand_limit = min(cand_limit if cand_limit > 0 else 100, 100)
        test_limit = min(test_limit if test_limit > 0 else args.debug_samples, args.debug_samples)
        if args.eval_from_train:
            eval_offset = min(eval_offset, max(0, len(train_data) - 1)) if train_data is not None else eval_offset

    # -------- Stage 1: Data sanity & label validation (NO LLM) --------
    if args.stage == 'sanity':
        eval_subset = _subset(dev_data, eval_limit, full=args.full_eval)
        print(f"[SANITY] Using {len(eval_subset)}/{len(dev_data)} dev docs")
        no_pos = 0
        bad_mentions = 0
        rel_outside = 0
        for item in eval_subset:
            labels = item.get('labels', [])
            if len(labels) == 0:
                no_pos += 1
            for lbl in labels:
                if lbl.get('r') not in rel2id:
                    rel_outside += 1
            sents = item.get('sents', [])
            sent_lens = [len(s) for s in sents]
            vertex_set = item.get('vertex_set', item.get('vertexSet', []))
            for ent in vertex_set:
                for m in ent:
                    sid = int(m.get('sent_id', 0))
                    start, end = int(m['pos'][0]), int(m['pos'][1])
                    if sid < 0 or sid >= len(sent_lens):
                        bad_mentions += 1
                        continue
                    if not (0 <= start < end <= sent_lens[sid]):
                        bad_mentions += 1
        print(f"[SANITY] Docs with 0 positives: {no_pos}/{len(eval_subset)}")
        print(f"[SANITY] Mentions with invalid span/sent_id: {bad_mentions}")
        print(f"[SANITY] Labels with unknown relation: {rel_outside}")
        return

    # -------- Stage 2: Candidate generation / pruning stats (NO LLM) --------
    if args.stage == 'candidates':
        subset = _subset(dev_data, cand_limit, full=args.full_eval)
        print(f"[CANDIDATES] Using {len(subset)}/{len(dev_data)} dev docs")
        candidate_gen = CandidateGenerator()
        total_cands = 0
        total_gold = 0
        total_gold_kept = 0
        total_pref_kept = 0
        per_doc = []
        for item in subset:
            vertex_set = item.get('vertex_set', item.get('vertexSet', []))
            cands = candidate_gen.generate_candidates(item, vertex_set)
            gold_pairs = {(lbl['h'], lbl['t']) for lbl in item.get('labels', [])}
            kept = sum(1 for p in gold_pairs if p in set(cands))
            pref = candidate_gen.prefilter_candidates(
                item,
                vertex_set,
                cands,
                keep_ratio=args.candidate_keep_ratio,
                must_keep=gold_pairs,
            )
            pref = limit_candidates_preserve_must_keep(pref, gold_pairs, args.max_pairs_per_doc)
            pref_kept = sum(1 for p in gold_pairs if p in set(pref))
            total_cands += len(cands)
            total_gold += len(gold_pairs)
            total_gold_kept += kept
            total_pref_kept += pref_kept
            per_doc.append((len(cands), len(pref), len(gold_pairs), pref_kept))
        avg_c = total_cands / max(1, len(subset))
        recall = total_gold_kept / max(1, total_gold)
        pref_recall = total_pref_kept / max(1, total_gold)
        print(f"[CANDIDATES] Avg candidates/doc: {avg_c:.2f}")
        print(f"[CANDIDATES] Gold kept: {total_gold_kept}/{total_gold} => recall={recall*100:.2f}%")
        print(f"[CANDIDATES] Gold kept after prefilter+cap: {total_pref_kept}/{total_gold} => recall={pref_recall*100:.2f}%")
        worst = sorted(per_doc, key=lambda x: (0 if x[2] == 0 else x[3] / x[2]))[:5]
        if worst:
            print(f"[CANDIDATES] Worst (raw,pref,gold,kept_after_pref): {worst}")
        return

    # -------- Stage DS: build cleaned distant supervision set --------
    if args.stage == 'ds':
        if distant_data is None:
            distant_data = load_json_robust(args.distant_file)
        cleaned_ds, ds_stats = clean_distant_supervision_data(
            distant_data,
            _subset(dev_data, eval_limit, full=args.full_eval),
            rel2id,
            top_k=args.distant_topk,
        )
        with open(args.distant_clean_file, 'w') as f:
            json.dump(cleaned_ds, f)
        print(f"[DS] Input docs: {ds_stats['input_docs']} | Kept docs: {ds_stats['kept_docs']}")
        print(
            f"[DS] Dropped unknown_rel={ds_stats['dropped_unknown_rel']} "
            f"bad_index={ds_stats['dropped_bad_index']} type={ds_stats['dropped_type']} mil={ds_stats['dropped_by_mil']}"
        )
        print(f"[DS] Wrote cleaned DS file: {args.distant_clean_file}")
        return

    # Remaining stages require model/LLM. Encoder backbones (RoBERTa/BERT) only need
    # transformers; bitsandbytes is optional and used solely for 4-bit causal-LM loading.
    if AutoTokenizer is None or AutoModel is None or AutoModelForCausalLM is None or AutoConfig is None:
        raise ImportError(
            "transformers is required for overfit/train stages. "
            "Install with: pip install transformers accelerate (and bitsandbytes for 4-bit causal LMs)"
        )
    peft_available = not (get_peft_model is None or LoraConfig is None or TaskType is None)
    if (not peft_available) and rank == 0:
        print("[WARN] peft is unavailable; LoRA phase will be disabled and training will continue without LoRA.")
    if dgl is None:
        details = f" (import error: {DGL_IMPORT_ERROR})" if DGL_IMPORT_ERROR is not None else ""
        raise ImportError(
            "DGL is required for stages overfit/train. "
            "If you already installed dgl and still fail, run build_dgl_cuda128.sh to patch GraphBolt "
            "for missing optional GraphBolt deps (for example torchdata.datapipes or pandas)."
            f"{details}"
        )

    def _dgl_cuda_enabled():
        if not torch.cuda.is_available():
            return False
        try:
            g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
            g = g.to('cuda')
            return True
        except Exception:
            return False

    dgl_cuda_ok = _dgl_cuda_enabled()

    if args.device == 'cpu':
        DEVICE = 'cpu'
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda is not available")
        DEVICE = f'cuda:{local_rank}' if distributed else 'cuda:0'
        if not dgl_cuda_ok:
            if rank == 0:
                print("[WARN] DGL CUDA backend not available. Running GraphExpert on CPU and keeping LLM/router/classifier on CUDA.")
    else:
        # auto
        if torch.cuda.is_available():
            DEVICE = f'cuda:{local_rank}' if distributed else 'cuda:0'
        else:
            DEVICE = 'cpu'
        if DEVICE.startswith('cuda') and not dgl_cuda_ok:
            if rank == 0:
                print("[WARN] Auto-selected CUDA but DGL CUDA backend is unavailable. Using hybrid mode (DGL on CPU).")

    use_cuda = DEVICE.startswith('cuda')
    graph_device = 'cuda' if (use_cuda and dgl_cuda_ok) else 'cpu'
    MODEL_ID = args.model_id
    if DEVICE == "cpu" and (args.debug or args.stage == 'overfit') and not args.allow_cpu_large_model:
        MODEL_ID = args.cpu_debug_model
        if rank == 0:
            print(f"[INFO] CPU + debug/overfit detected, using smaller model: {MODEL_ID}")
    if DEVICE == "cpu" and (not args.allow_cpu_large_model) and ("4B" in str(MODEL_ID) or "Qwen" in str(MODEL_ID)):
        if rank == 0:
            print("[WARN] Large model on CPU may be extremely slow/OOM. Use --debug (auto tiny model) or set --model-id to a small model, or pass --allow-cpu-large-model.")

    if rank == 0:
        print(f"Using device: {DEVICE}")
    if use_cuda:
        gpu_idx = local_rank if distributed else 0
        if rank == 0:
            print(f"GPU: {torch.cuda.get_device_name(gpu_idx)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.2f} GB")

    if distributed and rank == 0:
        print(f"[DDP] Enabled with world_size={world_size}, local_rank={local_rank}")

    wandb_project = str(args.wandb_project).strip() if str(args.wandb_project).strip() else Config.wandb_project
    wandb_entity = str(args.wandb_entity).strip() if str(args.wandb_entity).strip() else None
    wandb_api_key_arg = str(args.wandb_api_key).strip()
    wandb_api_key_env = str(os.environ.get("WANDB_API_KEY", "")).strip()
    wandb_api_key_cfg = str(getattr(Config, "wandb_key", "")).strip()
    if wandb_api_key_arg:
        wandb_api_key = wandb_api_key_arg
    elif wandb_api_key_env:
        wandb_api_key = wandb_api_key_env
    else:
        wandb_api_key = wandb_api_key_cfg

    run_name = Config.get_unique_run_name(f"moe_{args.stage}")
    use_wandb = (not args.no_wandb) and (rank == 0)
    if use_wandb:
        if wandb_project:
            os.environ["WANDB_PROJECT"] = wandb_project
        allow_offline_fallback = (args.wandb_mode != 'online') and (not args.wandb_no_offline_fallback)
        use_wandb = bool(
            init_wandb_run(
                run_name,
                vars(args),
                project=wandb_project,
                entity=wandb_entity,
                api_key=wandb_api_key,
                mode=args.wandb_mode,
                allow_offline_fallback=allow_offline_fallback,
            )
        )
        if (not use_wandb) and rank == 0:
            print("[W&B] Initialization failed. Training continues without W&B logging.")
        if wandb is not None and getattr(wandb, "run", None) is not None and getattr(wandb.run, "name", None):
            run_name = str(wandb.run.name)

    workspace_root_abs = os.path.dirname(os.path.abspath(__file__))
    safe_run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_name))
    result_dir_abs = os.path.abspath(os.path.join(workspace_root_abs, args.result_dir))
    os.makedirs(result_dir_abs, exist_ok=True)
    result_base_name = os.path.basename(args.result_file) if args.result_file else "result.json"
    run_result_path = os.path.join(result_dir_abs, f"{safe_run_name}_{result_base_name}")
    checkpoints_root_abs = os.path.abspath(os.path.join(workspace_root_abs, "checkpoints"))
    os.makedirs(checkpoints_root_abs, exist_ok=True)
    ckpt_dir_abs = os.path.join(checkpoints_root_abs, safe_run_name)
    os.makedirs(ckpt_dir_abs, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir_abs, "best_model.pt")
    best_infer_ckpt_path = os.path.join(ckpt_dir_abs, "best_for_inference.pt")
    best_ckpt_result_path = os.path.join(result_dir_abs, f"{safe_run_name}_best_ckpt_result.json")
    result_evaluation_path = os.path.join(workspace_root_abs, "result_evalutation.txt")

    def _prune_checkpoint_dirs(root_dir, keep_dir_name):
        """Keep only the latest run checkpoint directory on server workspace."""
        if not root_dir or (not os.path.isdir(root_dir)):
            return
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if not os.path.isdir(full):
                continue
            if name == keep_dir_name:
                continue
            try:
                shutil.rmtree(full)
                if rank == 0:
                    print(f"[CKPT] Removed old checkpoint dir: {full}")
            except Exception as e:
                if rank == 0:
                    print(f"[WARN] Failed to remove old checkpoint dir {full}: {e}")

    _prune_checkpoint_dirs(checkpoints_root_abs, safe_run_name)

    if rank == 0:
        print(f"[RESULT] Inference outputs will be saved to: {run_result_path}")
        print(f"[RESULT] Best-checkpoint inference output: {best_ckpt_result_path}")
        print(f"[RESULT] Epoch evaluation log: {result_evaluation_path}")
        print(f"[CKPT] Best checkpoint path: {best_ckpt_path}")
        print(f"[CKPT] Best checkpoint for inference path: {best_infer_ckpt_path}")

    def _upload_inference_json_to_wandb(path, artifact_name):
        if not use_wandb:
            return
        if not path or (not os.path.exists(path)):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                results_payload = json.load(f)
            ok = save_evaluation_artifact(results_payload, path, artifact_name=artifact_name)
            if not ok:
                print(f"[WARN] W&B run is unavailable; cannot upload evaluation artifact: {artifact_name}")
        except Exception as e:
            # Fallback keeps artifact upload robust even if JSON re-serialization fails.
            print(f"[WARN] save_evaluation_artifact failed for {path}: {e}. Falling back to file artifact upload.")
            ok = save_model_artifact(path, name=artifact_name, artifact_type="evaluation")
            if not ok:
                print(f"[WARN] W&B run is unavailable; cannot upload fallback artifact: {artifact_name}")

    def _upload_inference_results_dir_to_wandb(results_dir, artifact_name):
        """Upload all prediction files in inference_results folder for this run."""
        if not use_wandb:
            return
        if not results_dir or (not os.path.isdir(results_dir)):
            return
        try:
            ok = save_model_artifact(results_dir, name=artifact_name, artifact_type="evaluation")
            if not ok:
                print(f"[WARN] W&B run is unavailable; cannot upload inference_results artifact: {artifact_name}")
                return
            json_count = 0
            for fname in os.listdir(results_dir):
                if fname.endswith('.json') and os.path.isfile(os.path.join(results_dir, fname)):
                    json_count += 1
            print(f"[W&B] Uploaded inference_results artifact: {artifact_name} ({json_count} json files)")
        except Exception as e:
            print(f"[WARN] Failed to upload inference_results folder: {e}")

    def _upload_workspace_snapshot_to_wandb(root_dir, artifact_name):
        if not use_wandb:
            return
        if not root_dir or (not os.path.isdir(root_dir)):
            return
        try:
            added_files = 0
            has_moe_py = False
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    if not os.path.isfile(fpath):
                        continue
                    relpath = os.path.relpath(fpath, root_dir).replace("\\", "/")
                    if relpath == "moe.py":
                        has_moe_py = True
                    added_files += 1
            if added_files == 0:
                print("[WARN] Workspace artifact has no files; skipping upload")
                return
            ok = save_model_artifact(root_dir, name=artifact_name, artifact_type="workspace")
            if not ok:
                print(f"[WARN] W&B run is unavailable; cannot upload workspace artifact: {artifact_name}")
                return
            print(f"[W&B] Uploaded workspace artifact: {artifact_name} ({added_files} files, contains_moe_py={has_moe_py})")
        except Exception as e:
            print(f"[WARN] Failed to upload workspace snapshot to W&B: {e}")

    def _upload_single_file_to_wandb(path, artifact_name, artifact_type="source"):
        if not use_wandb:
            return
        if not path or (not os.path.isfile(path)):
            return
        ok = save_model_artifact(path, name=artifact_name, artifact_type=artifact_type)
        if not ok:
            print(f"[WARN] W&B run is unavailable; cannot upload file artifact: {artifact_name}")

    def _get_wandb_sdk():
        wb = None
        if _wandb_utils_module is not None:
            wb = getattr(_wandb_utils_module, "wandb", None)
        if wb is None:
            wb = wandb
        return wb

    def _normalize_wandb_artifact_ref(raw_ref):
        ref = str(raw_ref).strip()
        if not ref:
            return "", None
        ref = ref.replace("wandb-artifact://", "").replace("wandb://", "")

        # URL format example:
        # https://wandb.ai/<entity>/<project>/artifacts/<type>/<artifact_name>/<version>
        if ref.startswith("http://") or ref.startswith("https://"):
            parsed = urlparse(ref)
            path = parsed.path or ""
            m = re.search(r"/([^/]+)/([^/]+)/artifacts/([^/]+)/([^/]+)/([^/]+)$", path)
            if m:
                entity, project, artifact_type, artifact_name, version = m.groups()
                return f"{entity}/{project}/{artifact_name}:{version}", artifact_type
            m_latest = re.search(r"/([^/]+)/([^/]+)/artifacts/([^/]+)/([^/]+)/?$", path)
            if m_latest:
                entity, project, artifact_type, artifact_name = m_latest.groups()
                return f"{entity}/{project}/{artifact_name}:latest", artifact_type
            raise ValueError(
                "Unsupported W&B artifact URL format. Expected .../artifacts/<type>/<name>/<version>."
            )

        # Already artifact ref (entity/project/name:alias).
        return ref, None

    def _pick_checkpoint_file_from_dir(root_dir):
        candidates = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith((".pt", ".pth", ".ckpt", ".bin")):
                    candidates.append(os.path.join(dirpath, fname))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-like file found under {root_dir}")

        preferred_order = ["best_for_inference.pt", "best_model.pt", "model.pt", "checkpoint.pt"]
        preferred_rank = {name: idx for idx, name in enumerate(preferred_order)}

        def _rank(path):
            base = os.path.basename(path)
            hit = 0 if base in preferred_rank else 1
            order = preferred_rank.get(base, 999)
            return (hit, order, len(path))

        candidates.sort(key=_rank)
        return candidates[0]

    def _resolve_checkpoint_source(source):
        src = str(source).strip()
        if not src:
            return None

        if os.path.isfile(src):
            return os.path.abspath(src)

        wb = _get_wandb_sdk()
        if wb is None:
            raise RuntimeError("wandb SDK is unavailable; cannot resolve W&B checkpoint source")

        ref, artifact_type = _normalize_wandb_artifact_ref(src)
        if not ref:
            raise ValueError("Empty W&B artifact reference")

        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = str(wandb_api_key)
            try:
                if hasattr(wb, "login"):
                    wb.login(key=str(wandb_api_key), relogin=True)
            except Exception as e:
                print(f"[WARN] wandb login for pretrained artifact failed, continuing with current session: {e}")

        artifact = None
        run_obj = getattr(wb, "run", None)
        if run_obj is not None and hasattr(run_obj, "use_artifact"):
            try:
                artifact = run_obj.use_artifact(ref, type=artifact_type) if artifact_type else run_obj.use_artifact(ref)
            except Exception:
                artifact = None

        if artifact is None:
            if not hasattr(wb, "Api"):
                raise RuntimeError("wandb.Api is unavailable for artifact download")
            api = wb.Api(api_key=str(wandb_api_key).strip() or None)
            artifact = api.artifact(ref, type=artifact_type) if artifact_type else api.artifact(ref)

        cache_root = os.path.join(workspace_root_abs, "tmp", "wandb_pretrained")
        os.makedirs(cache_root, exist_ok=True)
        artifact_dir = artifact.download(root=cache_root)
        ckpt_path = _pick_checkpoint_file_from_dir(artifact_dir)
        if rank == 0:
            print(f"[PRETRAIN] Resolved W&B artifact {ref} -> {ckpt_path}")
        return ckpt_path

    if rank == 0:
        with open(result_evaluation_path, 'w', encoding='utf-8') as f:
            f.write(f"run_name={run_name}\n")
            f.write(f"created_at={datetime.utcnow().isoformat()}Z\n")
            f.write("# per-epoch evaluation metrics (val/test)\n")

    def _print_official_metrics(prefix, metrics):
        if metrics is None:
            return
        print(
            f"[{prefix}-OFFICIAL] "
            f"RE(P/R/F1)={metrics.get('RE_P', 0.0):.4f}/{metrics.get('RE_R', 0.0):.4f}/{metrics.get('F1-RE', 0.0):.4f} "
            f"Evidence(P/R/F1)={metrics.get('Evidence_P', 0.0):.4f}/{metrics.get('Evidence_R', 0.0):.4f}/{metrics.get('F1-Evidence', 0.0):.4f} "
            f"RE_ignore_annotated(P/R/F1)={metrics.get('RE_ignore_annotated_P', 0.0):.4f}/{metrics.get('RE_ignore_annotated_R', 0.0):.4f}/{metrics.get('RE_ignore_annotated_F1', 0.0):.4f} "
            f"RE_ignore_distant(P/R/F1)={metrics.get('RE_ignore_distant_P', 0.0):.4f}/{metrics.get('RE_ignore_distant_R', 0.0):.4f}/{metrics.get('RE_ignore_distant_F1', 0.0):.4f}"
        )

    def _append_eval_result_line(split_name, epoch_idx, metrics, train_loss=None):
        if rank != 0:
            return
        if metrics is None:
            return
        with open(result_evaluation_path, 'a', encoding='utf-8') as f:
            f.write(
                f"epoch={epoch_idx} split={split_name} "
                f"train_loss={float(train_loss) if train_loss is not None else -1.0:.6f} "
                f"fact_P={metrics.get('precision', 0.0):.6f} "
                f"fact_R={metrics.get('recall', 0.0):.6f} "
                f"fact_F1={metrics.get('f1', 0.0):.6f} "
                f"RE_P={metrics.get('RE_P', 0.0):.6f} "
                f"RE_R={metrics.get('RE_R', 0.0):.6f} "
                f"F1-RE={metrics.get('F1-RE', 0.0):.6f} "
                f"Evidence_P={metrics.get('Evidence_P', 0.0):.6f} "
                f"Evidence_R={metrics.get('Evidence_R', 0.0):.6f} "
                f"F1-Evidence={metrics.get('F1-Evidence', 0.0):.6f} "
                f"RE_ignore_annotated_P={metrics.get('RE_ignore_annotated_P', 0.0):.6f} "
                f"RE_ignore_annotated_R={metrics.get('RE_ignore_annotated_R', 0.0):.6f} "
                f"RE_ignore_annotated_F1={metrics.get('RE_ignore_annotated_F1', 0.0):.6f} "
                f"RE_ignore_distant_P={metrics.get('RE_ignore_distant_P', 0.0):.6f} "
                f"RE_ignore_distant_R={metrics.get('RE_ignore_distant_R', 0.0):.6f} "
                f"RE_ignore_distant_F1={metrics.get('RE_ignore_distant_F1', 0.0):.6f}\n"
            )

    def _sync_eval_log_to_wandb():
        if rank != 0:
            return
        if not use_wandb:
            return
        _upload_single_file_to_wandb(
            result_evaluation_path,
            artifact_name=f"{safe_run_name}_result_evalutation",
            artifact_type="evaluation",
        )

    if use_wandb and (not args.skip_wandb_workspace_upload):
        _upload_workspace_snapshot_to_wandb(workspace_root_abs, artifact_name=f"{safe_run_name}_workspace")
    if use_wandb:
        _upload_single_file_to_wandb(os.path.join(workspace_root_abs, "moe.py"), artifact_name=f"{safe_run_name}_moe_py", artifact_type="source")

    try:
        tokenizer = _load_with_cache_retry(
            AutoTokenizer,
            MODEL_ID,
            {"cache_dir": os.environ.get("HF_HOME"), "add_prefix_space": True},
            artifact_name="Tokenizer",
        )
    except Exception:
        tokenizer = _load_with_cache_retry(
            AutoTokenizer,
            MODEL_ID,
            {"cache_dir": os.environ.get("HF_HOME")},
            artifact_name="Tokenizer",
        )
    tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
    # Encoder backbones (RoBERTa/BERT) already define <pad>; only causal LMs (Qwen) lack one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_pair_markers = not args.no_pair_markers
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    if use_pair_markers and (e1_id is None or e2_id is None or e1_id == tokenizer.unk_token_id or e2_id == tokenizer.unk_token_id):
        print("[WARN] Pair markers requested but special tokens not found; disabling markers.")
        use_pair_markers = False

    # Detect encoder backbones (RoBERTa/BERT/DeBERTa/ELECTRA): they are bidirectional
    # encoders, fit easily in memory (no 4-bit needed), and must use AutoModel — not
    # AutoModelForCausalLM, which would wrongly attach a causal mask / LM head.
    def _is_encoder_backbone(model_id):
        try:
            cfg = AutoConfig.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"))
        except Exception:
            mt = str(model_id).lower()
            return any(k in mt for k in ("roberta", "bert", "deberta", "electra"))
        if getattr(cfg, "is_decoder", False) or getattr(cfg, "is_encoder_decoder", False):
            return False
        return str(getattr(cfg, "model_type", "")).lower() in (
            "roberta", "bert", "deberta", "deberta-v2", "electra", "xlm-roberta",
        )

    is_encoder = _is_encoder_backbone(MODEL_ID)

    # LLM: Frozen backbone + LoRA fine-tuning (4-bit quantization on CUDA only for large causal LMs)
    def _load_base_model(model_id):
        if is_encoder:
            # RoBERTa-large (~355M) fits easily in fp32 on a single GPU; fp32 avoids the
            # training instability of pure-fp16 without an autocast scaler.
            print("Loading encoder LLM (RoBERTa/BERT-style, fp32, no quantization)...")
            m = _load_with_cache_retry(
                AutoModel,
                model_id,
                {
                    "cache_dir": os.environ.get("HF_HOME"),
                    "torch_dtype": torch.float32,
                },
                artifact_name="Model",
            )
            return m.to(DEVICE)

        print("Loading causal LLM..." + (" (4-bit quantization)" if use_cuda else " (CPU fp32)"))
        if use_cuda:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes is required for 4-bit causal-LM loading; install it or use an encoder backbone like roberta-large.")
            # Clear any residual GPU memory before loading model
            torch.cuda.empty_cache()
            return _load_with_cache_retry(
                AutoModelForCausalLM,
                model_id,
                {
                    "cache_dir": os.environ.get("HF_HOME"),
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_enable_fp32_cpu_offload=True
                    ),
                    "device_map": ({"": local_rank} if distributed else "auto"),
                    "max_memory": ({local_rank: "30GB"} if distributed else {0: "30GB"}),
                },
                artifact_name="Model",
            )

        m = _load_with_cache_retry(
            AutoModelForCausalLM,
            model_id,
            {
                "cache_dir": os.environ.get("HF_HOME"),
                "torch_dtype": torch.float32,
            },
            artifact_name="Model",
        )
        return m.to(DEVICE)

    try:
        base_model = _load_base_model(MODEL_ID)
    except Exception as e:
        msg = str(e)
        unsupported_arch = ("does not recognize this architecture" in msg) or ("model type" in msg and "does not recognize" in msg)
        if unsupported_arch and (str(MODEL_ID) != str(args.cpu_debug_model)):
            fallback_model = args.cpu_debug_model
            if rank == 0:
                print(f"[WARN] Model architecture for '{MODEL_ID}' is unsupported by installed transformers. Falling back to '{fallback_model}'.")
            MODEL_ID = fallback_model
            tokenizer = _load_with_cache_retry(
                AutoTokenizer,
                MODEL_ID,
                {"cache_dir": os.environ.get("HF_HOME")},
                artifact_name="Tokenizer",
            )
            tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            e1_id = tokenizer.convert_tokens_to_ids("[E1]")
            e2_id = tokenizer.convert_tokens_to_ids("[E2]")
            if use_pair_markers and (e1_id is None or e2_id is None or e1_id == tokenizer.unk_token_id or e2_id == tokenizer.unk_token_id):
                if rank == 0:
                    print("[WARN] Pair markers unavailable after tokenizer fallback; disabling markers.")
                use_pair_markers = False
            base_model = _load_base_model(MODEL_ID)
        else:
            raise
    
    base_model.resize_token_embeddings(len(tokenizer))
    
    # LoRA: Parameter-efficient fine-tuning on attention weights
    # Choose target modules based on backbone architecture.
    module_names = [name for name, _ in base_model.named_modules()]
    if any(n.endswith('q_proj') for n in module_names) and any(n.endswith('v_proj') for n in module_names):
        target_modules = ["q_proj", "v_proj"]  # Qwen / LLaMA-style causal LMs
    elif any(n.endswith('attention.self.query') for n in module_names) and any(n.endswith('attention.self.value') for n in module_names):
        target_modules = ["query", "value"]  # RoBERTa / BERT-style encoders
    elif any(n.endswith('c_attn') for n in module_names):
        target_modules = ["c_attn"]  # GPT-2-style
    else:
        target_modules = []

    lora_enabled = bool(target_modules) and peft_available
    if lora_enabled:
        lora_model = get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=32,
                target_modules=target_modules,
            ),
        )
    else:
        if rank == 0:
            if not target_modules:
                print("[WARN] No known LoRA target modules found; proceeding without LoRA.")
            else:
                print("[WARN] LoRA target modules found but peft is unavailable; proceeding without LoRA.")
        lora_model = base_model
    
    # MoE Graph RE Model with Sparse Experts
    # Parse heterogeneous expert depths (shallow→deep). Empty → homogeneous from --num-experts.
    _depths_str = (args.expert_depths or "").strip()
    if _depths_str:
        try:
            expert_depths = [max(1, int(x)) for x in _depths_str.split(",") if x.strip() != ""]
        except ValueError:
            if rank == 0:
                print(f"[WARN] Could not parse --expert-depths='{args.expert_depths}'; falling back to --num-experts={args.num_experts}.")
            expert_depths = None
    else:
        expert_depths = None

    model = MoEGraphRE(
        lora_model,
        num_relations,
        num_experts=args.num_experts,
        capacity_factor=args.capacity_factor,
        graph_device=graph_device,
        expert_depths=expert_depths,
        use_shared_expert=(not args.no_shared_expert),
    ).to(DEVICE)

    if graph_device == 'cpu':
        model.experts = model.experts.to('cpu')
        if model.shared_expert is not None:
            model.shared_expert = model.shared_expert.to('cpu')

    # Learnable relation prototypes for structural contrastive alignment.
    prototype_dim = model.expert_dim * 2
    prototypes = RelationPrototype(num_relations, prototype_dim).to(DEVICE)

    def _load_checkpoint_into_modules(checkpoint_path, strict=False):
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

        return loaded_any

    pretrained_source = str(args.pretrained_gnn_checkpoint).strip()
    if pretrained_source and rank == 0:
        resolved_path = _resolve_checkpoint_source(pretrained_source)
        if not resolved_path or (not os.path.isfile(resolved_path)):
            raise FileNotFoundError(f"[PRETRAIN] Checkpoint source not found: {pretrained_source}")
        loaded = _load_checkpoint_into_modules(
            resolved_path,
            strict=bool(args.pretrained_gnn_strict),
        )
        if not loaded:
            raise RuntimeError(
                f"[PRETRAIN] Loaded checkpoint but no compatible keys were applied: {resolved_path}"
            )
        print(f"[PRETRAIN] Loaded pretrained GNN checkpoint from: {resolved_path}")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    if distributed:
        prototypes = DDP(
            prototypes,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    # Verify all modules are on CPU
    if rank == 0:
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Prototypes device: {next(prototypes.parameters()).device}")
    
    graph_builder = DocREDGraphBuilder(graph_device)
    candidate_gen = CandidateGenerator()
    def _set_module_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _configure_training_phase(phase_name):
        """Set trainable parameters by phase and create matching optimizer."""
        core = _unwrap_model(model)
        proto_core = _unwrap_model(prototypes)

        if phase_name == 'lora':
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
                print(f"[PHASE] GNN phase active | lr={phase_lr}")

        trainable = [p for p in list(model.parameters()) + list(prototypes.parameters()) if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable parameters found for phase={phase_name}")
        if ADOPTOptimizer is None:
            raise RuntimeError(f"ADOPT optimizer is unavailable from adopt-main/src: {ADOPT_IMPORT_ERROR}")
        return ADOPTOptimizer(
            trainable,
            lr=phase_lr,
            betas=(float(args.adopt_beta1), float(args.adopt_beta2)),
            eps=float(args.adopt_eps),
            weight_decay=float(args.weight_decay),
            decouple=True,
        )

    # Subset selection (NEVER full unless explicitly requested)
    overfit_stage = (args.stage == 'overfit')
    overfit_one_doc = bool(args.overfit)

    # For train/overfit, always build a fresh 8:2 train/val split from --train-file.
    if train_data is not None:
        split_train_data, split_val_data = _split_train_val(
            train_data,
            train_ratio=args.train_val_ratio,
            seed=args.split_seed,
        )
    else:
        split_train_data, split_val_data = None, None

    train_pool = split_train_data if split_train_data is not None else train_data
    val_pool = split_val_data if split_val_data is not None else dev_data

    train_data_subset = _subset(train_pool, (1 if overfit_one_doc else train_limit), full=args.full_train)
    dev_data_subset = _subset(val_pool, (1 if overfit_one_doc else eval_limit), full=args.full_eval)

    test_data_subset = None
    if test_data is not None:
        test_data_subset = _subset(test_data, (1 if overfit_one_doc else test_limit), full=args.full_test)

    if args.build_distant_clean and distant_data is not None:
        cleaned_ds, ds_stats = clean_distant_supervision_data(
            distant_data,
            dev_data_subset if dev_data_subset is not None else dev_data,
            rel2id,
            top_k=args.distant_topk,
        )
        with open(args.distant_clean_file, 'w') as f:
            json.dump(cleaned_ds, f)
        if rank == 0:
            print(f"[DS] Built cleaned DS: {len(cleaned_ds)} docs -> {args.distant_clean_file}")
            print(
                f"[DS] Drop stats unknown_rel={ds_stats['dropped_unknown_rel']} bad_index={ds_stats['dropped_bad_index']} "
                f"type={ds_stats['dropped_type']} mil={ds_stats['dropped_by_mil']}"
            )

    if args.distant_mix_ratio > 0.0 and (not os.path.exists(args.distant_clean_file)):
        if distant_data is None:
            distant_data = load_json_robust(args.distant_file)
        cleaned_ds, ds_stats = clean_distant_supervision_data(
            distant_data,
            dev_data_subset if dev_data_subset is not None else dev_data,
            rel2id,
            top_k=args.distant_topk,
        )
        with open(args.distant_clean_file, 'w') as f:
            json.dump(cleaned_ds, f)
        if rank == 0:
            print(f"[DS] Auto-built cleaned DS (missing file): {len(cleaned_ds)} docs -> {args.distant_clean_file}")

    if args.distant_mix_ratio > 0.0 and os.path.exists(args.distant_clean_file):
        try:
            distant_clean = load_json_robust(args.distant_clean_file)
            mix_n = max(1, int(len(train_data_subset) * float(args.distant_mix_ratio)))
            train_data_subset = train_data_subset + _subset(distant_clean, mix_n, full=False)
            if rank == 0:
                print(f"[DS] Mixed {min(mix_n, len(distant_clean))} cleaned DS docs into supervised training set")
        except Exception as e:
            if rank == 0:
                print(f"[WARN] Failed to mix cleaned DS data: {e}")

    if rank == 0:
        if split_train_data is not None:
            print(
                f"[SPLIT] train_file split ratio={args.train_val_ratio:.2f} seed={args.split_seed} "
                f"-> train={len(split_train_data)} val={len(split_val_data)}"
            )
        print(f"[DATA] Train docs: {len(train_data_subset)}/{len(train_pool)} (full={args.full_train})")
        print(f"[DATA] Eval  docs: {len(dev_data_subset)}/{len(val_pool)} (new val split, full={args.full_eval})")
        if test_data_subset is not None:
            print(f"[DATA] Test docs: {len(test_data_subset)}/{len(test_data)} (full={args.full_test})")

    metrics_train_data = split_train_data if split_train_data is not None else train_data_subset
    train_facts_annotated = _build_entity_relation_facts(metrics_train_data)
    train_facts_distant = set()
    if args.distant_file and os.path.exists(args.distant_file):
        try:
            distant_for_metrics = load_json_robust(args.distant_file)
            train_facts_distant = _build_entity_relation_facts(distant_for_metrics)
        except Exception as e:
            if rank == 0:
                print(f"[WARN] Failed to load distant facts for ignore-distant metric: {e}")

    train_rel_freq = compute_relation_frequency(train_data_subset, rel2id)
    tail_buckets = build_tail_buckets(train_rel_freq)
    if rank == 0:
        non_zero = int(np.sum(train_rel_freq > 0))
        print(
            f"[LONG-TAIL] Seen relations in train: {non_zero}/{len(rel2id)} "
            f"| buckets head={len(tail_buckets['head'])} medium={len(tail_buckets['medium'])} tail={len(tail_buckets['tail'])}"
        )

    if overfit_one_doc or overfit_stage:
        # Overfit is a logic check: typically needs more epochs, but don't override user choice.
        if args.epochs < 50 and rank == 0:
            print("[INFO] Overfit stage often needs many epochs; consider --epochs 200 (or higher)")
        args.lr_lora = 1e-4
        args.lr_gnn = 1e-4

    train_dataset = DocREDDataset(train_data_subset)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_collate = functools.partial(collate_fn, tokenizer=tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=use_cuda,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=train_collate,
    )
    
    # Memory optimization: limit max pairs per document
    max_pairs = args.max_pairs_per_doc
    
    lora_epochs = max(0, int(args.lora_epochs))
    if not lora_enabled and lora_epochs > 0:
        if rank == 0:
            print(f"[SCHEDULE] Disabling LoRA phase (requested {lora_epochs} epochs) because LoRA is unavailable.")
        lora_epochs = 0
    gnn_epochs = int(args.gnn_epochs) if int(args.gnn_epochs) >= 0 else int(args.epochs)
    if lora_epochs == 0 and gnn_epochs <= 0:
        gnn_epochs = int(args.epochs)
    total_epochs = lora_epochs + max(0, gnn_epochs)
    if total_epochs <= 0:
        raise ValueError("Total epochs must be > 0. Check --lora-epochs/--gnn-epochs/--epochs.")

    if rank == 0:
        print(f"[SCHEDULE] LoRA epochs={lora_epochs} | GNN epochs={max(0, gnn_epochs)} | Total={total_epochs}")

    phase = 'lora' if lora_epochs > 0 else 'gnn'
    optimizer = _configure_training_phase(phase)

    best_f1 = -1
    epochs_no_improve = 0
    best_epoch = -1
    best_test_f1 = None
    best_test_metrics = None

    def _build_compact_checkpoint_payload(epoch, best_val_f1):
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
        }

    def _safe_save_checkpoint(payload, output_path, compact_builder=None):
        tmp_path = output_path + ".tmp"
        save_variants = [
            {"_use_new_zipfile_serialization": True},
            {"_use_new_zipfile_serialization": False},
        ]
        errors = []

        for kwargs in save_variants:
            try:
                torch.save(payload, tmp_path, **kwargs)
                os.replace(tmp_path, output_path)
                return True
            except Exception as e:
                errors.append(e)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        if compact_builder is not None:
            try:
                compact_payload = compact_builder()
                torch.save(compact_payload, tmp_path, _use_new_zipfile_serialization=False)
                os.replace(tmp_path, output_path)
                print("[WARN] Full checkpoint save failed; saved compact checkpoint without LLM backbone.")
                return True
            except Exception as compact_err:
                errors.append(compact_err)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        if errors:
            print(f"[WARN] Failed to save checkpoint at {output_path}: {errors[-1]}")
        return False

    def _load_checkpoint_into_current_model(checkpoint_path):
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
        return True

    for epoch in range(total_epochs):
        target_phase = 'lora' if epoch < lora_epochs else 'gnn'
        if target_phase != phase:
            phase = target_phase
            optimizer = _configure_training_phase(phase)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, DEVICE)
            enc = batch['encodings']
            item = batch['items'][0]
            core_model = _unwrap_model(model)

            # Layer 1-2: LLM Encoder (frozen + LoRA) → Entity mention embeddings
            doc_emb = core_model.llm(**enc, output_hidden_states=True).hidden_states[-1][0]
            w_ids = enc.word_ids(0)

            vertex_set = item.get('vertex_set', item.get('vertexSet', []))
            candidates = candidate_gen.generate_candidates(item, vertex_set)
            pair_to_rels = build_pair_relation_sets(item.get('labels', []), rel2id)
            must_keep_pairs = set(pair_to_rels.keys())

            candidates = candidate_gen.prefilter_candidates(
                item,
                vertex_set,
                candidates,
                keep_ratio=args.candidate_keep_ratio,
                must_keep=must_keep_pairs,
            )

            # Limit pairs to max_pairs
            candidates = limit_candidates_preserve_must_keep(candidates, must_keep_pairs, max_pairs)

            pos = [p for p in candidates if p in pair_to_rels]
            neg = [p for p in candidates if p not in pair_to_rels]
            neg_budget = int(max(0.0, float(args.neg_pos_ratio)) * len(pos)) + max(0, int(args.neg_buffer))
            sampled_neg = random.sample(neg, min(len(neg), neg_budget)) if neg_budget > 0 else []
            train_p = pos + sampled_neg
            random.shuffle(train_p)

            if not train_p:
                continue

            optimizer.zero_grad()
            doc_loss = 0.0

            # Layer 3: Pair subgraph construction (k-hop neighborhoods)
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

                # Total Loss = L_CE + λ_moe * L_balance + λ_scl * L_scl
                batch_loss = cls_loss + args.lambda_moe * switch_loss + args.lambda_scl * scl_loss

                if not torch.isfinite(batch_loss):
                    if rank == 0:
                        print(f"[WARN] Non-finite batch loss at epoch={epoch}, doc={i}, pair_range=({p_s},{p_e}). Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if not batch_loss.requires_grad:
                    if rank == 0:
                        print(f"[WARN] Loss has no grad_fn at epoch={epoch}, doc={i}, pair_range=({p_s},{p_e}); skipping this pair-batch.")
                    continue

                batch_loss.backward(retain_graph=(p_e < len(train_p)))
                doc_loss += float(batch_loss.item())
                
                # Clear memory after backward pass
                del batch_loss, cls_loss, switch_loss, scl_loss, logits, router_logits, pair_repr, sgs, adjs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            grad_params = []
            for g in optimizer.param_groups:
                grad_params.extend(g['params'])
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_params, max_norm=args.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                if rank == 0:
                    print(f"[WARN] Non-finite grad norm at epoch={epoch}, doc={i}. Skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            epoch_loss += doc_loss
            if rank == 0 and i % 10 == 0:
                print(f"E {epoch} ({phase}) | D {i} | L: {epoch_loss / (i + 1):.4f}")

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_train_loss = _all_reduce_scalar(avg_train_loss, DEVICE)

        eval_f1 = best_f1 if best_f1 >= 0 else 0.0
        eval_metrics = None
        if rank == 0:
            eval_target = train_data_subset if (overfit_one_doc or overfit_stage) else dev_data_subset
            eval_metrics = evaluate_model(
                model,
                tokenizer,
                eval_target,
                graph_builder,
                rel2id,
                DEVICE,
                id2rel,
                debug=False,
                debug_samples=len(eval_target),
                candidate_gen=candidate_gen,
                use_pair_markers=use_pair_markers,
                e1_id=e1_id,
                e2_id=e2_id,
                candidate_keep_ratio=args.candidate_keep_ratio,
                threshold_scale=args.adaptive_threshold_scale,
                max_pairs_per_doc=max_pairs,
                args_max_length=args.max_seq_length,
                tail_buckets=tail_buckets,
                train_facts_annotated=train_facts_annotated,
                train_facts_distant=train_facts_distant,
            )
            eval_f1 = float(eval_metrics.get('f1', 0.0))
            print(f"Epoch {epoch} F1: {eval_f1:.4f}")
            if eval_metrics is not None:
                print(
                    "[EVAL] "
                    f"macro_f1={eval_metrics.get('macro_f1', 0.0):.4f} "
                    f"tail_f1(head/med/tail)="
                    f"{eval_metrics.get('tail_f1_head', 0.0):.4f}/"
                    f"{eval_metrics.get('tail_f1_medium', 0.0):.4f}/"
                    f"{eval_metrics.get('tail_f1_tail', 0.0):.4f} "
                    f"coverage={int(eval_metrics.get('coverage_relations_nonzero', 0.0))}/{len(rel2id)}"
                )
                _print_official_metrics("EVAL", eval_metrics)
                _append_eval_result_line("val", epoch, eval_metrics, train_loss=avg_train_loss)
                _sync_eval_log_to_wandb()
            if use_wandb:
                log_metrics({
                    "val/f1": eval_f1,
                    "val/precision": float(eval_metrics.get('precision', 0.0)) if eval_metrics else 0.0,
                    "val/recall": float(eval_metrics.get('recall', 0.0)) if eval_metrics else 0.0,
                    "val/RE_P": float(eval_metrics.get('RE_P', 0.0)) if eval_metrics else 0.0,
                    "val/RE_R": float(eval_metrics.get('RE_R', 0.0)) if eval_metrics else 0.0,
                    "val/F1-RE": float(eval_metrics.get('F1-RE', 0.0)) if eval_metrics else 0.0,
                    "val/Evidence_P": float(eval_metrics.get('Evidence_P', 0.0)) if eval_metrics else 0.0,
                    "val/Evidence_R": float(eval_metrics.get('Evidence_R', 0.0)) if eval_metrics else 0.0,
                    "val/F1-Evidence": float(eval_metrics.get('F1-Evidence', 0.0)) if eval_metrics else 0.0,
                    "val/RE_ignore_annotated_P": float(eval_metrics.get('RE_ignore_annotated_P', 0.0)) if eval_metrics else 0.0,
                    "val/RE_ignore_annotated_R": float(eval_metrics.get('RE_ignore_annotated_R', 0.0)) if eval_metrics else 0.0,
                    "val/RE_ignore_annotated_F1": float(eval_metrics.get('RE_ignore_annotated_F1', 0.0)) if eval_metrics else 0.0,
                    "val/RE_ignore_distant_P": float(eval_metrics.get('RE_ignore_distant_P', 0.0)) if eval_metrics else 0.0,
                    "val/RE_ignore_distant_R": float(eval_metrics.get('RE_ignore_distant_R', 0.0)) if eval_metrics else 0.0,
                    "val/RE_ignore_distant_F1": float(eval_metrics.get('RE_ignore_distant_F1', 0.0)) if eval_metrics else 0.0,
                    "val/macro_f1": float(eval_metrics.get('macro_f1', 0.0)) if eval_metrics else 0.0,
                    "val/tail_f1_head": float(eval_metrics.get('tail_f1_head', 0.0)) if eval_metrics else 0.0,
                    "val/tail_f1_medium": float(eval_metrics.get('tail_f1_medium', 0.0)) if eval_metrics else 0.0,
                    "val/tail_f1_tail": float(eval_metrics.get('tail_f1_tail', 0.0)) if eval_metrics else 0.0,
                    "val/coverage_relations_nonzero": float(eval_metrics.get('coverage_relations_nonzero', 0.0)) if eval_metrics else 0.0,
                    "val/coverage_ratio": float(eval_metrics.get('coverage_ratio', 0.0)) if eval_metrics else 0.0,
                    "train/loss": avg_train_loss,
                })

        if distributed:
            eval_f1_t = torch.tensor(eval_f1, device=DEVICE, dtype=torch.float32)
            dist.broadcast(eval_f1_t, src=0)
            eval_f1 = float(eval_f1_t.item())

        # Early stopping logic
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_epoch = epoch
            epochs_no_improve = 0
            if rank == 0:
                ckpt_payload = {
                    "epoch": int(epoch),
                    "best_val_f1": float(best_f1),
                    "model_state_dict": _unwrap_model(model).state_dict(),
                    "prototype_state_dict": _unwrap_model(prototypes).state_dict(),
                    "args": vars(args),
                    "run_name": str(run_name),
                }
                ckpt_saved = _safe_save_checkpoint(
                    ckpt_payload,
                    best_ckpt_path,
                    compact_builder=lambda: _build_compact_checkpoint_payload(epoch, best_f1),
                )
                if ckpt_saved and os.path.exists(best_ckpt_path):
                    try:
                        shutil.copy2(best_ckpt_path, best_infer_ckpt_path)
                    except Exception as e:
                        print(f"[WARN] Failed to copy best checkpoint for inference: {e}")
                if use_wandb and ckpt_saved and os.path.exists(best_ckpt_path):
                    save_model_artifact(best_ckpt_path, name=f"{safe_run_name}_best_ckpt", artifact_type="model")
                if use_wandb and os.path.exists(best_infer_ckpt_path):
                    save_model_artifact(best_infer_ckpt_path, name=f"{safe_run_name}_best_ckpt_for_inference", artifact_type="model")
            if rank == 0 and test_data_subset is not None and len(test_data_subset) > 0:
                best_test_metrics = evaluate_model(
                    model,
                    tokenizer,
                    test_data_subset,
                    graph_builder,
                    rel2id,
                    DEVICE,
                    id2rel,
                    debug=False,
                    debug_samples=len(test_data_subset),
                    candidate_gen=candidate_gen,
                    use_pair_markers=use_pair_markers,
                    e1_id=e1_id,
                    e2_id=e2_id,
                    candidate_keep_ratio=args.candidate_keep_ratio,
                    threshold_scale=args.adaptive_threshold_scale,
                    max_pairs_per_doc=max_pairs,
                    args_max_length=args.max_seq_length,
                    result_output_path=best_ckpt_result_path,
                    print_infer_sample=True,
                    tail_buckets=tail_buckets,
                    train_facts_annotated=train_facts_annotated,
                    train_facts_distant=train_facts_distant,
                )
                best_test_f1 = float(best_test_metrics.get('f1', 0.0)) if best_test_metrics else 0.0
                _upload_inference_json_to_wandb(best_ckpt_result_path, artifact_name=f"{safe_run_name}_best_ckpt_result")
                _print_official_metrics("BEST-TEST", best_test_metrics)
                _append_eval_result_line("best_test", epoch, best_test_metrics, train_loss=avg_train_loss)
                _sync_eval_log_to_wandb()
                print(f"[BEST] Updated best checkpoint at epoch {best_epoch} | val F1={best_f1:.4f} | test F1={best_test_f1:.4f}")
        else:
            epochs_no_improve += 1
            if rank == 0:
                print(f"[INFO] No improvement in F1 for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.patience:
            if rank == 0:
                print(f"[INFO] Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    if rank == 0 and test_data_subset is not None and len(test_data_subset) > 0:
        print("\n" + "=" * 60)
        print(f"[TEST] Best-checkpoint summary | epoch={best_epoch} | val F1={best_f1:.4f}")
        ckpt_for_infer = best_infer_ckpt_path if os.path.exists(best_infer_ckpt_path) else best_ckpt_path
        if _load_checkpoint_into_current_model(ckpt_for_infer):
            print(f"[TEST] Loaded checkpoint for inference: {ckpt_for_infer}")
        else:
            print(f"[WARN] Could not load checkpoint for inference: {ckpt_for_infer}. Using current in-memory weights.")

        best_test_metrics = evaluate_model(
            model,
            tokenizer,
            test_data_subset,
            graph_builder,
            rel2id,
            DEVICE,
            id2rel,
            debug=False,
            debug_samples=len(test_data_subset),
            candidate_gen=candidate_gen,
            use_pair_markers=use_pair_markers,
            e1_id=e1_id,
            e2_id=e2_id,
            candidate_keep_ratio=args.candidate_keep_ratio,
            threshold_scale=args.adaptive_threshold_scale,
            max_pairs_per_doc=max_pairs,
            args_max_length=args.max_seq_length,
            result_output_path=best_ckpt_result_path,
            print_infer_sample=True,
            tail_buckets=tail_buckets,
            train_facts_annotated=train_facts_annotated,
            train_facts_distant=train_facts_distant,
        )
        best_test_f1 = float(best_test_metrics.get('f1', 0.0)) if best_test_metrics else 0.0
        _upload_inference_json_to_wandb(best_ckpt_result_path, artifact_name=f"{safe_run_name}_best_ckpt_result_final")
        print(f"[TEST] Test docs: {len(test_data_subset)} from {args.test_file}")
        print(f"[TEST] F1 (best-checkpoint): {best_test_f1:.4f}")
        if best_test_metrics is not None:
            print(
                "[TEST] "
                f"macro_f1={best_test_metrics.get('macro_f1', 0.0):.4f} "
                f"tail_f1(head/med/tail)="
                f"{best_test_metrics.get('tail_f1_head', 0.0):.4f}/"
                f"{best_test_metrics.get('tail_f1_medium', 0.0):.4f}/"
                f"{best_test_metrics.get('tail_f1_tail', 0.0):.4f} "
                f"coverage={int(best_test_metrics.get('coverage_relations_nonzero', 0.0))}/{len(rel2id)}"
            )
            _print_official_metrics("TEST", best_test_metrics)
            _append_eval_result_line("final_test", best_epoch, best_test_metrics)
            _sync_eval_log_to_wandb()
        if use_wandb:
            payload = {
                "test/f1": best_test_f1,
                "best/epoch": best_epoch,
                "best/val_f1": best_f1,
            }
            if best_test_metrics is not None:
                payload.update({
                    "test/precision": float(best_test_metrics.get('precision', 0.0)),
                    "test/recall": float(best_test_metrics.get('recall', 0.0)),
                    "test/macro_f1": float(best_test_metrics.get('macro_f1', 0.0)),
                    "test/tail_f1_head": float(best_test_metrics.get('tail_f1_head', 0.0)),
                    "test/tail_f1_medium": float(best_test_metrics.get('tail_f1_medium', 0.0)),
                    "test/tail_f1_tail": float(best_test_metrics.get('tail_f1_tail', 0.0)),
                    "test/coverage_relations_nonzero": float(best_test_metrics.get('coverage_relations_nonzero', 0.0)),
                    "test/coverage_ratio": float(best_test_metrics.get('coverage_ratio', 0.0)),
                    "test/RE_P": float(best_test_metrics.get('RE_P', 0.0)),
                    "test/RE_R": float(best_test_metrics.get('RE_R', 0.0)),
                    "test/F1-RE": float(best_test_metrics.get('F1-RE', 0.0)),
                    "test/Evidence_P": float(best_test_metrics.get('Evidence_P', 0.0)),
                    "test/Evidence_R": float(best_test_metrics.get('Evidence_R', 0.0)),
                    "test/F1-Evidence": float(best_test_metrics.get('F1-Evidence', 0.0)),
                    "test/RE_ignore_annotated_P": float(best_test_metrics.get('RE_ignore_annotated_P', 0.0)),
                    "test/RE_ignore_annotated_R": float(best_test_metrics.get('RE_ignore_annotated_R', 0.0)),
                    "test/RE_ignore_annotated_F1": float(best_test_metrics.get('RE_ignore_annotated_F1', 0.0)),
                    "test/RE_ignore_distant_P": float(best_test_metrics.get('RE_ignore_distant_P', 0.0)),
                    "test/RE_ignore_distant_R": float(best_test_metrics.get('RE_ignore_distant_R', 0.0)),
                    "test/RE_ignore_distant_F1": float(best_test_metrics.get('RE_ignore_distant_F1', 0.0)),
                    "RE_P": float(best_test_metrics.get('RE_P', 0.0)),
                    "RE_R": float(best_test_metrics.get('RE_R', 0.0)),
                    "F1-RE": float(best_test_metrics.get('F1-RE', 0.0)),
                    "Evidence_P": float(best_test_metrics.get('Evidence_P', 0.0)),
                    "Evidence_R": float(best_test_metrics.get('Evidence_R', 0.0)),
                    "F1-Evidence": float(best_test_metrics.get('F1-Evidence', 0.0)),
                    "RE_ignore_annotated_P": float(best_test_metrics.get('RE_ignore_annotated_P', 0.0)),
                    "RE_ignore_annotated_R": float(best_test_metrics.get('RE_ignore_annotated_R', 0.0)),
                    "RE_ignore_annotated_F1": float(best_test_metrics.get('RE_ignore_annotated_F1', 0.0)),
                    "RE_ignore_distant_P": float(best_test_metrics.get('RE_ignore_distant_P', 0.0)),
                    "RE_ignore_distant_R": float(best_test_metrics.get('RE_ignore_distant_R', 0.0)),
                    "RE_ignore_distant_F1": float(best_test_metrics.get('RE_ignore_distant_F1', 0.0)),
                })
            log_metrics(payload)
            if os.path.exists(best_ckpt_path):
                save_model_artifact(best_ckpt_path, name=f"{safe_run_name}_best_ckpt_final", artifact_type="model")
            if os.path.exists(best_infer_ckpt_path):
                save_model_artifact(best_infer_ckpt_path, name=f"{safe_run_name}_best_ckpt_for_inference_final", artifact_type="model")

    if use_wandb:
        _sync_eval_log_to_wandb()

    if use_wandb:
        _upload_inference_results_dir_to_wandb(
            result_dir_abs,
            artifact_name=f"{safe_run_name}_inference_results",
        )

    if use_wandb:
        _upload_workspace_snapshot_to_wandb(workspace_root_abs, artifact_name=f"{safe_run_name}_workspace_final")
        _upload_single_file_to_wandb(os.path.join(workspace_root_abs, "moe.py"), artifact_name=f"{safe_run_name}_moe_py_final", artifact_type="source")

    if use_wandb:
        finish_run()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Exiting gracefully.")
        try:
            finish_run()
        except Exception:
            pass