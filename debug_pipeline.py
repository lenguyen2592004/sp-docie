import json
import torch
from collections import Counter
import numpy as np

def load_data():
    with open('DocRED/rel_info.json', 'r') as f:
        rel_info = json.load(f)
    
    with open('DocRED/train_annotated.json', 'r') as f:
        train_data = json.load(f)
        
    with open('DocRED/dev.json', 'r') as f:
        dev_data = json.load(f)
        
    return rel_info, train_data, dev_data

def step_1_label_mapping(rel_info, train_data):
    print("\n" + "="*50)
    print("🔴 STEP 1 — LABEL MAPPING & STATISTICS")
    print("="*50)
    
    label2id = {rel: i for i, rel in enumerate(rel_info.keys())}
    # DocRED in moe.py seems to treat NA as the last ID
    NA_ID = len(label2id)
    id2label = {i: rel for rel, i in label2id.items()}
    id2label[NA_ID] = "NA"
    
    print(f"1.1 mapping ID ↔ relation name")
    print(f"label2id: {list(label2id.items())[:5]} ... (total {len(label2id)})")
    print(f"id2label: {list(id2label.items())[:5]} ...")
    print(f"Sorted labels: {sorted(label2id.keys())[:5]} ...")
    print(f"NA ID: {NA_ID}")
    
    # Check uniqueness
    if "NA" in label2id or "no_relation" in label2id:
        print("⚠️ WARNING: 'NA' or 'no_relation' found in rel_info. Check if NA_ID is correctly handled.")

    print(f"\n1.2 Gold labels statistics (Train)")
    cnt = Counter()
    for item in train_data:
        for lbl in item.get('labels', []):
            cnt[lbl['r']] += 1
    
    print(f"Relation counts: {cnt.most_common(10)}")
    zero_count = [rel for rel in label2id.keys() if cnt[rel] == 0]
    if zero_count:
        print(f"❌ Relations with 0 count: {zero_count}")
    else:
        print("✅ All relations have at least one sample in training data.")

    return label2id, id2label, NA_ID

def step_2_sanity_check(dev_data, label2id, NA_ID):
    print("\n" + "="*50)
    print("🔴 STEP 2 — SANITY CHECK EVALUATION")
    print("="*50)
    
    all_gt = []
    all_oracle_preds = []
    all_all_pos_preds = []
    
    # Pick a random relation for "all positive" baseline (e.g., the first one)
    some_rel_id = 0
    
    for item in dev_data[:100]: # Sample 100 docs for speed
        vertex_set = item.get('vertex_set', item.get('vertexSet', []))
        num_entities = len(vertex_set)
        pairs = [(h, t) for h in range(num_entities) for t in range(num_entities) if h != t]
        gt_dict = {(lbl['h'], lbl['t']): label2id.get(lbl['r'], NA_ID) for lbl in item.get('labels', [])}
        
        for pair in pairs:
            gold = gt_dict.get(pair, NA_ID)
            all_gt.append(gold)
            all_oracle_preds.append(gold)
            all_all_pos_preds.append(some_rel_id)

    from sklearn.metrics import precision_recall_fscore_support
    
    # 2.1 Oracle Test
    p, r, f1, _ = precision_recall_fscore_support(all_gt, all_oracle_preds, average='micro', labels=[i for i in range(len(label2id))])
    print(f"2.1 Oracle Test -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    if f1 < 1.0:
        print("❌ ERROR: Oracle test failed! F1 should be 1.0.")
    else:
        print("✅ Oracle test passed.")

    # 2.2 Predict-all-positive
    # Note: Micro F1 excluding NA
    p, r, f1, _ = precision_recall_fscore_support(all_gt, all_all_pos_preds, average='micro', labels=[i for i in range(len(label2id))])
    print(f"2.2 Predict-all-positive baseline -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

def step_3_candidate_explosion(dev_data, label2id, NA_ID):
    print("\n" + "="*50)
    print("🔴 STEP 3 — CANDIDATE EXPLOSION")
    print("="*50)
    
    num_pairs = []
    num_gt = []
    
    distances_pos = []
    distances_neg = []

    for item in dev_data:
        vertex_set = item.get('vertex_set', item.get('vertexSet', []))
        num_entities = len(vertex_set)
        pairs = [(h, t) for h in range(num_entities) for t in range(num_entities) if h != t]
        labels = item.get('labels', [])
        
        num_pairs.append(len(pairs))
        num_gt.append(len(labels))
        
        gt_pairs = {(lbl['h'], lbl['t']): True for lbl in labels}
        
        for pair in pairs:
            # Distance = min distance between any mentions
            h_mentions = vertex_set[pair[0]]
            t_mentions = vertex_set[pair[1]]
            
            min_dist = float('inf')
            for hm in h_mentions:
                for tm in t_mentions:
                    d = abs(hm['pos'][0] - tm['pos'][0])
                    if d < min_dist: min_dist = d
            
            if pair in gt_pairs:
                distances_pos.append(min_dist)
            else:
                distances_neg.append(min_dist)

    avg_pairs = sum(num_pairs)/len(num_pairs)
    avg_gt = sum(num_gt)/len(num_gt)
    print(f"Avg pairs: {avg_pairs:.2f}")
    print(f"Avg GT relations: {avg_gt:.2f}")
    
    if avg_pairs > 300 and avg_gt < 10:
        print("⚠️ PIPELINE WARNING: Heavy negative explosion detected.")
    
    print("\n3.1 Distance Statistics (Word Distance)")
    print(f"Mean distance (Positive): {np.mean(distances_pos):.2f}")
    print(f"Mean distance (Negative): {np.mean(distances_neg):.2f}")
    
    # Bucket distance precision
    buckets = [0, 5, 10, 20, 50, 100, 500]
    for i in range(len(buckets)-1):
        b_min, b_max = buckets[i], buckets[i+1]
        p_count = sum(1 for d in distances_pos if b_min <= d < b_max)
        n_count = sum(1 for d in distances_neg if b_min <= d < b_max)
        total = p_count + n_count
        prec = p_count / total if total > 0 else 0
        print(f"Bucket [{b_min}-{b_max}]: Prec={prec:.4f} ({p_count}/{total})")

if __name__ == "__main__":
    rel_info, train_data, dev_data = load_data()
    label2id, id2label, NA_ID = step_1_label_mapping(rel_info, train_data)
    step_2_sanity_check(dev_data, label2id, NA_ID)
    step_3_candidate_explosion(dev_data, label2id, NA_ID)
