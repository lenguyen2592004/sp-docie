import json
from collections import Counter

def rel_counts(path):
    data = json.load(open(path, 'r', encoding='utf-8'))
    cnt = Counter()
    for item in data:
        for lbl in item.get('labels', []):
            cnt[lbl['r']] += 1
    return cnt

dev_cnt = rel_counts("dev.json")
ds_cnt = rel_counts("train_distant.json")
ann_cnt = rel_counts("train_annotated.json")

print("DEV top20:", dev_cnt.most_common(20))
print("DS top20:", ds_cnt.most_common(20))
print("ANN top20:", ann_cnt.most_common(20))