import json
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

with open('dev.json', 'r') as f:
    data = json.load(f)

item = data[0]
first_sent = item['sents'][0]
full_text = " ".join([" ".join(s) for s in item['sents']])

inputs = tokenizer(full_text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print(f"First 10 words in DocRED: {first_sent[:10]}")
print(f"First 10 tokens from LLM: {tokens[:10]}")

# Check an entity
entity = item['vertex_set'][0][0] # first mention of first entity
name = entity['name']
pos = entity['pos']
print(f"\nEntity: {name}, Pos: {pos}")
print(f"Slice from full_text tokens by word index (naively): {tokens[pos[0]:pos[1]]}")
