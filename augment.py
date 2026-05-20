"""
DocRED T5 Paraphrase Augmentation
Entity-preserving paraphrase with position realignment.
"""
import re
import copy
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _word_tokenize(text):
    """Simple regex-based tokenization matching DocRED style."""
    return re.findall(r"\w+|[^\w\s]", text)


class DocParaphraser:
    def __init__(self, model_name="humarin/chatgpt_paraphraser_on_T5_base", device="cpu", max_length=256):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def paraphrase_text(self, text, num_return_sequences=1):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.max_length,
            num_return_sequences=num_return_sequences,
            num_beams=max(4, num_return_sequences),
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @staticmethod
    def _tokenize_name(name):
        return _word_tokenize(name)

    def _find_mention_positions(self, words, mention_name):
        """Find all (start, end) positions of mention_name tokens in words list."""
        name_tokens = self._tokenize_name(mention_name)
        n = len(name_tokens)
        positions = []
        for i in range(len(words) - n + 1):
            if words[i : i + n] == name_tokens:
                positions.append((i, i + n))
        return positions

    def _realign_sentence(self, paraphrased_text, sent_idx, vertex_set):
        """
        Tokenize paraphrased text and try to find every mention in this sentence.
        Returns (new_words, success).
        """
        new_words = _word_tokenize(paraphrased_text)
        for ent_idx, entity in enumerate(vertex_set):
            for ment in entity:
                if int(ment.get("sent_id", 0)) != sent_idx:
                    continue
                name = ment["name"]
                pos_list = self._find_mention_positions(new_words, name)
                if not pos_list:
                    return None, False
        return new_words, True

    def augment_doc(self, doc, augment_sent_prob=0.3):
        """
        Augment a single DocRED document.
        Paraphrases sentences that contain entity mentions with given probability.
        Returns a new doc dict or None if no sentence was successfully updated.
        """
        doc = copy.deepcopy(doc)
        vertex_set = doc.get("vertexSet", doc.get("vertex_set", []))
        sents = doc["sents"]

        # Map sent_id -> has mentions?
        sent_has_mentions = set()
        for entity in vertex_set:
            for ment in entity:
                sent_has_mentions.add(int(ment.get("sent_id", 0)))

        new_sents = []
        updated_any = False
        for sid, sent_words in enumerate(sents):
            if sid not in sent_has_mentions or random.random() > augment_sent_prob:
                new_sents.append(sent_words)
                continue

            original_text = " ".join(sent_words)
            try:
                para_texts = self.paraphrase_text(original_text, num_return_sequences=1)
            except Exception:
                new_sents.append(sent_words)
                continue

            para_text = para_texts[0]
            new_words, ok = self._realign_sentence(para_text, sid, vertex_set)
            if not ok:
                new_sents.append(sent_words)
                continue

            new_sents.append(new_words)
            updated_any = True

            # Update positions for mentions in this sentence
            for ent_idx, entity in enumerate(vertex_set):
                updated_mentions = []
                for ment in entity:
                    if int(ment.get("sent_id", 0)) != sid:
                        updated_mentions.append(ment)
                        continue
                    name = ment["name"]
                    positions = self._find_mention_positions(new_words, name)
                    if positions:
                        start, end = positions[0]
                        new_ment = dict(ment)
                        new_ment["pos"] = [start, end]
                        updated_mentions.append(new_ment)
                    else:
                        updated_mentions.append(ment)
                vertex_set[ent_idx] = updated_mentions

        if not updated_any:
            return None

        doc["sents"] = new_sents
        doc["vertexSet"] = vertex_set
        if "vertex_set" in doc:
            doc["vertex_set"] = vertex_set
        return doc

    def augment_dataset(self, data, ratio=0.3, augment_sent_prob=0.3):
        """
        Augment a dataset by creating paraphrased copies of ratio*len(data) documents.
        Returns list of augmented docs.
        """
        n_aug = max(1, int(len(data) * ratio))
        sampled = random.sample(data, min(n_aug, len(data)))
        augmented = []
        for doc in sampled:
            new_doc = self.augment_doc(doc, augment_sent_prob=augment_sent_prob)
            if new_doc is not None:
                augmented.append(new_doc)
        return augmented


def _normalize_text(text):
    """Normalize text by removing spaces around punctuation for matching."""
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.lower()


def quick_entity_presence_filter(original_doc, aug_doc):
    """
    Simple filter: ensure all original entity names still appear in augmented doc.
    """
    orig_names = set()
    for ent in original_doc.get("vertexSet", original_doc.get("vertex_set", [])):
        for m in ent:
            orig_names.add(m["name"])
    aug_text = _normalize_text(" ".join([" ".join(s) for s in aug_doc["sents"]]))
    for name in orig_names:
        if _normalize_text(name) not in aug_text:
            return False
    return True
