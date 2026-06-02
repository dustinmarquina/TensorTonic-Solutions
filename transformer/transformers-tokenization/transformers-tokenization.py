import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        #special token:
        special_token = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_token):
            self.word_to_id[token] = i

        # use text.lower().split() to get tokens and add unique tokens to vocab
        tokens = []
        for text in texts:
            tokens.extend(text.lower().split())
        tokens = sorted(set(tokens))  # Get unique tokens and sort them
        for i in range(len(special_token), len(special_token) + len(tokens)):
            self.word_to_id[tokens[i - len(special_token)]] = i

        # Build reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.lower().split()
        return [self.word_to_id.get(token, self.word_to_id[self.unk_token]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # Return words for ids; use unknown token string as default
        return " ".join([self.id_to_word.get(i, self.unk_token) for i in ids])