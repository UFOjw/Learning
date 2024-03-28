from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import List, Generator, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import numpy as np
import math
import torch


@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        with open(self.path, 'r') as file:
            lines = file.readlines()

        # Убираем заголовок
        lines.pop(0)

        # Считать данные
        data = []
        for line in lines:
            row = line.replace('\n', '').split(',')
            data.append(row)

        data_len = len(data)

        return math.ceil(data_len / self.batch_size)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        tokens = []
        max_batch_len = 0
        
        for row in batch:
            token = self.tokenizer.encode(
                                        row,
                                        add_special_tokens=True,
                                        max_length=self.max_length, 
                                        truncation=True)
            # Максимальная длина в батче
            if max_batch_len < len(token):
                max_batch_len = len(token)
            tokens.append(token)
        
        if self.padding == 'max_lenght':
            tokens = self._padded_tokens(tokens, max_batch_len)
        elif self.padding == 'batch':
            tokens = self._padded_tokens(tokens, self.max_length)
        return tokens
    
    def _padded_tokens(self, tokens: List[List[int]], max_length: int) -> List[List[int]]:
        for idx, token in enumerate(tokens):
            token_len = len(token)
            if token_len < max_length:
                dev = max_length - token_len
                empty_list = [None for _ in range(dev)]
                token.extend(empty_list)
                tokens[idx] = token
        return tokens
    
    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        texts = []
        labels = []

        with open(self.path, 'r') as file:
            lines = file.readlines()

        # Убираем заголовок
        lines.pop(0)

        # Считать данные
        data = []
        for line in lines:
            row = line.replace('\n', '').split(',')
            data.append(row)

        data_len = len(data)

        start_line = i * self.batch_size
        end_line = (i + 1) * self.batch_size
        if data_len < end_line:
            end_line = data_len

        for idx in range(start_line, end_line):
            score = data[idx][2]
            labels.append(self._convert_label(score))
            text = ",".join(data[idx][4:]).replace("\n", "")
            texts.append(text)
        return texts, labels

    def _convert_label(self, score: int) -> int:
        if score == '1':
            return -1
        elif score == '5':
            return 1
        else:
            return 0

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels
    
def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    for idx, token in enumerate(padded):
        pos = bin_search(token)
        token_len = len(token)
        dev_len = token_len - pos
        ones = [1 for _ in range(pos)]
        zeros = [0 for _ in range(dev_len)]
        ones.extend(zeros)
        padded[idx] = ones
    return padded

def bin_search(padded_token: List[int]) -> int:
    left = 0
    right = len(padded_token)
    while left < right:
        m = (left + right) // 2
        if padded_token[m] is None:
            right = m
        else:
            left = m + 1
    return left

def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    # Attention mask
    mask = attention_mask(tokens)

    # Calculate embeddings
    token = torch.tensor(tokens) 
    mask = torch.tensor(mask)

    with torch.no_grad():
        last_hidden_states = model(token, attention_mask=mask)

    # Embeddings for [CLS]-tokens
    cls_embeds = last_hidden_states[0][:,0,:].tolist()
    return cls_embeds

def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    kf = KFold(n_splits=cv)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    embeddings.reshape(1, -1)
    labels.reshape(1, -1)
    scores = []
    for train, test in kf.split(embeddings):
        model.fit(embeddings[train], labels[train])
        preds = model.predict_proba(embeddings[test])
        score = log_loss(labels[test], preds)
        scores.append(score)
    return scores