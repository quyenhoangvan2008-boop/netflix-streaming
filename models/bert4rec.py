"""
BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer (CIKM 2019)
Reference: Sun et al., 2019.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class BERTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.dropout_1 = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        norm_x = self.ln_1(x)
        attn_out = self.attn(norm_x, mask=mask)
        x = x + self.dropout_1(attn_out)

        norm_x2 = self.ln_2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out
        return x

class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        mask_prob: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob

        # 0: PAD, 1..num_items: Items, num_items + 1: [MASK]
        self.mask_token = num_items + 1
        vocab_size = num_items + 2

        self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_final = nn.LayerNorm(hidden_dim, eps=1e-8)

        self.blocks = nn.ModuleList([
            BERTBlock(hidden_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_blocks)
        ])

        self.out_proj = nn.Linear(hidden_dim, num_items + 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_seqs: torch.Tensor):
        batch_size, seq_len = input_seqs.shape
        device = input_seqs.device

        seq_embs = self.item_embedding(input_seqs) * math.sqrt(self.hidden_dim)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embs = self.pos_embedding(positions)

        x = self.emb_dropout(seq_embs + pos_embs)
        # Bidirectional attention: can attend to all positions except padding
        pad_mask = (input_seqs != 0).unsqueeze(1).unsqueeze(2) # [B, 1, 1, seq_len]

        for block in self.blocks:
            x = block(x, mask=pad_mask)

        x = self.ln_final(x)
        logits = self.out_proj(x) # [batch_size, seq_len, num_items + 1]
        return logits

    def create_masked_training_sample(self, seq: list):
        masked_seq = list(seq)
        labels = [0] * len(seq)
        for i in range(len(seq)):
            if random.random() < self.mask_prob:
                labels[i] = seq[i]
                masked_seq[i] = self.mask_token
        
        if sum(labels) == 0 and len(seq) > 0:
            labels[-1] = seq[-1]
            masked_seq[-1] = self.mask_token

        return masked_seq, labels

    def compute_loss(self, input_seqs: torch.Tensor, labels: torch.Tensor):
        logits = self.forward(input_seqs)
        logits = logits.view(-1, logits.size(-1))
        targets = labels.view(-1)

        mask = targets > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=input_seqs.device)

        loss = F.cross_entropy(logits[mask], targets[mask])
        return loss

    def predict_next_item(self, input_seq: torch.Tensor):
        logits = self.forward(input_seq)
        last_logits = logits[0, -1, 1:self.num_items + 1]
        return last_logits
