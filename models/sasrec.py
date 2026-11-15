"""
SASRec: Self-Attentive Sequential Recommendation (ICDM 2018)
Reference: Kang & McAuley, 2018.
"""

import math
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
        # x: [batch_size, seq_len, hidden_dim]
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, num_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SASRecBlock(nn.Module):
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

class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Index 0 is reserved for padding, items are 1..num_items
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_final = nn.LayerNorm(hidden_dim, eps=1e-8)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_sequence(self, input_seqs: torch.Tensor):
        """
        input_seqs: [batch_size, seq_len], 0-padded
        """
        batch_size, seq_len = input_seqs.shape
        device = input_seqs.device

        seq_embs = self.item_embedding(input_seqs) * math.sqrt(self.hidden_dim)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embs = self.pos_embedding(positions)

        x = self.emb_dropout(seq_embs + pos_embs)

        # Causal mask: position t can only attend to <= t
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
        # Padding mask: do not attend to token 0
        pad_mask = (input_seqs != 0).unsqueeze(1).unsqueeze(2) # [B, 1, 1, seq_len]
        total_mask = causal_mask & pad_mask # [B, 1, seq_len, seq_len]

        for block in self.blocks:
            x = block(x, mask=total_mask)

        x = self.ln_final(x)
        return x

    def compute_loss(self, input_seqs: torch.Tensor, pos_targets: torch.Tensor, neg_targets: torch.Tensor):
        """
        Computes BCE loss over sequential predictions.
        """
        seq_embs = self.encode_sequence(input_seqs) # [B, T, D]
        pos_embs = self.item_embedding(pos_targets) # [B, T, D]
        neg_embs = self.item_embedding(neg_targets) # [B, T, D]

        pos_logits = (seq_embs * pos_embs).sum(dim=-1) # [B, T]
        neg_logits = (seq_embs * neg_embs).sum(dim=-1) # [B, T]

        mask = (pos_targets != 0).float()

        loss = -(
            F.logsigmoid(pos_logits) * mask +
            F.logsigmoid(-neg_logits) * mask
        ).sum() / (mask.sum() + 1e-8)

        return loss

    def predict_next_item(self, input_seq: torch.Tensor):
        """
        input_seq: [1, seq_len]
        Returns retrieval scores over all items (1..num_items).
        """
        seq_emb = self.encode_sequence(input_seq)
        last_hidden = seq_emb[:, -1, :] # [1, D]
        
        all_item_ids = torch.arange(1, self.num_items + 1, device=input_seq.device)
        candidate_embs = self.item_embedding(all_item_ids) # [num_items, D]
        
        scores = torch.matmul(candidate_embs, last_hidden.squeeze(0)) # [num_items]
        return scores
