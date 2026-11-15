"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)
Reference: He et al., 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        reg_weight: float = 1e-4
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg_weight = reg_weight

        # 0 is user 1 in graph, item 0 is item 1 in graph
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, adj_sparse: torch.Tensor):
        """
        Lightweight Graph Convolution without feature transformations or non-linear activations.
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_sparse, all_emb)
            embs.append(all_emb)

        # Uniform average over all layers
        final_embs = torch.stack(embs, dim=1).mean(dim=1)
        users_final, items_final = torch.split(final_embs, [self.num_users, self.num_items], dim=0)
        return users_final, items_final

    def compute_bpr_loss(
        self,
        users_final: torch.Tensor,
        items_final: torch.Tensor,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ):
        """
        BPR Loss: -ln sigmoid(pos_score - neg_score) + L2 regularization
        """
        u_emb = users_final[user_ids]
        pos_emb = items_final[pos_item_ids]
        neg_emb = items_final[neg_item_ids]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # L2 regularizer on base embeddings (0-th layer)
        u_base = self.user_embedding(user_ids)
        pos_base = self.item_embedding(pos_item_ids)
        neg_base = self.item_embedding(neg_item_ids)
        reg_loss = (
            (u_base.norm(2).pow(2) + pos_base.norm(2).pow(2) + neg_base.norm(2).pow(2))
            / (2 * len(user_ids))
        )

        return bpr_loss + self.reg_weight * reg_loss

    def predict_for_user(self, user_id: int, users_final: torch.Tensor, items_final: torch.Tensor):
        """
        Returns retrieval scores for all items for a given user.
        user_id: 1-indexed
        """
        u_emb = users_final[user_id - 1]
        scores = torch.matmul(items_final, u_emb)
        return scores
