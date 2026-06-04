import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyEvidenceTokenizer(nn.Module):
    """Convert unordered visual patch tokens into learnable anatomy evidence slots."""

    def __init__(self, feature_dim, num_anatomy_queries=8, num_heads=8, dropout=0.1):
        super(AnatomyEvidenceTokenizer, self).__init__()
        self.num_anatomy_queries = num_anatomy_queries
        self.anatomy_queries = nn.Parameter(torch.randn(num_anatomy_queries, feature_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_feats):
        batch_size = patch_feats.size(0)
        queries = self.anatomy_queries.unsqueeze(0).expand(batch_size, -1, -1)
        evidence, attn = self.cross_attn(
            queries,
            patch_feats,
            patch_feats,
            need_weights=True,
            average_attn_weights=False,
        )
        evidence = self.norm(queries + self.dropout(evidence))
        evidence = self.norm(evidence + self.dropout(self.ffn(evidence)))
        return evidence, attn


class DiseaseStateRefiner(nn.Module):
    """Derive disease-specific evidence tokens from anatomy evidence tokens."""

    def __init__(self, feature_dim, num_diseases=14, num_heads=8, dropout=0.1):
        super(DiseaseStateRefiner, self).__init__()
        self.num_diseases = num_diseases
        self.disease_queries = nn.Parameter(torch.randn(num_diseases, feature_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_diseases)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim + 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, anatomy_tokens):
        batch_size = anatomy_tokens.size(0)
        pooled_anatomy = self.pool_norm(torch.mean(anatomy_tokens, dim=1))
        disease_logits = self.classifier(pooled_anatomy)
        disease_probs = torch.sigmoid(disease_logits)
        uncertainty = disease_probs * (1.0 - disease_probs)

        queries = self.disease_queries.unsqueeze(0).expand(batch_size, -1, -1)
        disease_tokens, disease_to_anatomy_attn = self.cross_attn(
            queries,
            anatomy_tokens,
            anatomy_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        gate_input = torch.cat(
            [queries, disease_probs.unsqueeze(-1), uncertainty.unsqueeze(-1)], dim=-1
        )
        gates = torch.sigmoid(self.gate(gate_input))
        disease_tokens = self.norm(queries + self.dropout(gates * disease_tokens))
        return disease_tokens, disease_logits, gates, disease_to_anatomy_attn


class EvidenceChain(nn.Module):
    """End-to-end evidence chain: anatomy slots -> disease-state evidence tokens."""

    def __init__(
        self,
        feature_dim,
        num_anatomy_queries=8,
        num_diseases=14,
        num_heads=8,
        dropout=0.1,
    ):
        super(EvidenceChain, self).__init__()
        self.anatomy_tokenizer = AnatomyEvidenceTokenizer(
            feature_dim=feature_dim,
            num_anatomy_queries=num_anatomy_queries,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.disease_refiner = DiseaseStateRefiner(
            feature_dim=feature_dim,
            num_diseases=num_diseases,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.num_anatomy_queries = num_anatomy_queries
        self.num_diseases = num_diseases

    def forward(self, patch_feats):
        anatomy_tokens, anatomy_attn = self.anatomy_tokenizer(patch_feats)
        disease_tokens, disease_logits, disease_gates, disease_attn = self.disease_refiner(anatomy_tokens)
        evidence_tokens = torch.cat([anatomy_tokens, disease_tokens], dim=1)
        aux = {
            'anatomy_attn': anatomy_attn,
            'disease_attn': disease_attn,
            'disease_gates': disease_gates,
            'disease_token_start': self.num_anatomy_queries,
            'disease_token_end': self.num_anatomy_queries + self.num_diseases,
        }
        return evidence_tokens, disease_logits, aux
