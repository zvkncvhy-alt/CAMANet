import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceCoverageLoss(nn.Module):
    """Coverage loss over decoder cross-attention to disease evidence tokens."""

    def __init__(self, pos_tau=0.15, neg_weight=0.25, div_weight=0.01):
        super(EvidenceCoverageLoss, self).__init__()
        self.pos_tau = pos_tau
        self.neg_weight = neg_weight
        self.div_weight = div_weight

    def forward(self, evidence_outputs, labels, reports_masks=None):
        align_attns = evidence_outputs['align_attns']
        disease_start = evidence_outputs['disease_token_start']
        disease_end = evidence_outputs['disease_token_end']
        anatomy_attn = evidence_outputs.get('anatomy_attn')

        attn = align_attns[-1].mean(dim=1)
        disease_attn = attn[:, :, disease_start:disease_end]
        if reports_masks is not None:
            token_mask = reports_masks[:, 1:1 + disease_attn.size(1)].float().unsqueeze(-1)
            disease_attn = disease_attn * token_mask
            denom = token_mask.sum(dim=1).clamp_min(1.0)
        else:
            denom = disease_attn.new_full((disease_attn.size(0), 1), disease_attn.size(1))

        labels = labels.float()
        max_disease_attn = disease_attn.max(dim=1).values
        mean_disease_attn = disease_attn.sum(dim=1) / denom

        pos_loss = F.relu(self.pos_tau - max_disease_attn) * labels
        pos_loss = pos_loss.sum() / labels.sum().clamp_min(1.0)

        neg_labels = 1.0 - labels
        neg_loss = mean_disease_attn * neg_labels
        neg_loss = neg_loss.sum() / neg_labels.sum().clamp_min(1.0)

        div_loss = disease_attn.new_tensor(0.0)
        if anatomy_attn is not None and self.div_weight > 0:
            anatomy_map = anatomy_attn.mean(dim=1)
            anatomy_map = F.normalize(anatomy_map, p=2, dim=-1)
            sim = torch.matmul(anatomy_map, anatomy_map.transpose(-2, -1))
            eye = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool).unsqueeze(0)
            div_loss = sim.masked_select(~eye).mean()

        total = pos_loss + self.neg_weight * neg_loss + self.div_weight * div_loss
        return total, {
            'cov_pos_loss': pos_loss.detach(),
            'cov_neg_loss': neg_loss.detach(),
            'evidence_div_loss': div_loss.detach(),
        }
