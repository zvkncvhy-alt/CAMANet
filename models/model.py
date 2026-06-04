import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.my_encoder_decoder import EncoderDecoder as r2gen
from modules.standard_trans import EncoderDecoder as st_trans
from modules.cam_attn_con import CamAttnCon
from modules.my_encoder_decoder import LayerNorm
from modules.old_forebacklearning import ForeBackLearning
from modules.evidence_chain import EvidenceChain


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, logger=None, config=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.addcls = args.addcls
        self.vis = args.vis
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args, logger, config)
        self.evidence_chain = getattr(args, 'evidence_chain', False)
        self.fbl = args.fbl and not self.evidence_chain
        self.wmse = args.wmse
        self.attn_cam = args.attn_cam and not self.evidence_chain
        if self.evidence_chain:
            self.evidence = EvidenceChain(
                feature_dim=self.visual_extractor.num_features,
                num_anatomy_queries=args.num_anatomy_queries,
                num_diseases=args.num_disease_labels,
                num_heads=args.evidence_num_heads,
                dropout=args.evidence_dropout,
            )
        if self.fbl:
            self.fore_back_learn = ForeBackLearning(norm=LayerNorm(self.visual_extractor.num_features))
        if self.attn_cam:
            self.attn_cam_con = CamAttnCon(method=args.attn_method, topk=args.topk, layer_id=args.layer_id, vis=args.vis)
        self.sub_back = args.sub_back
        self.records = []
        if args.ed_name == 'r2gen':
            self.encoder_decoder = r2gen(args, tokenizer)
        elif args.ed_name == 'st_trans':
            self.encoder_decoder = st_trans(args, tokenizer)
        else:
            raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _apply_evidence_chain(self, patch_feats):
        evidence_tokens, disease_logits, evidence_outputs = self.evidence(patch_feats)
        patch_feats = torch.cat([evidence_tokens, patch_feats], dim=1)
        return patch_feats, disease_logits, evidence_outputs

    def forward(self, images, targets=None, labels=None, mode='train'):
        fore_map, total_attns, idxs, align_attns_train = None, None, None, None
        clip_loss, logits, evidence_outputs = None, None, None

        if self.evidence_chain:
            patch_feats, gbl_feats = self.visual_extractor(images)
            patch_feats, logits, evidence_outputs = self._apply_evidence_chain(patch_feats)
        elif self.addcls:
            patch_feats, gbl_feats, logits, cams = self.visual_extractor(images)
            if self.fbl:
                fore_rep, back_rep, fore_map = self.fore_back_learn(patch_feats, cams, logits)
                if self.sub_back:
                    patch_feats = patch_feats - back_rep
                patch_feats = torch.cat((fore_rep, patch_feats), dim=1)
        else:
            patch_feats, gbl_feats = self.visual_extractor(images)

        if mode == 'train':
            output, fore_rep_encoded, target_embed, align_attns, clip_loss = self.encoder_decoder(
                gbl_feats, patch_feats, targets, mode='forward'
            )
            if self.evidence_chain:
                evidence_outputs['align_attns'] = align_attns
            elif self.addcls and self.attn_cam:
                total_attns, idxs, align_attns_train = self.attn_cam_con(
                    fore_rep_encoded, target_embed, align_attns, targets
                )
        elif mode == 'sample':
            output, _, attns = self.encoder_decoder(gbl_feats, patch_feats, mode='sample')
        else:
            raise ValueError

        if mode == 'train':
            if self.evidence_chain:
                return output, logits, evidence_outputs, clip_loss
            if self.addcls:
                return output, logits, cams, fore_map, total_attns, idxs, align_attns_train, clip_loss
            return output, clip_loss
        return output, attns, logits
