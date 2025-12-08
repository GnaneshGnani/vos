import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TemporalAssociationModule(nn.Module):
    def __init__(self, input_dim, embedding_dim = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, tokens): return self.mlp(tokens)

class VITA_TCOVIS(nn.Module):
    def __init__(self, num_tokens = 5, hidden_dim = 256):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Pretrained Backbone
        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[ : -2])
        self.conv_project = nn.Conv2d(2048, hidden_dim, kernel_size = 1)
        
        self.object_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model = hidden_dim, nhead = 8, batch_first = True),
            num_layers = 3 
        )

        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) 
        ) # Retained for processing tokens

        self.temporal_module = TemporalAssociationModule(input_dim = hidden_dim)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        features = self.conv_project(self.backbone(frames_flat)) # B x hidden_dim X W x H
        
        feat_H, feat_W = features.shape[2], features.shape[3]
        features_seq = features.flatten(2).permute(0, 2, 1) # B X (W X H) X hidden_dim
        
        tokens = self.object_tokens.expand(B * T, -1, -1) # (B x T) X num_tokens X hidden_dim
        out_tokens = self.transformer(tokens, features_seq) # (B x T) X num_tokens X hidden_dim
        out_tokens = out_tokens.view(B, T, self.num_tokens, -1) 
        features_aligned = features.view(B, T, -1, feat_H, feat_W)
        
        all_masks, all_embs = [], []
        for t in range(T):
            toks_t = out_tokens[:, t]
            
            # Use self.mask_head to process tokens before mask generation
            processed_toks = self.mask_head(toks_t)
            
            masks_low = torch.bmm(processed_toks, features_aligned[:, t].flatten(2))
            masks_low = masks_low.view(B, self.num_tokens, feat_H, feat_W)
            masks_up = F.interpolate(masks_low, size = (H, W), mode = 'bilinear')
            
            all_masks.append(masks_up)
            all_embs.append(self.temporal_module(toks_t))
            
        return torch.stack(all_masks, dim = 1), torch.stack(all_embs, dim = 1)