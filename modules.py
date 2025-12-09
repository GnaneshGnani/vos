import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class TemporalAssociationModule(nn.Module):
    def __init__(self, input_dim, embedding_dim = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, tokens): 
        return self.mlp(tokens)

class SimplePixelDecoder(nn.Module):
    def __init__(self, input_dims, hidden_dim = 256):
        super().__init__()
        # Input dims: [512, 1024, 2048] corresponding to ResNet layers 2, 3, 4
        self.layer_3_proj = nn.Conv2d(input_dims[0], hidden_dim, 1)
        self.layer_4_proj = nn.Conv2d(input_dims[1], hidden_dim, 1)
        self.layer_5_proj = nn.Conv2d(input_dims[2], hidden_dim, 1)
        
        # Final smoothing convolution
        self.adapter = nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1)

    def forward(self, features):
        # Extract features from dictionary
        c3 = features["layer2"] # Stride 8
        c4 = features["layer3"] # Stride 16
        c5 = features["layer4"] # Stride 32
        
        # Top-down FPN pathway
        p5 = self.layer_5_proj(c5)
        
        # Upsample P5 and add to P4
        p4 = self.layer_4_proj(c4) + F.interpolate(p5, scale_factor = 2, mode = 'nearest')
        
        # Upsample P4 and add to P3
        p3 = self.layer_3_proj(c3) + F.interpolate(p4, scale_factor = 2, mode = 'nearest')
        
        # Smooth the final Stride 8 feature map
        mask_features = self.adapter(p3)
        
        # Return High-Res (Stride 8) for Masks, and Low-Res (Stride 32) for Transformer
        return mask_features, p5 

class VITA_TCOVIS(nn.Module):
    def __init__(self, num_tokens = 10, hidden_dim = 256):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Multi-Scale Backbone (ResNet50)
        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        # We need intermediate layers for the Pixel Decoder
        self.backbone = create_feature_extractor(resnet, return_nodes = {
            'layer2': 'layer2', # Stride 8  (High Res)
            'layer3': 'layer3', # Stride 16 (Medium)
            'layer4': 'layer4'  # Stride 32 (Low Res / Semantic)
        })
        
        # Pixel Decoder
        self.pixel_decoder = SimplePixelDecoder(input_dims = [512, 1024, 2048], hidden_dim = hidden_dim)
        
        # Transformer Decoder (Operates on tokens)
        self.object_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model = hidden_dim, nhead = 8, batch_first = True),
            num_layers = 3 
        )

        # Prediction Heads
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) 
        )
        self.temporal_module = TemporalAssociationModule(input_dim = hidden_dim)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        
        # Extract Multi-scale features
        ms_features = self.backbone(frames_flat)
        
        # Pixel Decoder: Fuse features
        # mask_features: (B*T, 256, H/8, W/8) -> High Resolution
        # transformer_features: (B*T, 256, H/32, W/32) -> Low Resolution
        mask_features, transformer_features = self.pixel_decoder(ms_features)
        
        # Transformer Step (using Low Res features for efficiency)
        features_seq = transformer_features.flatten(2).permute(0, 2, 1) # BxTx(HW)xC
        
        tokens = self.object_tokens.expand(B * T, -1, -1)
        out_tokens = self.transformer(tokens, features_seq)
        out_tokens = out_tokens.view(B, T, self.num_tokens, -1)
        
        # Mask Generation Step (using High Res features for quality)
        mask_feat_H, mask_feat_W = mask_features.shape[2], mask_features.shape[3]
        mask_features_aligned = mask_features.view(B, T, -1, mask_feat_H, mask_feat_W)
        
        all_masks, all_embs = [], []
        for t in range(T):
            toks_t = out_tokens[:, t]
            processed_toks = self.mask_head(toks_t)
            
            # Dot Product: Token (N, C) vs Feature Map (C, H/8, W/8)
            # Result: (N, H/8, W/8)
            masks_low = torch.einsum("nc, chw -> nhw", processed_toks, mask_features_aligned[:, t])
            
            # Upsample Stride 8 -> Stride 1 (Original Image Size)
            masks_low = masks_low.unsqueeze(0) 
            masks_up = F.interpolate(masks_low, size = (H, W), mode = 'bilinear', align_corners = False)
            
            all_masks.append(masks_up.squeeze(0))
            all_embs.append(self.temporal_module(toks_t))
            
        return torch.stack(all_masks, dim = 1), torch.stack(all_embs, dim = 1)