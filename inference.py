import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class GlobalTracker:
    def __init__(self, beta = 0.2, match_threshold = 0.5):
        self.beta = beta
        self.match_threshold = match_threshold
        
        # Memory Bank: {track_id: embedding_tensor}
        self.active_tracks = {} 

        # Store the last mask to calculate IoU/Location overlap
        self.last_masks = {}
        self.next_track_id = 1
        
    def reset(self):
        self.active_tracks = {}
        self.last_masks = {}
        self.next_track_id = 1

    def update(self, pred_masks, pred_embs):
        num_tokens = pred_masks.shape[0]
        pred_embs = F.normalize(pred_embs, p = 2, dim = -1)
        
        # Initialize if empty
        if len(self.active_tracks) == 0:
            results = []
            for i in range(num_tokens):
                # Only track confident masks
                if pred_masks[i].mean() > 0.01:
                    tid = self.next_track_id
                    self.active_tracks[tid] = pred_embs[i]
                    self.last_masks[tid] = pred_masks[i] # Save mask
                    self.next_track_id += 1
                    results.append((tid, pred_masks[i]))
            return results
        
        # Hungarian Matching Step
        track_ids = list(self.active_tracks.keys())
        track_embs = torch.stack([self.active_tracks[tid] for tid in track_ids]) 
        
        # Embedding Similarity: (Num_Tokens, Num_Tracks)
        sim_emb = torch.mm(pred_embs, track_embs.t())
        
        # IoU Similarity (Robustness Check)
        sim_iou = torch.zeros_like(sim_emb)
        for i in range(num_tokens):
            for j, tid in enumerate(track_ids):
                if tid in self.last_masks:
                    # Calculate simple Intersection over Union
                    prev_mask = self.last_masks[tid]
                    curr_mask = pred_masks[i]
                    
                    inter = (curr_mask * prev_mask).sum()
                    union = curr_mask.sum() + prev_mask.sum() - inter
                    sim_iou[i, j] = inter / (union + 1e-6)
        
        # Fused Similarity (70% Appearance, 30% Location)
        # This prevents assigning an ID to an object that jumped across the screen
        sim_matrix = 0.7 * sim_emb + 0.3 * sim_iou
        
        # Convert to Cost (Maximize Sim -> Minimize Cost)
        cost_matrix = 1.0 - sim_matrix.cpu().numpy()
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_tokens = set()
        results = []
        
        # Thresholding & Memory Update
        for r, c in zip(row_ind, col_ind):
            score = sim_matrix[r, c].item()
            tid = track_ids[c]
            
            # Check Threshold (tau_match)
            if score > self.match_threshold:
                # Update Memory: mu <- (1-beta)mu + beta * e
                curr_emb = pred_embs[r]
                old_mem = self.active_tracks[tid]
                new_mem = (1 - self.beta) * old_mem + self.beta * curr_emb
                self.active_tracks[tid] = F.normalize(new_mem, p = 2, dim = 0)
                
                # Update Spatial Memory
                self.last_masks[tid] = pred_masks[r]
                
                results.append((tid, pred_masks[r]))
                assigned_tokens.add(r)
                
        # Handle New Objects
        for i in range(num_tokens):
            if i not in assigned_tokens:
                if pred_masks[i].mean() > 0.05:
                    tid = self.next_track_id
                    self.active_tracks[tid] = pred_embs[i]
                    self.last_masks[tid] = pred_masks[i]
                    self.next_track_id += 1
                    results.append((tid, pred_masks[i]))
                    
        return results