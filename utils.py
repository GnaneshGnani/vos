import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class = 1.0, cost_mask = 1.0, cost_dice = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, pred_masks, gt_masks):
        B, num_queries, H, W = pred_masks.shape
        _, num_gt, _, _ = gt_masks.shape
        
        indices = []
        for b in range(B):
            out_mask = pred_masks[b].flatten(1).sigmoid() 
            gt_mask = gt_masks[b].flatten(1)
            
            cost_mask = -torch.mm(out_mask, gt_mask.t())
            
            numerator = 2 * torch.mm(out_mask, gt_mask.t())
            denominator = out_mask.sum(1).unsqueeze(1) + gt_mask.sum(1).unsqueeze(0) + 1e-6
            cost_dice = 1 - (numerator / denominator)
            
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = C.cpu().numpy()
            
            # Returns tuple (row_ind, col_ind)
            indices.append(linear_sum_assignment(C))
            
        return indices

class TCOVISCriterion(nn.Module):
    def __init__(self, matcher, weight_dict = None, temperature = 0.1, alpha = 5.0):
        super().__init__()
        self.matcher = matcher

        self.weight_dict = weight_dict if weight_dict else {
            'loss_mask': 5.0, 
            'loss_dice': 5.0, 
            'loss_match': 1.0, 
            'loss_contrastive': 0.5
        }

        self.temperature = temperature
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')

    def loss_masks(self, pred_masks, gt_masks, indices):
        loss_bce = 0.0
        loss_dice = 0.0

        num_pairs = 0
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0: continue
            
            src_masks = pred_masks[b, pred_idx]
            target_masks = gt_masks[b, gt_idx]

            loss_bce += self.bce_loss(src_masks, target_masks).mean()
            
            src_masks_sigmoid = src_masks.sigmoid()
            src_flat = src_masks_sigmoid.flatten(1)
            target_flat = target_masks.flatten(1)
            
            numerator = 2 * (src_flat * target_flat).sum(1)
            denominator = src_flat.sum(1) + target_flat.sum(1) + 1e-6
            loss_dice += (1 - numerator / denominator).mean()
            
            num_pairs += 1
            
        if num_pairs == 0: 
            return torch.tensor(0.0, device = pred_masks.device), torch.tensor(0.0, device = pred_masks.device)
        
        return loss_bce / num_pairs, loss_dice / num_pairs

    def loss_matching(self, pred_embs, indices_list):
        B, T, N, D = pred_embs.shape

        assert len(indices_list) == T, f"Expected {T} frame indices, got {len(indices_list)}"

        total_loss = 0.0
        num_frames_matched = 0
        
        for t in range(T - 1):
            curr_embs = F.normalize(pred_embs[:, t], p = 2, dim = -1)
            next_embs = F.normalize(pred_embs[:, t+1], p = 2, dim = -1)
            
            idx_curr_batch = indices_list[t]
            idx_next_batch = indices_list[t+1]
            
            for b in range(B):
                pred_i_curr, gt_i_curr = idx_curr_batch[b] 
                pred_i_next, gt_i_next = idx_next_batch[b]
                
                # Create maps from GT index (channel) -> Predicted Token index
                # This ensures we are linking the same GT object across frames.
                gt_to_pred_curr = {gt: pred for pred, gt in zip(pred_i_curr, gt_i_curr)}
                gt_to_pred_next = {gt: pred for pred, gt in zip(pred_i_next, gt_i_next)}
                
                common_gts = set(gt_to_pred_curr.keys()) & set(gt_to_pred_next.keys())
                if not common_gts: continue
                
                # Token indices in current frame (preds) that matched a GT object
                pred_indices = [gt_to_pred_curr[gt] for gt in common_gts]

                # Token indices in next frame (targets) that matched the SAME GT object
                target_indices = [gt_to_pred_next[gt] for gt in common_gts]
                
                targets = next_embs[b, target_indices]
                preds = curr_embs[b, pred_indices]
                
                sim_matrix = torch.mm(preds, targets.t())
                scaled_sim = self.alpha * sim_matrix
                # Labels must match the identity matrix (token_i in t matches token_i in t + 1)
                labels = torch.arange(len(common_gts), device = pred_embs.device)
                
                total_loss += F.cross_entropy(scaled_sim, labels)
                num_frames_matched += 1

        if num_frames_matched == 0:
            return torch.tensor(0.0, device = pred_embs.device, requires_grad = True)
            
        return total_loss / num_frames_matched

    def loss_contrastive(self, pred_embs, indices_list):
        B, T, N, D = pred_embs.shape
        total_loss = 0.0
        num_comparisons = 0
        
        for t in range(T - 1):
            curr_embs = pred_embs[:, t]
            next_embs = pred_embs[:, t + 1]
            
            idx_curr_batch = indices_list[t]
            idx_next_batch = indices_list[t + 1]
            
            for b in range(B):
                pred_i_curr, gt_i_curr = idx_curr_batch[b]
                pred_i_next, gt_i_next = idx_next_batch[b]
                
                # Create maps from GT index (channel) -> Predicted Token index
                gt_to_pred_curr = {gt: pred for pred, gt in zip(pred_i_curr, gt_i_curr)}
                gt_to_pred_next = {gt: pred for pred, gt in zip(pred_i_next, gt_i_next)}
                
                common_gts = set(gt_to_pred_curr.keys()) & set(gt_to_pred_next.keys())
                if not common_gts: continue
                
                anchors, positives = [], []
                for gt in common_gts:
                    anchors.append(curr_embs[b, gt_to_pred_curr[gt]])
                    positives.append(next_embs[b, gt_to_pred_next[gt]])
                
                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                anchors = F.normalize(anchors, p = 2, dim = -1)
                positives = F.normalize(positives, p = 2, dim = -1)
                
                logits = torch.mm(anchors, positives.t()) / self.temperature
                labels = torch.arange(len(anchors), device = pred_embs.device)
                
                total_loss += F.cross_entropy(logits, labels)
                num_comparisons += 1
                
        if num_comparisons == 0:
            return torch.tensor(0.0, device = pred_embs.device, requires_grad = True)
            
        return total_loss / num_comparisons

    def forward(self, pred_masks, pred_embs, gt_masks):
        B, T, N, H, W = pred_masks.shape
        
        indices_list = []
        mask_loss_accum = 0.0
        dice_loss_accum = 0.0 # Accumulator for Dice
        
        for t in range(T):
            indices = self.matcher(pred_masks[:, t].detach(), gt_masks[:, t])
            indices_list.append(indices)
            
            l_bce, l_dice = self.loss_masks(pred_masks[:, t], gt_masks[:, t], indices)
            mask_loss_accum += l_bce
            dice_loss_accum += l_dice
        
        # Average over time
        mask_loss_accum /= T
        dice_loss_accum /= T
        
        loss_ctr = self.loss_contrastive(pred_embs, indices_list)
        loss_match = self.loss_matching(pred_embs, indices_list)
        
        losses = {
            "loss_mask": mask_loss_accum,
            "loss_dice": dice_loss_accum,
            "loss_contrastive": loss_ctr,
            "loss_match": loss_match
        }
        
        # Final Weighted Sum
        total_loss = (losses['loss_mask'] * self.weight_dict['loss_mask'] + 
                      losses['loss_dice'] * self.weight_dict['loss_dice'] + 
                      losses['loss_match'] * self.weight_dict['loss_match'] + 
                      losses['loss_contrastive'] * self.weight_dict['loss_contrastive'])
        
        return losses, total_loss

def calculate_true_id_switches(pred_results, gt_masks):
    id_switches = 0
    gt_assignments = {} 
    
    T = len(pred_results)
    
    for t in range(T):
        current_preds = {tid: mask for tid, mask in pred_results[t]}
        
        # Iterate over active Ground Truth objects using channel index (0-4) as ID
        for gt_id in range(gt_masks.shape[1]): # gt_id is the consistent channel index
            gt_mask_t = gt_masks[t, gt_id]
            if gt_mask_t.sum() == 0: continue 
            
            best_iou = 0
            best_pred_id = -1
            
            gt_mask_t = gt_mask_t.cpu()

            for pred_id, pred_mask in current_preds.items():
                pred_mask = pred_mask.cpu() 
                
                intersection = (pred_mask * gt_mask_t).sum()
                union = pred_mask.sum() + gt_mask_t.sum() - intersection
                iou = intersection / (union + 1e-6)
                
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id
            
            if best_pred_id != -1:
                # Check for an ID switch for this GT object (channel index)
                if gt_id in gt_assignments:
                    if gt_assignments[gt_id] != best_pred_id:
                        id_switches += 1
                gt_assignments[gt_id] = best_pred_id
                
    return id_switches