import io
import cv2
import torch
import numpy as np
from PIL import Image
from clearml import Logger
import matplotlib.pyplot as plt

from inference import GlobalTracker
from utils import calculate_true_id_switches

def compute_j_and_f(pred_mask, gt_mask):
    # Region Similarity (J) - IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        J = 1.0 if intersection == 0 else 0.0
    else:
        J = intersection / union

    # Boundary Accuracy (F)
    # Convert masks to boundary maps
    def get_boundary(mask):
        # Get boundaries via morphological erosion (mask - eroded_mask)
        mask = mask.astype(np.uint8)
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = mask - eroded
        return boundary

    pred_boundary = get_boundary(pred_mask)
    gt_boundary = get_boundary(gt_mask)

    if gt_boundary.sum() == 0:
        F_score = 1.0 if pred_boundary.sum() == 0 else 0.0

    else:
        # Use a slight dilation for tolerance (standard practice in VOS benchmarks)
        dilated_gt = cv2.dilate(gt_boundary, np.ones((3, 3), np.uint8), iterations=1)
        dilated_pred = cv2.dilate(pred_boundary, np.ones((3, 3), np.uint8), iterations=1)

        # Precision: How many predicted boundary pixels are close to GT boundary?
        # Recall: How many GT boundary pixels are close to predicted boundary?
        tp_precision = np.logical_and(pred_boundary, dilated_gt).sum()
        tp_recall = np.logical_and(gt_boundary, dilated_pred).sum()

        precision = tp_precision / (pred_boundary.sum() + 1e-6)
        recall = tp_recall / (gt_boundary.sum() + 1e-6)

        F_score = 2 * precision * recall / (precision + recall + 1e-6)

    return J, F_score

def visualize_tracker(frames, tracker_results, title="Tracked"):
    T = len(frames)
    fig, axes = plt.subplots(2, T, figsize=(15, 6))
    plt.suptitle(title)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i)[:3] for i in range(10)]
    
    for t in range(T):
        img = frames[t].permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
        
        axes[0, t].imshow(img)
        axes[0, t].axis('off')
        axes[0, t].set_title(f"Frame {t}")
        
        mask_vis = np.zeros((img.shape[0], img.shape[1], 3))
        for tid, mask in tracker_results[t]:
            m = mask.detach().cpu().numpy() > 0.5
            if m.sum() == 0: continue
            
            color = np.array(colors[tid % 10])
            colored_mask = np.zeros_like(mask_vis)
            for c in range(3): colored_mask[:, :, c] = m * color[c]
            mask_vis = np.clip(mask_vis + colored_mask, 0, 1)
        
        axes[1, t].imshow(mask_vis)
        axes[1, t].axis('off')
        axes[1, t].set_title(f"Tracking")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig) 
    return Image.open(buf)

def run_base_vita_inference(model, frames):
    device = frames.device
    frames_gpu = frames.unsqueeze(0).to(device)
    with torch.no_grad(): pred_masks, _ = model(frames_gpu)
        
    T = frames.shape[0] 
    baseline_results = []
    
    for t in range(T):
        p_masks = torch.sigmoid(pred_masks[0, t])
        frame_results = []
        num_tokens = p_masks.shape[0]
        for i in range(num_tokens):
             if p_masks[i].mean() > 0.01:
                frame_results.append((i, p_masks[i])) 
        baseline_results.append(frame_results)
    return baseline_results

def calculate_frag_and_assa(pred_results, gt_masks):
    T = len(pred_results)
    num_gt_objs = gt_masks.shape[1]
    
    total_frag = 0
    total_assa = 0
    count_objs = 0
    
    for gt_id in range(num_gt_objs):
        # Extract GT trajectory
        gt_traj = gt_masks[:, gt_id].numpy()
        if gt_traj.sum() == 0: continue # Object doesn't exist
        
        # Find the best matching predicted Global ID
        best_pred_id = -1
        max_total_intersection = 0
        
        # Organize predictions by Track ID
        pred_maps = {} 
        for t in range(T):
            for pid, mask in pred_results[t]:
                if pid not in pred_maps: 
                    pred_maps[pid] = np.zeros((T, *mask.shape[-2:]))
                pred_maps[pid][t] = (mask.numpy() > 0.5)
        
        # Check overlaps to find the dominant track
        for pid, p_traj in pred_maps.items():
            intersection = np.logical_and(p_traj, gt_traj).sum()
            if intersection > max_total_intersection:
                max_total_intersection = intersection
                best_pred_id = pid
                
        if best_pred_id == -1: continue 
        
        count_objs += 1
        p_traj = pred_maps[best_pred_id]

        # Calculate AssA (Temporal IoU)
        # Intersection over Union of the entire 3D volume (Time x Height x Width)
        vol_inter = np.logical_and(p_traj, gt_traj).sum()
        vol_union = np.logical_or(p_traj, gt_traj).sum()
        total_assa += vol_inter / (vol_union + 1e-6)

        # Calculate Frag (Interruptions)
        # Check frame-level detection status
        is_matched = []
        for t in range(T):
            # Frame-level IoU > 0.5 considered a "hit"
            u = np.logical_or(p_traj[t], gt_traj[t]).sum()
            i = np.logical_and(p_traj[t], gt_traj[t]).sum()
            is_matched.append((i / (u + 1e-6)) > 0.1) 
            
        # Count transitions from Match -> Miss -> Match
        interruptions = 0
        
        # Find start and end of the object's life
        first_match = next((i for i, x in enumerate(is_matched) if x), None)
        last_match = next((i for i, x in enumerate(reversed(is_matched)) if x), None)
        
        if first_match is not None and last_match is not None:
            last_match = T - 1 - last_match
            # Scan between first and last detection
            for t in range(first_match, last_match):
                # If we have a miss (False) preceded by a match (True)
                if not is_matched[t] and is_matched[t-1]:
                    interruptions += 1
                    
        total_frag += interruptions

    avg_assa = total_assa / max(count_objs, 1)
    avg_frag = total_frag / max(count_objs, 1) # Average frag per object
    
    return avg_frag, avg_assa

def evaluate_and_log(model, dataset, device, epoch, beta=0.2, match_threshold=0.5):
    logger = Logger.current_logger()
    model.eval()
    
    # Visualization Clip (First clip in dataset)
    frames, _ = dataset[0]  
    frames_gpu = frames.unsqueeze(0).to(device) 

    # Run Baseline (VITA Only)
    baseline_results = run_base_vita_inference(model, frames.to(device))
    baseline_vis_img = visualize_tracker(frames, baseline_results, title="Baseline (VITA Segmentation Only)")
    if logger:
        logger.report_image(title="Visual Comparison", series="Baseline VITA Output", image=baseline_vis_img, iteration=epoch)

    # Run TCOVIS Tracking
    tracker = GlobalTracker(beta=beta, match_threshold=match_threshold)
    with torch.no_grad(): pred_masks, pred_embs = model(frames_gpu)
    tracker.reset()
    tracked_results = []
    T = frames.shape[0] 
    for t in range(T):
        p_masks = torch.sigmoid(pred_masks[0, t])
        p_embs = pred_embs[0, t]
        frame_results = tracker.update(p_masks, p_embs)
        tracked_results.append(frame_results)
        
    tracked_vis_img = visualize_tracker(frames, tracked_results, title="TCOVIS-Inspired VITA Tracking")
    if logger:
        logger.report_image(title="Visual Comparison", series="TCOVIS-Inspired VITA Output", image=tracked_vis_img, iteration=epoch)

    # Metric Calculation (J, F, IDSW) - Full Validation Set
    print(f"Calculating Metrics (J, F, IDSW)...")
    
    total_J = 0.0
    total_F = 0.0
    total_switches = 0
    count_objects = 0
    
    # Get GT masks for the metric calculation
    _, gt_masks = dataset[0] # Using same clip as visualization
    
    # Clean results for evaluation
    clean_results = []
    for t_res in tracked_results:
        clean_frame = []
        for tid, m in t_res: clean_frame.append((tid, m.cpu())) 
        clean_results.append(clean_frame)

    # Calculate ID Switches
    switches = calculate_true_id_switches(clean_results, gt_masks)
    total_switches += switches
    
    # Calculate J and F for each Ground Truth Object
    num_gt_objs = gt_masks.shape[1] # Channels
    T_len = gt_masks.shape[0]
    
    for gt_id in range(num_gt_objs):
        # We need to find which TRACK ID was assigned to this GT ID
        # Heuristic: Find the track that overlaps most with this GT object in the first frame it appears
        
        # Simplification: Calculate average J&F over all frames for the best matching track
        obj_J = 0
        obj_F = 0
        frames_present = 0
        
        for t in range(T_len):
            gt_m = gt_masks[t, gt_id].numpy()
            if gt_m.sum() == 0: continue # Object not in this frame
            
            frames_present += 1
            
            # Find best matching prediction in this frame
            best_iou = 0
            best_f = 0
            
            current_preds = clean_results[t]
            for _, pred_m in current_preds:
                p_m = (pred_m.numpy() > 0.5).astype(np.float32)
                j_score, f_score = compute_j_and_f(p_m, gt_m)
                
                if j_score > best_iou:
                    best_iou = j_score
                    best_f = f_score
            
            obj_J += best_iou
            obj_F += best_f
            
        if frames_present > 0:
            total_J += (obj_J / frames_present)
            total_F += (obj_F / frames_present)
            count_objects += 1

    avg_J = total_J / max(count_objects, 1)
    avg_F = total_F / max(count_objects, 1)
    
    print(f"Metrics -> J (IoU): {avg_J:.3f} | F (Boundary): {avg_F:.3f} | ID Switches: {total_switches}")

    switches = calculate_true_id_switches(clean_results, gt_masks)
    frag, assa = calculate_frag_and_assa(clean_results, gt_masks)
    
    print(f"Metrics -> IDSW: {switches} | Frag: {frag:.2f} | AssA: {assa:.3f}")

    if logger:
        logger.report_single_value(name = 'Final ID Switches', value = total_switches)
        logger.report_single_value(name = 'Region Similarity (J)', value = avg_J)
        logger.report_single_value(name = 'Boundary Accuracy (F)', value = avg_F)
        logger.report_single_value(name = 'ID Switches', value = switches)
        logger.report_single_value(name = 'Fragmentation', value = frag)
        logger.report_single_value(name = 'Association Accuracy', value = assa)
    
    return total_switches