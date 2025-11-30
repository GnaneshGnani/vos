import io
import torch
import numpy as np
from PIL import Image
from clearml import Logger
import matplotlib.pyplot as plt

from inference import GlobalTracker
from utils import calculate_true_id_switches

def visualize_tracker(frames, tracker_results, title = "Tracked"):
    T = len(frames)

    fig, axes = plt.subplots(2, T, figsize = (15, 6))
    plt.suptitle(title)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i)[:3] for i in range(10)]
    
    for t in range(T):
        img = frames[t].permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0:
             img = (img * 255).astype(np.uint8)
        
        axes[0, t].imshow(img)
        axes[0, t].axis('off')
        axes[0, t].set_title(f"Frame {t}")
        
        mask_vis = np.zeros((img.shape[0], img.shape[1], 3))
        
        for tid, mask in tracker_results[t]:
            m = mask.detach().cpu().numpy() > 0.5
            if m.sum() == 0: continue
            
            color = np.array(colors[tid % 10])
            
            colored_mask = np.zeros_like(mask_vis)
            for c in range(3): 
                colored_mask[:, :, c] = m * color[c]
            
            mask_vis = np.clip(mask_vis + colored_mask, 0, 1)
        
        axes[1, t].imshow(mask_vis)
        axes[1, t].axis('off')
        axes[1, t].set_title(f"Tracking")
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    buf.seek(0)
    plt.close(fig) 
    
    return Image.open(buf)

def run_base_vita_inference(model, frames):
    device = frames.device
    frames_gpu = frames.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_masks, _ = model(frames_gpu)
        
    T = frames.shape[0] 
    baseline_results = []
    
    for t in range(T):
        p_masks = torch.sigmoid(pred_masks[0, t])
        frame_results = []
        num_tokens = p_masks.shape[0]

        # Assign ID based on token index (simulating inconsistent frame-to-frame IDs)
        for i in range(num_tokens):
             if p_masks[i].mean() > 0.01:
                frame_results.append((i, p_masks[i])) 

        baseline_results.append(frame_results)
        
    return baseline_results

def evaluate_and_log(model, dataset, device, epoch, beta = 0.2, match_threshold = 0.5):
    logger = Logger.current_logger()
    model.eval()
    
    # Get the fixed CPU Data for evaluation (the first clip)
    frames, gt_masks = dataset[0]  
    frames_gpu = frames.unsqueeze(0).to(device) 

    # BASELINE (VITA Segmentation Only)
    print("\nRunning Baseline Inference (VITA Only)...")
    baseline_results = run_base_vita_inference(model, frames.to(device))
    baseline_vis_img = visualize_tracker(frames, baseline_results, title = "Baseline (VITA Segmentation Only)")
    logger.report_image(title = "Visual Comparison", series = "Baseline VITA Output", image = baseline_vis_img, iteration = epoch)

    # TRAINED MODEL (VITA-TCOVIS Tracking)
    print("Running TCOVIS-Inspired Global Tracker Inference...")
    tracker = GlobalTracker(beta = beta, match_threshold = match_threshold)
    
    with torch.no_grad():
        pred_masks, pred_embs = model(frames_gpu)
        
    tracker.reset()
    tracked_results = []
    
    T = frames.shape[0] 
    for t in range(T):
        p_masks = torch.sigmoid(pred_masks[0, t])
        p_embs = pred_embs[0, t]
        frame_results = tracker.update(p_masks, p_embs)
        tracked_results.append(frame_results)
        
    tracked_vis_img = visualize_tracker(frames, tracked_results, title = "TCOVIS-Inspired VITA Tracking")
    logger.report_image(title = "Visual Comparison", series = "TCOVIS-Inspired VITA Output", image = tracked_vis_img, iteration = epoch)

    # METRIC CALCULATION
    clean_results = []
    for t_res in tracked_results:
        clean_frame = []
        for tid, m in t_res:
            clean_frame.append((tid, m.cpu())) 

        clean_results.append(clean_frame)

    # Uses the utility function from utils.py
    switches = calculate_true_id_switches(clean_results, gt_masks)
    
    print(f"Total ID Switches (vs Ground Truth): {switches}")

    # Log the final ID switch metric
    logger.report_single_value(name = 'Final ID Switches', value = switches)
    
    return switches