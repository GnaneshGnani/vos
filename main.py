import os
import glob
import torch
from clearml import Task
import torch.optim as optim
from torch.utils.data import DataLoader

from modules import VITA_TCOVIS
from dataset import YouTubeVOSDataset
from evaluate import evaluate_and_log
from utils import TCOVISCriterion, HungarianMatcher

def setup_run_directory(base_dir = "model"):
    os.makedirs(base_dir, exist_ok = True)
    existing_runs = glob.glob(os.path.join(base_dir, 'run_*'))
    
    # Determine the next run number
    run_numbers = []
    for r in existing_runs:
        try:
            num = int(r.split('_')[-1])
            run_numbers.append(num)
        except ValueError:
            continue
            
    next_run_num = max(run_numbers) + 1 if run_numbers else 1
    run_dir = os.path.join(base_dir, f'run_{next_run_num}')
    os.makedirs(run_dir, exist_ok = True)

    print(f"Saving checkpoints to: {run_dir}")

    return run_dir


def main():
    RUN_DIR = setup_run_directory()

    task = Task.init(project_name = 'VITA-TCOVIS Video Segmentation', 
                     task_name = f'TCOVIS-Inspired VITA Run {RUN_DIR.split("_")[-1]}',
                     tags = ['TCOVIS', 'VITA', 'Video Segmentation'])
    
    logger = task.get_logger()
    
    DATASET_ROOT = "./dataset"
    if not os.path.exists(os.path.join(DATASET_ROOT, 'train')):
        print("Error: Dataset not found. Run setup_commands.sh first.")
        return

    train_dataset = YouTubeVOSDataset(
        root_dir = DATASET_ROOT, split = 'train', 
        num_frames = 5, img_size = (256, 448), max_objs = 10
    )
    dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 8, pin_memory = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VITA_TCOVIS(num_tokens = 10, hidden_dim = 256).to(device)
    
    weight_dict = {'loss_mask': 1.0, 'loss_match': 1.0, 'loss_contrastive': 0.5}
    task.connect_configuration(
        {'num_tokens': 10, 'hidden_dim': 256, 'beta': 0.2, 'match_threshold': 0.5,
         'epochs': 5, 'batch_size': 2, 'learning_rate_head': 1e-4, 'learning_rate_backbone': 1e-5}
    )

    matcher = HungarianMatcher(cost_class = 1, cost_mask = 2, cost_dice = 2)
    criterion = TCOVISCriterion(matcher, weight_dict = weight_dict)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]

    optimizer = optim.AdamW(param_dicts, lr = 1e-4)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0
        
        for batch_idx, (frames, gt_masks) in enumerate(dataloader):
            iteration = epoch * len(dataloader) + batch_idx
            frames, gt_masks = frames.to(device), gt_masks.to(device)
            optimizer.zero_grad()
            
            pred_masks, pred_embs = model(frames)
            
            losses_dict, loss = criterion(pred_masks, pred_embs, gt_masks)
            
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Total: {loss.item():.4f} | "
                      f"Mask: {losses_dict['loss_mask']:.3f} | "
                      f"Match: {losses_dict['loss_match']:.3f} | "
                      f"Ctr: {losses_dict['loss_contrastive']:.3f}")
                
                # Log individual losses
                logger.report_scalar(title = 'Training Losses', series = 'Total Loss', value = loss.item(), iteration = iteration)
                logger.report_scalar(title = 'Training Losses', series = 'Mask Loss', value = losses_dict['loss_mask'].item(), iteration = iteration)
                logger.report_scalar(title = 'Training Losses', series = 'Matching Loss', value = losses_dict['loss_match'].item(), iteration = iteration)
                logger.report_scalar(title = 'Training Losses', series = 'Contrastive Loss', value = losses_dict['loss_contrastive'].item(), iteration = iteration)

        avg_loss = total_loss_epoch / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        logger.report_scalar(title = 'Epoch Metrics', series = 'Average Epoch Loss', value = avg_loss, iteration = epoch)
        
        checkpoint_path = os.path.join(RUN_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("\n--- Training Complete. Starting Final Evaluation ---")
    
    switches = evaluate_and_log(
        model = model, 
        dataset = train_dataset, 
        device = device, 
        epoch = epochs,
        beta = 0.2, 
        match_threshold = 0.5
    )

    final_model_path = os.path.join(RUN_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    task.update_output_model(model_path = final_model_path,
                             name = f'VITA_TCOVIS_Run_{RUN_DIR.split("_")[-1]}_Final',
                             tags = ['final', f'IDSW_{switches}'])
    
    task.close() 

if __name__ == "__main__":
    main()