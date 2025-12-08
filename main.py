import os
import glob
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from modules import VITA_TCOVIS
from dataset import YouTubeVOSDataset
from evaluate import evaluate_and_log
from utils import TCOVISCriterion, HungarianMatcher

def get_args_parser():
    parser = argparse.ArgumentParser(description = 'VITA-TCOVIS Training')
    
    # --- ClearML Logging Args ---
    parser.add_argument('--enable_clearml', action = 'store_true', help = 'Enable ClearML logging')
    parser.add_argument('--project_name', default = 'VITA-TCOVIS Project', type = str)
    parser.add_argument('--task_name', default = 'TCOVIS Experiment', type = str)
    
    # Data & Training
    parser.add_argument('--dataset_root', default = './dataset', type = str)
    parser.add_argument('--run_dir', default = 'model', type = str)
    parser.add_argument('--epochs', default = 10, type = int)
    parser.add_argument('--batch_size', default = 4, type = int, help = 'Reduced default to 4 for A100 safety with gradients')
    parser.add_argument('--num_workers', default = 4, type = int)
    parser.add_argument('--lr_head', default = 1e-4, type = float)
    parser.add_argument('--lr_backbone', default = 1e-5, type = float)
    
    # Model Hyperparameters
    # FIX for Heuristics: Set this to 50 or 100 to avoid dropping objects
    parser.add_argument('--num_tokens', default = 10, type = int, help = 'Number of query tokens. Set high (e.g. 50) to capture all objects.')
    parser.add_argument('--hidden_dim', default = 256, type = int)
    parser.add_argument('--beta', default = 0.2, type = float, help = 'Momentum for memory update')
    
    # Loss Weights
    parser.add_argument('--w_mask', default = 5.0, type = float, help = 'BCE Mask Loss weight')
    parser.add_argument('--w_dice', default = 5.0, type = float, help = 'Dice Loss weight (CRITICAL for segmentation)')
    parser.add_argument('--w_match', default = 1.0, type = float, help = 'Supervised Matching weight')
    parser.add_argument('--w_ctr', default = 0.5, type = float, help = 'Contrastive weight')
    
    # Inference / Eval
    parser.add_argument('--match_threshold', default = 0.5, type = float)
    
    return parser

def setup_run_directory(base_dir = "model"):
    os.makedirs(base_dir, exist_ok = True)
    existing_runs = glob.glob(os.path.join(base_dir, 'run_*'))
    
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
    parser = get_args_parser()
    args = parser.parse_args()

    RUN_DIR = setup_run_directory(args.run_dir)

    logger = None
    if args.enable_clearml:
        try:
            from clearml import Task
            task = Task.init(
                project_name = args.project_name, 
                task_name = f'{args.task_name} Run {RUN_DIR.split("_")[-1]}',
                tags = ['TCOVIS', 'VITA', 'Video Segmentation']
            )
            task.connect(args)
            logger = task.get_logger()
            print(f"ClearML Task Initialized: {args.task_name}")
        except ImportError:
            print("ClearML not installed. Skipping logging.")
    else:
        print("ClearML logging disabled.")

    if not os.path.exists(os.path.join(args.dataset_root, 'train')):
        print("Error: Dataset not found.")
        return

    train_dataset = YouTubeVOSDataset(
        root_dir = args.dataset_root, split = 'train', 
        num_frames = 5, img_size = (256, 448), max_objs = args.num_tokens
    )
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True, 
        num_workers = args.num_workers, 
        pin_memory = True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VITA_TCOVIS(num_tokens = args.num_tokens, hidden_dim = args.hidden_dim).to(device)
    
    weight_dict = {
        'loss_mask': args.w_mask, 
        'loss_dice': args.w_dice, 
        'loss_match': args.w_match, 
        'loss_contrastive': args.w_ctr
    }

    matcher = HungarianMatcher(cost_class = 1, cost_mask = args.w_mask, cost_dice = args.w_dice)
    criterion = TCOVISCriterion(matcher, weight_dict = weight_dict)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr = args.lr_head)

    print(f"Starting training on {device} for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss_epoch = 0
        
        for batch_idx, (frames, gt_masks) in enumerate(dataloader):
            iteration = epoch * len(dataloader) + batch_idx
            frames, gt_masks = frames.to(device), gt_masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_masks, pred_embs = model(frames)
            
            # Loss Calculation
            losses_dict, loss = criterion(pred_masks, pred_embs, gt_masks)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            
            if batch_idx % 10 == 0:
                log_msg = (f"Epoch {epoch+1} | Batch {batch_idx} | Total: {loss.item():.4f} | "
                           f"Mask: {losses_dict['loss_mask']:.3f} | "
                           f"Dice: {losses_dict['loss_dice']:.3f} | "
                           f"Match: {losses_dict['loss_match']:.3f}")
                print(log_msg)
                
                if logger:
                    logger.report_scalar(title = 'Training Losses', series = 'Total Loss', value = loss.item(), iteration = iteration)
                    logger.report_scalar(title = 'Training Losses', series = 'Mask Loss', value = losses_dict['loss_mask'].item(), iteration = iteration)
                    logger.report_scalar(title = 'Training Losses', series = 'Dice Loss', value = losses_dict['loss_dice'].item(), iteration = iteration)
                    logger.report_scalar(title = 'Training Losses', series = 'Matching Loss', value = losses_dict['loss_match'].item(), iteration = iteration)

        # End of Epoch
        avg_loss = total_loss_epoch / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        if logger:
            logger.report_scalar(title = 'Epoch Metrics', series = 'Average Epoch Loss', value = avg_loss, iteration = epoch)
        
        # Save Checkpoint
        checkpoint_path = os.path.join(RUN_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("\n--- Training Complete. Starting Final Evaluation ---")
    
    switches = evaluate_and_log(
        model = model, 
        dataset = train_dataset, 
        device = device, 
        epoch = args.epochs,
        beta = args.beta, 
        match_threshold = args.match_threshold
    )

    final_model_path = os.path.join(RUN_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    if args.enable_clearml:
        try:
            from clearml import Task
            task = Task.current_task()
            if task:
                task.update_output_model(
                    model_path = final_model_path,
                    name = f'VITA_TCOVIS_Final_Model',
                    tags = ['final', f'IDSW_{switches}']
                )
                task.close()
        except ImportError:
            pass

if __name__ == "__main__":
    main()