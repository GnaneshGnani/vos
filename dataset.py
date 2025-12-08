import os
import cv2
import torch
import numpy as np
from PIL import Image
from collections import Counter 
from torch.utils.data import Dataset

class YouTubeVOSDataset(Dataset):
    def __init__(self, root_dir, split = 'train', num_frames = 5, img_size = (256, 448), max_objs = 5):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.max_objs = max_objs
        self.img_dir = os.path.join(root_dir, split, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, split, 'Annotations')
        self.videos = sorted([v for v in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, v))])
        
    def __len__(self): return len(self.videos)

    def __getitem__(self, idx):
        vid = self.videos[idx]
        img_path = os.path.join(self.img_dir, vid)
        mask_path = os.path.join(self.mask_dir, vid)
        
        frames_list = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])
        
        # Random clip sampling
        if len(frames_list) < self.num_frames:
            # Skip this video or pad with repeated frames
            # For now, let's repeat the last frame
            selected = frames_list[:]
            while len(selected) < self.num_frames:
                selected.append(frames_list[-1])

        elif len(frames_list) > self.num_frames:
            start = np.random.randint(0, len(frames_list) - self.num_frames)
            selected = frames_list[start : start + self.num_frames]

        else:
            selected = frames_list
        
        # Count all unique object IDs across all frames in the clip
        id_counts = Counter()
        for f in selected:
            m_path = os.path.join(mask_path, f.replace('.jpg', '.png'))

            if os.path.exists(m_path):
                m_temp = np.array(Image.open(m_path), dtype = np.uint8)
                m_ids = np.unique(m_temp[m_temp != 0])
                id_counts.update(m_ids)

        # Get the top N most frequent IDs
        target_ids = sorted([uid for uid, count in id_counts.most_common(self.max_objs)])

        id_to_channel = {uid: i for i, uid in enumerate(target_ids)}
        
        imgs, masks = [], []
        
        for f in selected:
            # Image Loading
            im_path = os.path.join(img_path, f)
            im = cv2.imread(im_path)

            # Convert BGR (OpenCV default) to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
            im = cv2.resize(im, (self.img_size[1], self.img_size[0]))
            
            # Normalize and Transpose (H,W,C) -> (C,H,W)
            imgs.append(np.transpose(im / 255.0, (2, 0, 1)))
            
            # Mask Loading
            m_path = os.path.join(mask_path, f.replace('.jpg', '.png'))
            if os.path.exists(m_path):
                m = np.array(Image.open(m_path).resize((self.img_size[1], self.img_size[0]), Image.NEAREST))
            else: 
                m = np.zeros(self.img_size, dtype = np.uint8)
            
            # Create Multi-channel Mask (consistent channels)
            obj_m = np.zeros((self.max_objs, *self.img_size), dtype = np.float32)
            
            # Only map IDs that were selected in the pre-scan
            current_frame_ids = np.unique(m[m != 0])
            for uid in current_frame_ids:
                if uid in id_to_channel:
                    ch_idx = id_to_channel[uid]
                    obj_m[ch_idx] = (m == uid).astype(np.float32)
            
            masks.append(obj_m)
            
        return torch.tensor(np.stack(imgs).astype(np.float32)), torch.tensor(np.stack(masks))