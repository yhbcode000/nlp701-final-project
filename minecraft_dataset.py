"""
Minecraft Dataset loader for sequential frame prediction and action recognition tasks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import sys

# Add minecraft dataset path
sys.path.append('datasets/minecraft')
from read_data import action2word, voxel2word


class MinecraftDataset(Dataset):
    """Dataset for Minecraft voxel-based frame prediction and action recognition."""

    def __init__(self, data_dir: str = "datasets/minecraft/data", tokenizer=None, max_length: int = 512, context_examples=None):
        """
        Initialize Minecraft dataset.

        Args:
            data_dir: Directory containing sequential .npy files
            tokenizer: Tokenizer for text encoding (optional)
            max_length: Maximum sequence length for tokenization
            context_examples: List of example data pairs to use as context for in-context learning
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_pairs = []
        self.context_examples = context_examples or []

        if self.data_dir.exists():
            self.load_data()
        else:
            raise ValueError(f"Dataset directory '{data_dir}' not found!")

    def load_data(self):
        """Load sequential .npy files and create training pairs."""
        # Get all .npy files sorted by name
        npy_files = sorted(glob.glob(str(self.data_dir / "*.npy")))

        if len(npy_files) < 2:
            raise ValueError(f"Need at least 2 files for sequential pairs, found {len(npy_files)}")

        print(f"Loading {len(npy_files)} frames from {self.data_dir}")

        # Load all frames
        frames = []
        for npy_file in npy_files:
            data = np.load(npy_file, allow_pickle=True).item()
            frames.append(data)

        # Create sequential pairs: (frame_i, action_i) -> frame_(i+1)
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]

            # Convert voxel and action to text
            current_voxel_text = voxel2word(current_frame['voxel'])
            next_voxel_text = voxel2word(next_frame['voxel'])
            action_text = action2word(current_frame['action'])

            self.data_pairs.append({
                'x': current_voxel_text,  # Current frame voxel
                'y': action_text,          # Action taken
                'z': next_voxel_text,      # Next frame voxel
                'x_raw': current_frame['voxel'],  # Raw voxel data
                'y_raw': current_frame['action'],  # Raw action data
                'z_raw': next_frame['voxel']       # Raw next voxel data
            })

        print(f"Created {len(self.data_pairs)} training pairs")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """Get a single data pair."""
        item = self.data_pairs[idx]

        if self.tokenizer is None:
            # Return raw data
            return item

        # Return tokenized data (method depends on task)
        return {
            'x': item['x'],
            'y': item['y'],
            'z': item['z'],
            'x_raw': item['x_raw'],
            'y_raw': item['y_raw'],
            'z_raw': item['z_raw']
        }

    def get_dataloader_for_task(self, task_type: str, batch_size: int = 8,
                                shuffle: bool = True, tokenizer=None):
        """
        Create a dataloader for a specific task.

        Args:
            task_type: 'frame_reconstruction' or 'action_recognition'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            tokenizer: Tokenizer for encoding text

        Returns:
            DataLoader configured for the specified task
        """
        if task_type == 'frame_reconstruction':
            # Input: x + y, Output: z
            collate_fn = lambda batch: self.collate_frame_reconstruction(batch, tokenizer)
        elif task_type == 'action_recognition':
            # Input: x + z, Output: y
            collate_fn = lambda batch: self.collate_action_recognition(batch, tokenizer)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn)

    def collate_frame_reconstruction(self, batch, tokenizer):
        """Collate function for frame reconstruction task."""
        inputs = []
        targets = []

        # Create context from examples (in-context learning)
        context = ""
        if self.context_examples:
            context = "Here are some examples:\n\n"
            for i, ex in enumerate(self.context_examples[:3], 1):  # Use up to 3 examples
                context += f"Example {i}:\n"
                context += f"Current Frame:\n{ex['x']}\n"
                context += f"Action:\n{ex['y']}\n"
                context += f"Next Frame:\n{ex['z']}\n\n"
            context += "Now predict the next frame:\n\n"

        for item in batch:
            # Format with context: examples + current query
            input_text = context + f"Current Frame:\n{item['x']}\nAction:\n{item['y']}\nNext Frame:"
            target_text = item['z']

            inputs.append(input_text)
            targets.append(target_text)

        # Tokenize
        input_enc = tokenizer(inputs, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        target_enc = tokenizer(targets, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors='pt')

        # Create labels (input + target concatenated for causal LM)
        full_texts = [inp + tgt for inp, tgt in zip(inputs, targets)]
        full_enc = tokenizer(full_texts, padding=True, truncation=True,
                            max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': full_enc['input_ids'],
            'attention_mask': full_enc['attention_mask'],
            'labels': full_enc['input_ids'].clone(),
            'target_text': targets
        }

    def collate_action_recognition(self, batch, tokenizer):
        """Collate function for action recognition task."""
        inputs = []
        targets = []

        # Create context from examples (in-context learning)
        context_blocks = [
            "Respond with ONLY the action block (three lines in the format shown).",
            "The pattern always alternates as: Frame → Action → Frame → Action ...",
        ]
        if self.context_examples:
            for ex in self.context_examples:
                context_blocks.append(f"Frame:\n{ex['x']}")
                context_blocks.append(f"Action:\n{ex['y']}")
                context_blocks.append(f"Frame:\n{ex['z']}")
        context = "\n\n".join(context_blocks) + "\n\n"

        for item in batch:
            # Format with context: examples + current query
            input_text = (
                context
                + f"Frame:\n{item['x']}\n\n"
                + f"Frame:\n{item['z']}\n\n"
                + "Action:"
            )
            target_text = item['y'].strip()

            inputs.append(input_text)
            targets.append(target_text)

        # Tokenize
        input_enc = tokenizer(inputs, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        target_enc = tokenizer(targets, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors='pt')

        # Create labels
        full_texts = [inp + tgt for inp, tgt in zip(inputs, targets)]
        full_enc = tokenizer(full_texts, padding=True, truncation=True,
                            max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': full_enc['input_ids'],
            'attention_mask': full_enc['attention_mask'],
            'labels': full_enc['input_ids'].clone(),
            'target_text': targets
        }


# Test loading
if __name__ == "__main__":
    dataset = MinecraftDataset()
    print(f"\nDataset loaded with {len(dataset)} pairs")

    # Print first sample
    print("\n" + "="*80)
    print("SAMPLE DATA PAIR:")
    print("="*80)
    sample = dataset[0]
    print("\nX (Current Frame):")
    print(sample['x'])
    print("\nY (Action):")
    print(sample['y'])
    print("\nZ (Next Frame):")
    print(sample['z'])
    print("="*80)
