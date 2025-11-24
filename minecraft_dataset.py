"""
Minecraft Dataset loader for sequential frame prediction and action recognition tasks.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add minecraft dataset path
sys.path.append('datasets/minecraft')
from read_data import action2word, voxel2word

warnings.filterwarnings("ignore", message="CUDA initialization: Unexpected error from cudaGetDeviceCount().*")


class MinecraftDataset(Dataset):
    """Dataset for Minecraft voxel-based frame prediction and action recognition."""

    def __init__(
        self,
        data_dir: str = "datasets/minecraft/data",
        tokenizer=None,
        max_length: int = 1024,
        context_examples=None,
        history_length: int = 10,
        max_creative_scenes: int = 10,
        device: str | torch.device | None = None,
        num_workers: int | None = None,
        voxel_size: int = 5,
    ):
        """
        Initialize Minecraft dataset.

        Args:
            data_dir: Directory containing sequential .npy files
            tokenizer: Tokenizer for text encoding (optional)
            max_length: Maximum sequence length for tokenization
            context_examples: List of example data pairs to use as context for in-context learning
            history_length: Number of sequential frames to include when building history strings
            max_creative_scenes: Limit on how many creative episodes to load (default 10)
            device: Target torch device for batched tensors (defaults to CUDA when available)
            num_workers: Number of DataLoader workers (defaults to half the CPU cores)
            voxel_size: Side length of the cubic voxel window to convert to text (passed to voxel2word)
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_pairs = []
        self.context_examples = context_examples or []
        self.max_creative_scenes = max_creative_scenes
        self.history_length = max(1, history_length)
        self.voxel_size = voxel_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        cpu_count = os.cpu_count() or 1
        default_workers = max(1, cpu_count // 2)
        self.num_workers = num_workers if num_workers is not None else default_workers

        if self.data_dir.exists():
            self.load_data()
        else:
            raise ValueError(f"Dataset directory '{data_dir}' not found!")

    def load_data(self):
        """Load sequential .npy files and create training pairs with history strings."""
        creative_dirs = sorted(
            p for p in self.data_dir.rglob("*")
            if p.is_dir() and p.name.startswith("creative:")
        )

        if self.max_creative_scenes is not None:
            creative_dirs = creative_dirs[: self.max_creative_scenes]

        if not creative_dirs:
            raise ValueError(
                f"No creative episodes found under '{self.data_dir}'. "
                "Expected directories like seq0-49/creative:0/."
            )

        total_frames = 0
        for creative_dir in creative_dirs:
            npy_files = sorted(creative_dir.glob("*.npy"))

            if len(npy_files) < 2:
                print(f"Skipping {creative_dir} (only {len(npy_files)} frame)")
                continue

            print(f"Loading {len(npy_files)} frames from {creative_dir}")

            frames = [np.load(path, allow_pickle=True).item() for path in npy_files]
            total_frames += len(frames)

            frame_texts = [voxel2word(frame["voxel"], size=self.voxel_size) for frame in frames]
            frame_raws = [frame["voxel"] for frame in frames]
            action_texts = [action2word(frame["action"]) for frame in frames[:-1]]
            action_raws = [frame["action"] for frame in frames[:-1]]

            if len(frames) <= self.history_length:
                print(
                    f"Skipping episode {creative_dir} for history_length {self.history_length}: "
                    f"requires at least {self.history_length + 1} frames."
                )
                continue

            for idx in range(self.history_length - 1, len(frames) - 1):
                x_text = frame_texts[idx]
                z_text = frame_texts[idx + 1]
                y_text = action_texts[idx]

                start_index = idx - self.history_length + 1
                history_length_used = self.history_length

                self.data_pairs.append({
                    "x": x_text,
                    "y": y_text,
                    "z": z_text,
                    "x_raw": frame_raws[idx],
                    "y_raw": action_raws[idx],
                    "z_raw": frame_raws[idx + 1],
                    "episode": creative_dir.name,
                    "history_reconstruction": self._build_frame_action_history(
                        frame_texts, action_texts, idx
                    ),
                    "history_action": self._build_frame_frame_action_history(
                        frame_texts, action_texts, idx
                    ),
                    "history_start_index": start_index,
                    "history_length_used": history_length_used,
                })

        if len(self.data_pairs) == 0:
            raise ValueError(
                f"No sequential frame pairs created from '{self.data_dir}'. "
                "Ensure each creative directory contains at least two frames."
            )

        print(f"Loaded {total_frames} frames across {len(creative_dirs)} episodes")
        print(f"Created {len(self.data_pairs)} training pairs")

    def _build_frame_action_history(self, frame_texts, action_texts, idx: int) -> str:
        """Construct frame/action alternating history ending with the current frame and action."""
        start = idx - self.history_length + 1
        parts = []
        step = 1
        for position in range(start, idx + 1):
            parts.append(f"Frame {step}:\n{frame_texts[position]}")
            parts.append(f"Action {step}:\n{action_texts[position]}")
            step += 1
        return "\n\n".join(parts).strip()

    def _build_frame_frame_action_history(self, frame_texts, action_texts, idx: int) -> str:
        """Construct frame/frame/action history ending with the current and next frames."""
        start = idx - self.history_length + 1
        parts = []
        step = 1
        for position in range(start, idx):
            parts.append(f"Frame {step}:\n{frame_texts[position]}")
            parts.append(f"Frame {step} Next:\n{frame_texts[position + 1]}")
            parts.append(f"Action {step}:\n{action_texts[position]}")
            step += 1
        parts.append(f"Frame {step}:\n{frame_texts[idx]}")
        parts.append(f"Frame {step} Next:\n{frame_texts[idx + 1]}")
        return "\n\n".join(parts).strip()

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

        num_workers = getattr(self, "num_workers", 0)
        device = getattr(self, "device", torch.device("cpu"))

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    def collate_frame_reconstruction(self, batch, tokenizer):
        """Collate function for frame reconstruction task."""
        inputs = []
        targets = []

        # Create context from examples (in-context learning)
        instruction_lines = [
            "Predict the next frame using ONLY the pipe-delimited grid format.",
            "Return the same number of rows as the input frame and do NOT add labels, reasoning, or commentary.",
            "DO NOT think aloud—output the grid only, nothing else.",
            "Do not prepend headings or words like 'Prediction'; just the grid.",
            "Keep rows pipe-delimited with the same row/column count as shown in the history frames."
        ]

        context_sections = ["\n".join(instruction_lines)]
        if self.context_examples:
            example_lines = ["Here are some examples using history chains:"]
            for i, ex in enumerate(self.context_examples[:3], 1):  # Use up to 3 examples
                history_text = ex.get("history_reconstruction") or (
                    f"Frame:\n{ex['x']}\nAction:\n{ex['y']}"
                )
                example_lines.append(f"Example {i} History:\n{history_text}")
                example_lines.append(f"Example {i} Next Frame:\n{ex['z']}\n")
            example_lines.append("Now predict the next frame given the history:")
            context_sections.append("\n".join(example_lines))

        context = "\n\n".join(context_sections) + ("\n\n" if context_sections else "")

        for item in batch:
            # Format with context: examples + current query
            history_text = item.get("history_reconstruction") or (
                f"Frame:\n{item['x']}\nAction:\n{item['y']}"
            )
            input_text = context + f"History:\n{history_text}\nNext Frame:"
            target_text = item['z']

            inputs.append(input_text)
            targets.append(target_text)

        # Create labels (input + target concatenated for causal LM)
        if tokenizer is None:
            raise ValueError("Tokenizer required for frame reconstruction task.")

        full_texts = [inp + tgt for inp, tgt in zip(inputs, targets)]
        full_enc = tokenizer(full_texts, padding=True, truncation=True,
                            max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': full_enc['input_ids'],
            'attention_mask': full_enc['attention_mask'],
            'labels': full_enc['input_ids'].clone(),
            'prompt_text': inputs,
            'target_text': targets
        }

    def collate_action_recognition(self, batch, tokenizer):
        """Collate function for action recognition task."""
        inputs = []
        targets = []

        # Create context from examples (in-context learning)
        context_blocks = [
            "Respond with ONLY the three-line action block in the exact format shown.",
            "Each line must be of the form `category: value` matching the provided options.",
            "History entries alternate `Frame` → `Frame Next` → `Action` for prior transitions.",
            "Do NOT include explanations, reasoning, labels, or extra text before or after the block.",
            "DO NOT think aloud—output the three lines only, nothing else.",
            "Do not add prefixes like 'Prediction'; respond with just the three lines.",
            "Exactly three lines are allowed:\nstraight: <noop|forward|backward>\npan: <noop|left|right>\njump: <noop|jump>."
        ]
        if self.context_examples:
            for ex in self.context_examples[:3]:
                history_text = ex.get("history_action")
                if history_text:
                    context_blocks.append(f"History:\n{history_text}")
                    context_blocks.append(f"Action:\n{ex['y']}")
                else:
                    context_blocks.append(f"Frame:\n{ex['x']}")
                    context_blocks.append(f"Frame:\n{ex['z']}")
                    context_blocks.append(f"Action:\n{ex['y']}")
        context = "\n\n".join(context_blocks) + "\n\n"

        for item in batch:
            # Format with context: examples + current query
            history_text = item.get("history_action")
            if history_text:
                input_text = context + f"History:\n{history_text}\nAction:"
            else:
                input_text = (
                    context
                    + f"Frame:\n{item['x']}\n\n"
                    + f"Frame:\n{item['z']}\n\n"
                    + "Action:"
                )
            target_text = item['y'].strip()

            inputs.append(input_text)
            targets.append(target_text)

        # Create labels
        if tokenizer is None:
            raise ValueError("Tokenizer required for action recognition task.")

        full_texts = [inp + tgt for inp, tgt in zip(inputs, targets)]
        full_enc = tokenizer(full_texts, padding=True, truncation=True,
                            max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': full_enc['input_ids'],
            'attention_mask': full_enc['attention_mask'],
            'labels': full_enc['input_ids'].clone(),
            'prompt_text': inputs,
            'target_text': targets
        }


# Test loading
if __name__ == "__main__":
    dataset = MinecraftDataset(max_creative_scenes=2)
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
    print("\nHistory (Frame→Action chain):")
    print(sample.get('history_reconstruction'))
    print("\nHistory (Frame/Frame/Action pattern):")
    print(sample.get('history_action'))
    print("="*80)
