from __future__ import annotations

import glob
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")
warnings.filterwarnings("ignore", message="CUDA initialization: Unexpected error from cudaGetDeviceCount().*")

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm

from dataset_utils import build_dataloader, compute_text_metrics
from utils_module import Utils


class ModelWrapper:
    """Wrapper for loading, training, and evaluating causal language models."""

    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize wrapper with target model name and device."""
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_lora = False
        self.checkpoint_dir: Optional[Path] = None
        self.log_dir: Optional[Path] = None

    def load_model(self, use_lora: bool = False, lora_config: Optional[Dict] = None):
        """Load model and tokenizer, optionally applying LoRA adapters."""
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if use_lora:
            default_config = {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            if lora_config:
                default_config.update(lora_config)

            lora_config_obj = LoraConfig(**default_config)
            self.model = get_peft_model(self.model, lora_config_obj)
            self.is_lora = True
            print("LoRA adapter applied (only adapter weights will be saved).")

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        print(f"Model loaded on {self.device}")
        return self

    def setup_training(self, task_name: str, run_name: Optional[str] = None):
        """Prepare checkpoint and logging directories for training."""
        if run_name is None:
            run_name = f"{task_name}_{self.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.checkpoint_dir = Utils.validate_path(Path("checkpoints") / run_name, create=True)
        self.log_dir = Utils.validate_path(Path("logs") / run_name, create=True)

        metadata = {
            "task_name": task_name,
            "run_name": run_name,
            "model_name": self.model_name,
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir),
            "is_lora": self.is_lora,
            "created_at": datetime.now().isoformat(),
        }
        return run_name, metadata

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Locate the latest checkpoint file for this run."""
        if not self.checkpoint_dir or not Path(self.checkpoint_dir).exists():
            return None

        checkpoints = glob.glob(str(Path(self.checkpoint_dir) / "checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        epochs = [int(Path(checkpoint).stem.split("_")[-1]) for checkpoint in checkpoints]
        latest_epoch = max(epochs)
        return Path(self.checkpoint_dir) / f"checkpoint_epoch_{latest_epoch}.pt"

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        task_name: str,
        use_wandb: bool = True,
    ):
        """Train the wrapped model and persist checkpoints/metadata."""
        run_name, metadata = self.setup_training(task_name)
        logger = Utils.setup_logging(self.log_dir, task_name)

        latest_checkpoint = self.find_latest_checkpoint()
        start_epoch = 0

        if latest_checkpoint:
            print(f"Found checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")

        if use_wandb:
            wandb.init(
                project=config.get("wandb_project", "minecraft-llm"),
                name=run_name,
                config={k: v for k, v in config.items() if not k.startswith("wandb")},
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["learning_rate"])
        num_training_steps = len(train_loader) * config["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.get("warmup_steps", 0),
            num_training_steps=num_training_steps,
        )

        best_val_loss = float("inf")

        for epoch in range(start_epoch, config["num_epochs"]):
            self.model.train()
            train_loss = 0.0

            with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config['num_epochs']}") as progress:
                for batch in progress:
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.get("max_grad_norm", 1.0))
                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()
                    progress.set_postfix({"loss": loss.item()})

                    if use_wandb:
                        wandb.log({"train_loss_step": loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            val_loss = self.evaluate_loss(val_loader)
            logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                    }
                )

            try:
                if self.is_lora:
                    adapter_path = Path(self.checkpoint_dir) / f"adapter_epoch_{epoch}"
                    self.model.save_pretrained(adapter_path)
                    state_path = Path(self.checkpoint_dir) / f"state_epoch_{epoch}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                        },
                        state_path,
                    )
                else:
                    checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                        },
                        checkpoint_path,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.is_lora:
                        best_adapter_path = Path(self.checkpoint_dir) / "best_lora_adapter"
                        self.model.save_pretrained(best_adapter_path)
                        logger.info(f"Best LoRA adapter saved with val_loss={val_loss:.4f}")
                    else:
                        best_model_path = Path(self.checkpoint_dir) / "best_model.pt"
                        torch.save(self.model.state_dict(), best_model_path)
                        logger.info(f"Best model saved with val_loss={val_loss:.4f}")
            except RuntimeError as error:
                logger.error(f"Failed to save checkpoint: {error}")
                print(f"WARNING: Could not save checkpoint. Error: {error}")

        if use_wandb:
            wandb.finish()

        metadata["training_completed"] = datetime.now().isoformat()
        metadata["best_val_loss"] = best_val_loss
        return metadata

    def evaluate_loss(self, dataloader: DataLoader):
        """Compute average loss across a dataloader."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, test_loader: DataLoader, max_new_tokens: int = 512):
        """Generate predictions for a test loader."""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                )

                for index, output in enumerate(outputs):
                    input_len = input_ids[index].shape[0]
                    generated = output[input_len:]
                    pred_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    predictions.append(pred_text)

                if "target_text" in batch:
                    targets.extend(batch["target_text"])

        return predictions, targets

    def evaluate_task(
        self,
        dataset,
        indices: Sequence[int],
        *,
        task_type: str,
        model_key: Optional[str] = None,
        context_examples: Optional[Sequence[Dict[str, Any]]] = None,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        action_embedder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run evaluation against specific dataset indices and compute metrics."""
        dataloader = build_dataloader(
            dataset,
            indices,
            self.tokenizer,
            task_type,
            batch_size=batch_size,
            shuffle=False,
            context_examples=context_examples,
        )
        predictions, targets = self.evaluate(dataloader, max_new_tokens=max_new_tokens)
        metrics = compute_text_metrics(
            predictions,
            targets,
            task_type=task_type,
            action_embedder=action_embedder,
        )
        metrics.update(
            {
                "model": model_key or self.model_name,
                "num_samples": len(targets),
                "num_context_examples": len(context_examples) if context_examples else 0,
                "predictions": predictions,
                "targets": targets,
            }
        )
        return metrics

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights or adapters from disk."""
        checkpoint = Path(checkpoint_path)
        if self.is_lora and checkpoint.is_dir():
            self.model = PeftModel.from_pretrained(self.model, checkpoint)
            print(f"LoRA adapter loaded from {checkpoint}")
        else:
            state = torch.load(checkpoint, map_location=self.device)
            payload = state["model_state_dict"] if "model_state_dict" in state else state
            self.model.load_state_dict(payload)
            print(f"Checkpoint loaded from {checkpoint}")
        return self


def main() -> None:
    """Quick smoke test that instantiates the wrapper (without loading weights)."""
    wrapper = ModelWrapper("dummy-model", device="cpu")
    print(f"Initialized ModelWrapper for {wrapper.model_name} on {wrapper.device}")


if __name__ == "__main__":
    main()
