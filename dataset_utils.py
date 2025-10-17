from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from minecraft_dataset import MinecraftDataset
from read_data import action2word


def compute_dataset_splits(
    dataset: MinecraftDataset,
    *,
    subset_fraction: float = 0.1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Compute train/val/test index splits with the legacy fallback rules."""
    total_samples = len(dataset)
    indices = list(range(total_samples))

    if total_samples > 0 and 0 < subset_fraction < 1:
        subset_size = max(1, int(np.ceil(subset_fraction * total_samples)))
        indices = indices[:subset_size]
        total_samples = len(indices)

    if total_samples < 3:
        train_indices = indices[: max(1, total_samples - 2)]
        val_indices = indices[len(train_indices) : len(train_indices) + (1 if total_samples - len(train_indices) > 1 else 0)]
        test_indices = indices[len(train_indices) + len(val_indices) :]
    else:
        train_size = max(1, int(np.floor(train_ratio * total_samples)))
        val_size = max(1, int(np.floor(val_ratio * total_samples)))
        remaining = total_samples - train_size - val_size

        if remaining < 1:
            deficit = 1 - remaining
            if val_size - deficit >= 1:
                val_size -= deficit
            else:
                deficit -= (val_size - 1)
                val_size = 1
                train_size = max(1, train_size - deficit)
            remaining = 1

        test_size = remaining
        train_indices = indices[:train_size]
        val_start = train_size
        val_indices = indices[val_start : val_start + val_size]
        test_indices = indices[val_start + val_size : val_start + val_size + test_size]

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    return splits, indices


def create_dataset_subset(
    dataset: MinecraftDataset,
    indices: Sequence[int],
    *,
    context_examples: Optional[Sequence[Dict]] = None,
) -> MinecraftDataset:
    """Create a lightweight dataset clone containing only selected indices."""
    subset = MinecraftDataset.__new__(MinecraftDataset)
    subset.data_dir = dataset.data_dir
    subset.tokenizer = None
    subset.max_length = dataset.max_length
    subset.context_examples = list(context_examples or [])
    subset.history_length = dataset.history_length
    subset.max_creative_scenes = dataset.max_creative_scenes
    subset.device = dataset.device
    subset.num_workers = getattr(dataset, "num_workers", 0)
    subset.data_pairs = [dataset.data_pairs[i] for i in indices]
    return subset


def build_dataloader(
    dataset: MinecraftDataset,
    indices: Sequence[int],
    tokenizer,
    task_type: str,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    context_examples: Optional[Sequence[Dict]] = None,
) -> DataLoader:
    """Create a DataLoader for the requested task and split."""
    dataset_subset = create_dataset_subset(dataset, indices, context_examples=context_examples)
    collate_fn = (
        dataset_subset.collate_frame_reconstruction
        if task_type == "frame_reconstruction"
        else dataset_subset.collate_action_recognition
    )
    return DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=getattr(dataset_subset, "num_workers", 0),
        pin_memory=dataset_subset.device.type == "cuda",
    )


class Word2VecManager:
    """Lightweight Word2Vec-style embeddings trained on local action text."""

    def __init__(self, texts: Iterable[str], embedding_dim: int = 32, window_size: int = 2, epochs: int = 300, lr: float = 5e-3):
        self.window_size = window_size
        self.epochs = epochs  # kept for backwards compatibility
        self.lr = lr  # kept for backwards compatibility
        self._target_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim

        self.sentences = [self.tokenize(text) for text in texts if text and str(text).strip()]
        self.vocab: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self._cooc_matrix: Optional[np.ndarray] = None
        self._embeddings: Optional[np.ndarray] = None

        if self.sentences:
            self._build_vocab()
            self._build_cooccurrence()
            self._build_embeddings()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[A-Za-z0-9_]+", text.lower())]

    def _build_vocab(self) -> None:
        vocab = sorted({token for sent in self.sentences for token in sent})
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def _build_cooccurrence(self) -> None:
        vocab_size = len(self.vocab)
        cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for sent in self.sentences:
            token_ids = [self.word_to_idx[token] for token in sent if token in self.word_to_idx]
            if not token_ids:
                continue
            for idx, center_id in enumerate(token_ids):
                start = max(0, idx - self.window_size)
                end = min(len(token_ids), idx + self.window_size + 1)
                for context_idx in range(start, end):
                    if context_idx == idx:
                        continue
                    context_id = token_ids[context_idx]
                    cooc[center_id, context_id] += 1.0

        self._cooc_matrix = cooc

    def _build_embeddings(self) -> None:
        if self._cooc_matrix is None:
            return

        cooc = self._cooc_matrix
        if not np.any(cooc):
            vocab_size = len(self.vocab)
            embed_dim = min(self._target_embedding_dim, vocab_size)
            self.embedding_dim = embed_dim
            identity = np.eye(vocab_size, dtype=np.float32)
            self._embeddings = identity[:, :embed_dim]
            return

        cooc = np.log1p(cooc)
        u, s, _ = np.linalg.svd(cooc, full_matrices=False)
        embed_dim = min(self._target_embedding_dim, u.shape[1])
        self.embedding_dim = embed_dim
        embeddings = (u[:, :embed_dim] * s[:embed_dim])
        self._embeddings = embeddings.astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        if self._embeddings is None:
            return np.zeros(self.embedding_dim, dtype=float)

        tokens = self.tokenize(text)
        ids = [self.word_to_idx[token] for token in tokens if token in self.word_to_idx]
        if not ids:
            return np.zeros(self.embedding_dim, dtype=float)

        vectors = self._embeddings[ids]
        return vectors.mean(axis=0)

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)


def _enumerate_action_texts() -> List[str]:
    """Enumerate all discrete action combinations defined in read_data.action2word."""
    actions = []
    for straight in (0, 1, 2):
        for pan in (0, 1, 2):
            for jump in (0, 1):
                actions.append(action2word(np.array([straight, pan, jump], dtype=int)))
    return actions


def build_action_embedder(*_ignored, **__ignored) -> Word2VecManager:
    """Create an action-text embedder using all known discrete action combinations."""
    texts = _enumerate_action_texts()
    return Word2VecManager(texts)


def regex_fullmatch(text: str, pattern: str) -> bool:
    """Return True when text fully matches pattern using regex with fallback to literal match."""
    text = text.strip()
    pattern = pattern.strip()
    if pattern == "":
        return text == pattern

    flags = re.IGNORECASE | re.DOTALL

    try:
        compiled = re.compile(pattern, flags)
        if compiled.fullmatch(text):
            return True
    except re.error:
        compiled = None

    literal_compiled = re.compile(re.escape(pattern), flags)
    return bool(literal_compiled.fullmatch(text))


def compute_text_metrics(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    task_type: str,
    action_embedder: Optional[Word2VecManager] = None,
) -> Dict[str, object]:
    """Compute metrics for text generation tasks, including regex and similarity metrics."""
    if not targets:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "labels": [],
            "confusion_matrix": [],
            "regex_matches": 0,
            "strict_match_accuracy": 0.0,
            "reconstruction_accuracy": 0.0,
            "word2vec_cosine": 0.0,
            "word2vec_scores": [],
        }

    regex_matches: List[int] = []
    normalized_predictions: List[str] = []
    reconstruction_scores: List[float] = []
    cosine_scores: List[float] = []

    for pred, target in zip(predictions, targets):
        match = regex_fullmatch(pred, target)
        regex_matches.append(1 if match else 0)
        normalized_predictions.append(target if match else pred)

        if task_type == "frame_reconstruction":
            score = SequenceMatcher(None, target, pred).ratio()
            reconstruction_scores.append(score)
        elif task_type == "action_recognition" and action_embedder is not None:
            cosine_scores.append(action_embedder.cosine_similarity(pred, target))

    strict_accuracy = float(np.mean(regex_matches))

    metrics: Dict[str, object] = {
        "regex_matches": int(sum(regex_matches)),
        "strict_match_accuracy": strict_accuracy,
    }

    if task_type == "frame_reconstruction":
        reconstruction_accuracy = float(np.mean(reconstruction_scores)) if reconstruction_scores else 0.0
        metrics.update(
            {
                "accuracy": reconstruction_accuracy,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "labels": [],
                "confusion_matrix": [],
                "reconstruction_accuracy": reconstruction_accuracy,
                "reconstruction_scores": reconstruction_scores,
                "word2vec_cosine": 0.0,
                "word2vec_scores": [],
            }
        )
    else:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        label_space = sorted(set(targets + normalized_predictions))
        label_to_idx = {label: idx for idx, label in enumerate(label_space)}
        y_true = [label_to_idx[t] for t in targets]
        y_pred = [label_to_idx[p] for p in normalized_predictions]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        conf = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(label_space))),
        )

        cosine_mean = float(np.mean(cosine_scores)) if cosine_scores else 0.0

        metrics.update(
            {
                "accuracy": strict_accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "labels": label_space,
                "confusion_matrix": conf.astype(int).tolist(),
                "reconstruction_accuracy": 0.0,
                "word2vec_cosine": cosine_mean,
                "word2vec_scores": cosine_scores,
            }
        )

    return metrics


__all__ = [
    "compute_dataset_splits",
    "create_dataset_subset",
    "build_dataloader",
    "Word2VecManager",
    "build_action_embedder",
    "regex_fullmatch",
    "compute_text_metrics",
]
