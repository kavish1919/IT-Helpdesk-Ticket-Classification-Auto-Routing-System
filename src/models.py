"""
models.py  —  IT Helpdesk Ticket Classifier
============================================
Contains the PyTorch LSTM model and associated utilities for sequence
preparation, training, and evaluation.

CLASSES / FUNCTIONS EXPORTED
  TicketLSTM           — bidirectional-optional LSTM classifier
  build_vocab()        — builds word-index vocabulary from training texts
  texts_to_sequences() — converts cleaned texts to padded integer tensors
  get_class_weights()  — computes inverse-frequency class weights for CrossEntropyLoss
  train_one_epoch()    — single training epoch
  evaluate_lstm()      — runs model on a DataLoader, returns predictions
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
#  VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

PAD_IDX = 0   # index reserved for padding
UNK_IDX = 1   # index for out-of-vocabulary tokens


def build_vocab(train_texts: list, min_freq: int = 2) -> dict:
    """
    Build a {token: index} vocabulary from training texts.

    Special tokens:
      0  →  <PAD>  (used for sequence padding)
      1  →  <UNK>  (used for OOV tokens at inference time)

    Parameters
    ----------
    train_texts : list[str]  — cleaned, space-joined texts
    min_freq    : int        — tokens appearing fewer than this many times are mapped to <UNK>

    Returns
    -------
    dict  {token: int_index}
    """
    counter = Counter()
    for text in train_texts:
        counter.update(text.split())

    # Build vocab: index 0 = PAD, 1 = UNK, then all tokens with freq >= min_freq
    vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    print(f'Vocabulary size: {len(vocab):,} tokens  (min_freq={min_freq})')
    return vocab


def texts_to_sequences(texts: list, vocab: dict, max_len: int = 50) -> torch.LongTensor:
    """
    Convert a list of cleaned texts to a padded 2-D integer tensor.

    Steps per text:
      1. Split on whitespace
      2. Map each token to its vocab index (UNK_IDX if OOV)
      3. Truncate / pad to max_len

    Parameters
    ----------
    texts   : list[str]  — cleaned texts
    vocab   : dict       — {token: int_index}
    max_len : int        — fixed sequence length after padding/truncation

    Returns
    -------
    tensor  : LongTensor  shape (n_texts, max_len)
    lengths : LongTensor  shape (n_texts,)  — actual (pre-pad) lengths, capped at max_len
    """
    sequences, lengths = [], []
    for text in texts:
        tokens = text.split()[:max_len]
        ids    = [vocab.get(t, UNK_IDX) for t in tokens]
        length = len(ids)
        # Pad to max_len with PAD_IDX
        ids   += [PAD_IDX] * (max_len - length)
        sequences.append(ids)
        lengths.append(max(1, length))   # min length 1 to avoid packing errors

    return (torch.tensor(sequences, dtype=torch.long),
            torch.tensor(lengths,   dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class TicketDataset(Dataset):
    """Simple Dataset wrapping padded sequences, lengths, and labels."""

    def __init__(self, sequences: torch.Tensor, lengths: torch.Tensor,
                 labels: np.ndarray):
        self.sequences = sequences
        self.lengths   = lengths
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TicketLSTM(nn.Module):
    """
    LSTM-based text classifier for IT ticket category prediction.

    Architecture:
      Embedding  →  Dropout  →  LSTM (stacked)  →  Dropout  →  Linear

    The embedding layer can be:
      (a) randomly initialised  — default
      (b) initialised with pre-trained Word2Vec vectors  — call init_embeddings_from_w2v()

    Forward pass uses pack_padded_sequence so padding tokens do not contribute
    to the LSTM hidden state, which is the correct approach for variable-length
    sequences.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int, n_classes: int, dropout: float = 0.3):
        super().__init__()

        # Embedding layer: maps integer token IDs to dense vectors
        # padding_idx=0 means the PAD token always maps to a zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # LSTM: processes the token sequence and produces a hidden state
        # batch_first=True  → input shape is (batch, seq_len, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,   # dropout only between layers
            bidirectional=False,   # keeping unidirectional for simplicity
        )

        self.dropout = nn.Dropout(dropout)

        # Final linear layer: maps last hidden state to class logits
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : LongTensor  (batch, seq_len)  — padded token IDs
        lengths : LongTensor  (batch,)           — actual sequence lengths

        Returns
        -------
        logits : FloatTensor  (batch, n_classes)
        """
        # 1. Embed tokens → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # 2. Pack sequences so LSTM ignores padding positions
        packed = pack_padded_sequence(
            embedded, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False   # allow unsorted batches
        )

        # 3. Run LSTM; hidden shape is (n_layers, batch, hidden_dim)
        _, (hidden, _) = self.lstm(packed)

        # 4. Use the hidden state from the last layer
        out = self.dropout(hidden[-1])   # (batch, hidden_dim)

        # 5. Project to class logits
        return self.fc(out)              # (batch, n_classes)

    def init_embeddings_from_w2v(self, vocab: dict, w2v_model) -> None:
        """
        Copy Word2Vec vectors into the embedding layer for tokens that exist
        in the W2V vocabulary. Tokens not in W2V keep their random init.
        """
        n_copied = 0
        with torch.no_grad():
            for token, idx in vocab.items():
                if token in w2v_model.wv:
                    self.embedding.weight[idx] = torch.tensor(
                        w2v_model.wv[token], dtype=torch.float32
                    )
                    n_copied += 1
        print(f'Initialised {n_copied:,} / {len(vocab):,} embedding vectors from Word2Vec')

    def freeze_embeddings(self):
        """Freeze embedding layer — call at start of training."""
        self.embedding.weight.requires_grad = False

    def unfreeze_embeddings(self):
        """Unfreeze embedding layer — call after a few warm-up epochs."""
        self.embedding.weight.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS WEIGHTS  (for imbalanced datasets)
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    weight[c] = total_samples / (n_classes * count[c])

    Returns
    -------
    FloatTensor of shape (n_classes,)
    """
    counts  = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = len(y_train) / (n_classes * counts)
    print('Class weights:', np.round(weights, 3))
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: TicketLSTM, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """
    Run one full training epoch.

    Returns
    -------
    float — mean training loss for this epoch
    """
    model.train()
    total_loss = 0.0

    for sequences, lengths, labels in loader:
        sequences = sequences.to(device)
        lengths   = lengths.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences, lengths)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients (common in RNNs)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_lstm(model: TicketLSTM, loader: DataLoader,
                  device: torch.device) -> tuple:
    """
    Run inference on a DataLoader.

    Returns
    -------
    (y_true, y_pred) : two np.ndarrays of integer class labels
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            lengths   = lengths.to(device)
            logits    = model(sequences, lengths)
            preds     = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)
