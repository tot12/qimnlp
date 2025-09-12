# train_qinet_m_nlp.py
import math
import os
import random
import glob
import PyPDF2
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from qimnlp import QINetMNLP  # Using your qimnlp model


# ---------------------------
# 1. Real multilingual dataset from text files
# ---------------------------
class RealMultilingualDataset(Dataset):
    def __init__(self, data_dir="data", seq_len=128):
        super().__init__()
        self.samples = []
        self.lang_labels = []
        self.seq_len = seq_len
        self.label_to_id = {}
        self.id_to_label = {}
        label_id = 0
        
        # Load all three languages from .txt files in /data
        language_files = [
            ('kikuyu', 'kikuyu.txt'),
            ('lubukusu', 'lubukusu.txt'),
            ('luo', 'luo.txt'),
        ]
        for lang_label, filename in language_files:
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                print(f"[Warning] {file_path} not found, skipping.")
                continue
            self.label_to_id[lang_label] = label_id
            self.id_to_label[label_id] = lang_label
            label_id += 1
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Remove numerical characters (0-9) but preserve all Unicode/tonal characters
            text = ''.join(char for char in text if not char.isdigit())
            # Encode as UTF-8 bytes to preserve all characters
            byte_data = text.encode('utf-8')
            chunks = self.chunk_bytes(byte_data)
            self.samples.extend(chunks)
            self.lang_labels.extend([self.label_to_id[lang_label]] * len(chunks))
            print(f"Loaded {len(chunks)} samples for {lang_label}.")
        print(f"Total samples loaded: {len(self.samples)} from {len(self.label_to_id)} languages.")

    def chunk_bytes(self, byte_data):
        # Remove numericals (digit bytes: 48-57)
        filtered_bytes = bytes(b for b in byte_data if b < 48 or b > 57)
        # Split filtered byte data into chunks of self.seq_len
        chunks = [list(filtered_bytes[i:i+self.seq_len]) for i in range(0, len(filtered_bytes), self.seq_len)]
        return chunks

    # chunk_text is no longer used; replaced by chunk_bytes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get text sequence (make it 1 shorter to account for CLS token)
        x = self.samples[idx]
        target_len = self.seq_len - 1  # Reserve space for CLS token
        x = x + [0]*(target_len - len(x)) if len(x) < target_len else x[:target_len]
        x = torch.tensor(x, dtype=torch.long)

        # LM target = shifted input
        y_lm = x.clone()

        # Language ID
        lang_id = self.lang_labels[idx]

        # Simple word boundary detection (spaces and punctuation)
        seg_labels = torch.zeros(target_len, dtype=torch.long)
        for i, token in enumerate(x):
            if token in [32, 46, 44, 33, 63, 10, 13]:  # space, period, comma, !, ?, newline, return
                if i < target_len - 1:
                    seg_labels[i + 1] = 1  # mark next position as word start

        return x, y_lm, lang_id, seg_labels


# ---------------------------
# 2. Collate function
# ---------------------------
def collate_fn(batch):
    xs, ys, langs, segs = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    langs = torch.tensor(langs, dtype=torch.long)
    segs = torch.stack(segs)
    return xs, ys, langs, segs


# ---------------------------
# 3. Training loop
# ---------------------------
def train_qinet(
    num_epochs=10,
    batch_size=8,
    lr=1e-3,
    seq_len=128,
    vocab_size=256,
    data_dir="data",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Dataset / Dataloader
    full_data = RealMultilingualDataset(data_dir=data_dir, seq_len=seq_len)
    num_langs = len(full_data.label_to_id)
    
    # Split into train/val
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model - using your QINetMNLP
    model = QINetMNLP(vocab_size=vocab_size, num_langs=num_langs, d=64, L_max=seq_len, use_boundary_head=True).to(device)

    # Losses
    lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    lang_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training loop

    best_val_loss = float("inf")
    save_path = Path("checkpoints")
    save_path.mkdir(parents=True, exist_ok=True)
    patience = 5
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_lm, total_lang, total_seg = 0, 0, 0, 0
        for x, y_lm, langs, segs in train_loader:
            x, y_lm, langs, segs = x.to(device), y_lm.to(device), langs.to(device), segs.to(device)

            # Add CLS token at the beginning
            cls_token = torch.full((x.size(0), 1), 256, dtype=torch.long, device=device)
            x_with_cls = torch.cat([cls_token, x], dim=1)

            out = model(x_with_cls)
            lm_logits = out['lm_logits']
            lang_logits = out['lang_logits']
            seg_logits = out['boundary_logits']

            # Language modeling loss (skip CLS token in targets)
            lm_loss = lm_loss_fn(lm_logits[:, 1:-1, :].reshape(-1, vocab_size + 1), y_lm[:, 1:].reshape(-1))

            # Language ID loss
            lang_loss = lang_loss_fn(lang_logits, langs)

            # Segmentation loss (skip CLS token)
            seg_loss = seg_loss_fn(seg_logits[:, 1:, :].reshape(-1, 2), segs.reshape(-1))

            # Weighted sum
            loss = lm_loss + 0.5 * lang_loss + 0.5 * seg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_lm += lm_loss.item()
            total_lang += lang_loss.item()
            total_seg += seg_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_lm = total_lm / len(train_loader)
        avg_train_lang = total_lang / len(train_loader)
        avg_train_seg = total_seg / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_lm, val_lang, val_seg = 0, 0, 0, 0
            for x, y_lm, langs, segs in val_loader:
                x, y_lm, langs, segs = x.to(device), y_lm.to(device), langs.to(device), segs.to(device)

                # Add CLS token
                cls_token = torch.full((x.size(0), 1), 256, dtype=torch.long, device=device)
                x_with_cls = torch.cat([cls_token, x], dim=1)

                out = model(x_with_cls)
                lm_logits = out['lm_logits']
                lang_logits = out['lang_logits']
                seg_logits = out['boundary_logits']

                lm_loss = lm_loss_fn(lm_logits[:, 1:-1, :].reshape(-1, vocab_size + 1), y_lm[:, 1:].reshape(-1))
                lang_loss = lang_loss_fn(lang_logits, langs)
                seg_loss = seg_loss_fn(seg_logits[:, 1:, :].reshape(-1, 2), segs.reshape(-1))

                loss = lm_loss + 0.5 * lang_loss + 0.5 * seg_loss

                val_loss += loss.item()
                val_lm += lm_loss.item()
                val_lang += lang_loss.item()
                val_seg += seg_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_lm = val_lm / len(val_loader)
            avg_val_lang = val_lang / len(val_loader)
            avg_val_seg = val_seg / len(val_loader)

        print(
            f"Epoch {epoch:02d}: "
            f"TrainLoss={avg_train_loss:.4f} (LM={avg_train_lm:.4f}, Lang={avg_train_lang:.4f}, Seg={avg_train_seg:.4f}) | "
            f"ValLoss={avg_val_loss:.4f} (LM={avg_val_lm:.4f}, Lang={avg_val_lang:.4f}, Seg={avg_val_seg:.4f})"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_to_id': full_data.label_to_id,
                'id_to_label': full_data.id_to_label,
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, save_path / "qinet_m_nlp_best.pt")
            print(f"Saved best model with val_loss={avg_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                break

    return model, full_data.label_to_id


if __name__ == "__main__":
    model, label_mappings = train_qinet(
        num_epochs=50, 
        batch_size=8, 
        lr=1e-3, 
        seq_len=128,
        data_dir="data"
    )
    print("Training completed!")
    print(f"Language mappings: {label_mappings}")
