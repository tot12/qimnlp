import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from qimnlp import QINetMNLP
from multilingual_trainer import RealMultilingualDataset, collate_fn
import PyPDF2

def adaptive_finetune(
    base_model_path,
    finetune_data_path,
    save_path,
    num_epochs=10,
    batch_size=8,
    lr=5e-5,  # Lower learning rate for stability
    seq_len=128,
    l2_lambda=5e-4,  # Stronger regularization
    device="cuda" if torch.cuda.is_available() else "cpu",
    multilingual_data_path=None,  # Optionally mix in some multilingual data
    multilingual_mix_ratio=0.1,  # 10% of each batch from multilingual data
):
    # Load base model
    checkpoint = torch.load(base_model_path, map_location=device)
    model = QINetMNLP(
        vocab_size=256,
        num_langs=len(checkpoint['label_to_id']),
        d=64,
        L_max=seq_len,
        use_boundary_head=True
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    # Save a copy of the original parameters for L2 regularization
    orig_params = {k: v.clone().detach() for k, v in model.named_parameters()}

    # Map language to PDF file
    import os
    def get_pdf_for_language(language):
        language = language.lower()
        if language == 'luo':
            return os.path.join('data', 'luo.pdf')
        elif language in ['bukusu', 'lubukusu']:
            return os.path.join('data', 'lubukusu.pdf')
        elif language == 'kikuyu':
            return os.path.join('data', '01_Genesis.pdf')
        else:
            raise ValueError(f"Unknown language: {language}")

    def pdf_to_byte_chunks(pdf_path, seq_len):
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        # Remove numericals (0-9) but preserve all Unicode/tonal characters
        text = ''.join(char for char in text if not char.isdigit())
        byte_data = text.encode('utf-8')
        # Remove digit bytes (48-57)
        filtered_bytes = bytes(b for b in byte_data if b < 48 or b > 57)
        # Split into chunks
        return [list(filtered_bytes[i:i+seq_len]) for i in range(0, len(filtered_bytes), seq_len)]

    # If language is specified, use the mapped PDF file
    if language is not None:
        pdf_path = get_pdf_for_language(language)
        chunks = pdf_to_byte_chunks(pdf_path, seq_len)
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, chunks):
                self.chunks = chunks
            def __len__(self):
                return len(self.chunks)
            def __getitem__(self, idx):
                x = self.chunks[idx]
                target_len = seq_len - 1
                x = x + [0]*(target_len - len(x)) if len(x) < target_len else x[:target_len]
                x = torch.tensor(x, dtype=torch.long)
                y_lm = x.clone()
                lang_id = 0
                seg_labels = torch.zeros(target_len, dtype=torch.long)
                for i, token in enumerate(x):
                    if token in [32, 46, 44, 33, 63, 10, 13]:
                        if i < target_len - 1:
                            seg_labels[i + 1] = 1
                return x, y_lm, lang_id, seg_labels
        dataset = SimpleDataset(chunks)
    elif finetune_data_path.lower().endswith('.pdf'):
        chunks = pdf_to_byte_chunks(finetune_data_path, seq_len)
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, chunks):
                self.chunks = chunks
            def __len__(self):
                return len(self.chunks)
            def __getitem__(self, idx):
                x = self.chunks[idx]
                target_len = seq_len - 1
                x = x + [0]*(target_len - len(x)) if len(x) < target_len else x[:target_len]
                x = torch.tensor(x, dtype=torch.long)
                y_lm = x.clone()
                lang_id = 0
                seg_labels = torch.zeros(target_len, dtype=torch.long)
                for i, token in enumerate(x):
                    if token in [32, 46, 44, 33, 63, 10, 13]:
                        if i < target_len - 1:
                            seg_labels[i + 1] = 1
                return x, y_lm, lang_id, seg_labels
        dataset = SimpleDataset(chunks)
    else:
        dataset = RealMultilingualDataset(data_dir=finetune_data_path, seq_len=seq_len)
    if multilingual_data_path:
        multilingual_dataset = RealMultilingualDataset(data_dir=multilingual_data_path, seq_len=seq_len)
    else:
        multilingual_dataset = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Losses and optimizer
    lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    lang_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for x, y_lm, langs, segs in loader:
            # Optionally mix in some multilingual data for stability
            if multilingual_dataset and multilingual_mix_ratio > 0:
                m_size = int(batch_size * multilingual_mix_ratio)
                if m_size > 0:
                    m_indices = torch.randperm(len(multilingual_dataset))[:m_size]
                    m_batch = [multilingual_dataset[i] for i in m_indices]
                    mx, my_lm, mlangs, msegs = collate_fn(m_batch)
                    # Concatenate with main batch
                    x = torch.cat([x, mx], dim=0)
                    y_lm = torch.cat([y_lm, my_lm], dim=0)
                    langs = torch.cat([langs, mlangs], dim=0)
                    segs = torch.cat([segs, msegs], dim=0)

            x, y_lm, langs, segs = x.to(device), y_lm.to(device), langs.to(device), segs.to(device)
            cls_token = torch.full((x.size(0), 1), 256, dtype=torch.long, device=device)
            x_with_cls = torch.cat([cls_token, x], dim=1)
            out = model(x_with_cls)
            lm_logits = out['lm_logits']
            lang_logits = out['lang_logits']
            seg_logits = out['boundary_logits']

            lm_loss = lm_loss_fn(lm_logits[:, 1:-1, :].reshape(-1, 257), y_lm[:, 1:].reshape(-1))
            lang_loss = lang_loss_fn(lang_logits, langs)
            seg_loss = seg_loss_fn(seg_logits[:, 1:, :].reshape(-1, 2), segs.reshape(-1))

            # L2 regularization to original weights
            l2_reg = 0.0
            for name, param in model.named_parameters():
                l2_reg += ((param - orig_params[name]) ** 2).sum()
            l2_reg = l2_lambda * l2_reg

            loss = lm_loss + 0.5 * lang_loss + 0.5 * seg_loss + l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

        # Data diversity check (warn if output is repetitive)
        if epoch == 0:
            sample_out = model.generate(torch.full((1, seq_len), 1, dtype=torch.long, device=device), max_new_tokens=50)
            sample_text = ''.join([chr(tok) for tok in sample_out[0].tolist() if tok < 256])
            if len(set(sample_text)) < 5:
                print("[Warning] Model output is low diversity. Check your data and parameters.")

    # Save adapted model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_id': checkpoint['label_to_id'],
        'id_to_label': checkpoint['id_to_label']
    }, save_path)
    print(f"Adapted model saved to {save_path}")

# Example usage:
# adaptive_finetune(
#     base_model_path='checkpoints/qinet_m_nlp_best.pt',
#     finetune_data_path='data/luo',
#     save_path='luo/qinet_m_nlp_adapted.pt'
# )
