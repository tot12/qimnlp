def detect_language(model, prompt, label_mappings, device="cpu"):
    """
    Detect the language of a given prompt using the model's language head.
    Returns (language_name, confidence, all_confidences).
    """
    if not prompt.strip():
        return "unknown", 0.0, {}

    # Convert prompt to bytes (same as generation) and add CLS token
    prompt_bytes = list(prompt.encode('utf-8'))
    prompt_bytes = [b for b in prompt_bytes if b < 48 or b > 57]  # Filter numericals
    if not prompt_bytes:
        return "unknown", 0.0, {}

    # Add CLS token (256) at the beginning, just like in training
    x = torch.tensor([[256] + prompt_bytes], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, dict) and 'lang_logits' in out:
            lang_logits = out['lang_logits']
            probs = torch.softmax(lang_logits, dim=-1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()
            
            # Get all language confidences
            all_confidences = {}
            if isinstance(label_mappings, dict):
                # Direct language mapping: {'kikuyu': 0, 'lubukusu': 1, 'luo': 2}
                id_to_label = {v: k for k, v in label_mappings.items()}
                for lang_id in range(probs.shape[1]):
                    lang_name = id_to_label.get(lang_id, f"lang_{lang_id}")
                    all_confidences[lang_name] = probs[0, lang_id].item()
                pred_lang = id_to_label.get(pred_id, f"lang_{pred_id}")
            else:
                # Fallback if no label mappings available
                for lang_id in range(probs.shape[1]):
                    lang_name = f"lang_{lang_id}"
                    all_confidences[lang_name] = probs[0, lang_id].item()
                pred_lang = f"lang_{pred_id}"
                
            return pred_lang, confidence, all_confidences
        else:
            return "unknown", 0.0, {}
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------
# Vocab utilities (no hardcoding, only model-trained vocab is used)
# --------------------------------------------------------------------

def get_language_specific_chars(label_mappings):
    return set(label_mappings.values())


def is_valid_for_language(char, valid_chars):
    return char in valid_chars


def clean_language_specific_text(text, valid_chars):
    return "".join(ch for ch in text if ch in valid_chars)


# --------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------

def load_model(model_class, checkpoint_path, device="cpu"):
    """
    Load a trained model checkpoint and its label mappings.
    Assumes checkpoint is a dict with:
      - "model_state": model weights
      - "label_mappings": {int: str}
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint["model_state"]

    # Infer model parameters from state_dict
    if 'lang_head.weight' in state_dict:
        num_langs = state_dict['lang_head.weight'].shape[0]
    else:
        num_langs = 4  # default

    if 'qim.theta_grid' in state_dict:
        L_max = state_dict['qim.theta_grid'].shape[0]
    else:
        L_max = 256  # default

    if 'embed.weight' in state_dict:
        d = state_dict['embed.weight'].shape[1]
    else:
        d = 64  # default

    # Instantiate model with correct parameters
    model = model_class(vocab_size=256, num_langs=num_langs, d=d, L_max=L_max, use_boundary_head=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Use the correct key for label mappings
    if "label_to_id" in checkpoint:
        label_mappings = checkpoint["label_to_id"]
    else:
        label_mappings = checkpoint.get("label_mappings", {})
    return model, label_mappings


# --------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------

def generate_text(
    model,
    prompt,
    label_mappings,
    max_length=100,
    temperature=0.7,  # Lower temperature for more conservative generation
    device="cpu",
    target_language=None
):
    """
    Generate text using only what the model learnt during training.
    Prompt is a string (will be tokenized via label_mappings).
    """
    model.eval()

    # Convert prompt to bytes (as in training) and add CLS token
    prompt_bytes = list(prompt.encode('utf-8'))
    prompt_bytes = [b for b in prompt_bytes if b < 48 or b > 57]  # Filter numericals
    if not prompt_bytes:
        raise ValueError("Prompt contains no valid bytes from training vocab")

    # Add CLS token (256) at the beginning, just like in training
    input_seq = torch.tensor([[256] + prompt_bytes], dtype=torch.long, device=device)
    generated_bytes = prompt_bytes.copy()

    for step in range(max_length):
        with torch.no_grad():
            out = model(input_seq)
            logits = out['lm_logits'][:, -1, :]  # Get logits for next token
            
            # Apply stronger language-specific conditioning
            if target_language and target_language.lower() != "unknown":
                if 'lang_logits' in out:
                    # Get the model's current language prediction
                    lang_logits = out['lang_logits']
                    lang_probs = F.softmax(lang_logits, dim=-1)
                    
                    # Find target language ID
                    target_lang_id = None
                    if isinstance(label_mappings, dict):
                        target_lang_id = label_mappings.get(target_language.lower())
                    
                    if target_lang_id is not None:
                        # Strong language conditioning - only if model is confident about target language
                        target_confidence = lang_probs[0, target_lang_id].item()
                        
                        # If model is not confident about the target language, reduce generation
                        if target_confidence < 0.3:  # Low confidence threshold
                            logits = logits * 0.5  # Reduce all token probabilities
                        else:
                            # Boost tokens when model is confident about target language
                            logits = logits * (1.0 + target_confidence)

        # Apply temperature (lower for more conservative generation)
        logits = logits / max(temperature, 0.1)
        
        # Filter out very low probability tokens to avoid nonsense
        logits_sorted, indices = torch.sort(logits, descending=True)
        # Keep only top 50% of tokens to avoid generating very unlikely sequences
        cutoff_idx = max(1, logits.shape[-1] // 2)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(-1, indices[:, :cutoff_idx], logits_sorted[:, :cutoff_idx])
        
        probs = F.softmax(logits_filtered, dim=-1)

        # Sample next byte
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Stop conditions
        if next_token == 0 or next_token > 255:  # Invalid byte
            break
            
        # Stop if generating repetitive patterns (sign of poor generation)
        if len(generated_bytes) > 10:
            recent_bytes = generated_bytes[-10:]
            if len(set(recent_bytes)) < 3:  # Too repetitive
                break

        generated_bytes.append(next_token)
        
        # Update input sequence with CLS token maintained
        new_token_tensor = torch.tensor([[next_token]], device=device)
        input_seq = torch.cat([input_seq, new_token_tensor], dim=1)

    # Convert generated bytes back to string
    try:
        generated_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
    except:
        generated_text = prompt  # Fallback to original prompt
        
    return generated_text


# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------

if __name__ == "__main__":
    from qimnlp import QINetMNLP  # Use the actual model class

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/qinet_m_nlp_best.pt"

    # load model + vocab
    model, label_mappings = load_model(QINetMNLP, checkpoint_path, device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (32-bit floats)")
    
    # Debug: print what's in label_mappings
    print(f"Debug - label_mappings keys: {label_mappings.keys() if isinstance(label_mappings, dict) else 'Not a dict'}")
    if isinstance(label_mappings, dict) and 'label_to_id' in label_mappings:
        print(f"Debug - label_to_id: {label_mappings['label_to_id']}")

    # Interactive loop for multiple prompts
    print("Enter prompts (type 'quit' or 'exit' to stop):")
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not prompt.strip():
                print("Please enter a valid prompt.")
                continue

            # Detect language
            detected_lang, confidence, all_confidences = detect_language(model, prompt, label_mappings, device)
            print(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
            
            # Show all language confidences
            print("All language confidences:")
            for lang, conf in sorted(all_confidences.items(), key=lambda x: x[1], reverse=True):
                print(f"  {lang}: {conf:.3f}")

            # generate
            output = generate_text(
                model,
                prompt=prompt,
                label_mappings=label_mappings,
                max_length=60,
                temperature=0.8,
                device=device,
                target_language=detected_lang  # Pass detected language to constrain generation
            )

            print(f"Generated: {output}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
