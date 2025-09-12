import PyPDF2
import re
from pathlib import Path

def extract_vocabulary_from_pdf(pdf_path):
    """Extracts a set of unique words from a PDF file, excluding English words."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    # Use regex to extract words (including tonal/Unicode)
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    vocab = set(word.lower() for word in words if not word.isdigit())

    # Built-in list of common English words
    english_words = set([
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he',
        'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
        'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
        'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
        'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
        'day', 'most', 'us', 'is', 'are', 'was', 'were', 'has', 'had', 'did', 'been', 'does', 'having', 'am',
        'were', 'shall', 'may', 'might', 'must', 'should', 'let', 'very', 'every', 'such', 'many', 'much',
        'where', 'why', 'whose', 'whom', 'which', 'who', 'whenever', 'wherever', 'however', 'whatever',
        'each', 'few', 'more', 'most', 'least', 'none', 'both', 'either', 'neither', 'nor', 'yet', 'though',
        'although', 'while', 'during', 'before', 'after', 'since', 'until', 'again', 'always', 'never', 'sometimes',
        'often', 'usually', 'once', 'twice', 'next', 'last', 'ago', 'soon', 'early', 'late', 'already', 'still',
        'just', 'ever', 'perhaps', 'maybe', 'sure', 'really', 'quite', 'too', 'enough', 'almost', 'around',
        'through', 'across', 'along', 'towards', 'upon', 'among', 'between', 'within', 'without', 'against',
        'above', 'below', 'under', 'beside', 'near', 'far', 'off', 'on', 'in', 'at', 'by', 'to', 'from', 'of',
        'with', 'for', 'about', 'as', 'like', 'than', 'but', 'or', 'if', 'because', 'so', 'though', 'although',
        'while', 'where', 'when', 'how', 'what', 'who', 'whom', 'whose', 'which', 'that', 'this', 'these', 'those'
    ])
    # Remove English words from vocab
    vocab = {word for word in vocab if word not in english_words}
    return vocab

if __name__ == "__main__":
    # Example usage
    pdf_path = "data/luo.pdf"
    vocab = extract_vocabulary_from_pdf(pdf_path)
    print(f"Extracted {len(vocab)} unique non-English words from {pdf_path}")
    # Optionally, save to file
    vocab_file = Path(pdf_path).with_suffix('.vocab.txt')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in sorted(vocab):
            f.write(word + '\n')
    print(f"Vocabulary saved to {vocab_file}")
