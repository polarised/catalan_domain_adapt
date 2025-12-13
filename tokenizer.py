from datasets import Dataset


def token_label_alignment(tokens: list[str], tags: list[str], tokenizer):
    """
    Align POS tags with subword tokens from tokenizer.
    
    Args:
        tokens: List of word tokens
        tags: List of POS tags (same length as tokens)
        tokenizer: HuggingFace tokenizer
    
    Returns:
        encoding: Tokenizer encoding with input_ids, attention_mask, etc.
        aligned_labels: List of labels aligned to subword tokens
                       (-100 for special tokens and continuation subtokens)
    """
    # Tokenize with word-level information
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512,  # Standard max length
        padding=False,  # Don't pad here, let DataCollator handle it
        return_offsets_mapping=False,  # Not needed for alignment
    )

    # Get word IDs for each subtoken
    word_ids = encoding.word_ids()
    
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        # Special tokens (CLS, SEP, PAD) have word_idx = None
        if word_idx is None:
            aligned_labels.append(-100)
        # First subtoken of a word gets the label
        elif word_idx != previous_word_idx:
            aligned_labels.append(tags[word_idx])
        # Continuation subtokens get -100 (only first subtoken is labeled)
        else:
            aligned_labels.append(-100)
        
        previous_word_idx = word_idx
    
    return encoding, aligned_labels
