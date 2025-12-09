def token_label_alignment(tokens: list[str], tags: list[str], tokenizer):

    encoding = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, truncation=True)

    subtokens = encoding.tokens()  # returns the list of subword tokens
    word_ids = encoding.word_ids() # returns the original word index for each subtoken

    aligned_labels = []
    for word in word_ids:
        if word is None:
            aligned_labels.append(-100)  # ignored in training, OK for inference too
        else:
            aligned_labels.append(tags[word])

    return encoding, aligned_labels
