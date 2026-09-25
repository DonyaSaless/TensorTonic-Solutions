def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    vocab = {i : bytes([i]) for i in range(256)}
    ## now want to convert every sequence to a seq of ints in vocab
    sequences = [list(sequence.encode("utf-8")) for sequence in corpus]

    merges = []

    ## we want to keep merging up until our vocab size reaches vocab_size.
    while len(vocab) < vocab_size:
        ## at every step we merge the most repeated pair into a new vocab
        pair_counts = dict()
        for sequence in sequences:
            for left, right in zip (sequence, sequence[1:]):
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
            ## if there is nothing to merge we should break this 
        if not pair_counts:
            break
    
        left , right = max(pair_counts, key = lambda pair: (pair_counts[pair], vocab[pair[0]] , vocab[pair[1]]))

        new_id = len(vocab)
        vocab[len(vocab)] = vocab[left] + vocab[right]

        merges.append([left, right, new_id])

        updated_sequences = []
        for sequence in sequences:
            updated = []
            index = 0
            while index < len(sequence):
                if index + 1 <  len(sequence) and sequence[index] == left and sequence[index+1] == right:
                    updated.append(new_id)
                    index = index + 2
                else: 
                    updated.append(sequence[index])
                    index = index + 1

            updated_sequences.append(updated)
        sequences = updated_sequences

    learned_vocab = [[i , list(vocab[i])] for i in range(256, len(vocab))]
    return  {"vocab": learned_vocab, "merges": merges}
      
        