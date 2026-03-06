def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    step = chunk_size - overlap
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), step)]
    if not chunks:
        return []
    return [c for i, c in enumerate(chunks) if len(c) == chunk_size or (i == 0)]