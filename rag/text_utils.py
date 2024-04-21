def text2chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split the text into chunks of a specified size.
    - text: str: The text to split into chunks.
    - chụnk_size: int: The size of each chunk.
    - overlap: int: The overlap between the chunks.
    Note: The size is the number of words in the chunk.
    """
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks
