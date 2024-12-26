import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import numpy as np
from joblib import Parallel, delayed
import threading
import pandas as pd
from rich import print
from typing import List, Tuple
from scipy.spatial.distance import cosine, euclidean, cityblock
from difflib import SequenceMatcher


def text_to_token_embeddings(model, tokenizer, text: str, batch_size: int = 4000, skip_beginning: int = 0, skip_end: int = 0, add_special_tokens: bool = False) -> Tuple[torch.Tensor, List[str]]:
    """
    Given a model and tokenizer from HuggingFace, return token embeddings of the input text,
    dynamically optimizing for CUDA or CPU, with the option to return embeddings for a subset of tokens.
    Special tokens are excluded from the embeddings.

    Args:
        model: HuggingFace model object.
        tokenizer: HuggingFace tokenizer object.
        text (str): Input text to be tokenized and processed.
        batch_size (int, optional): Maximum number of tokens to process in one batch.
        skip_beginning (int, optional): Number of tokens to skip from the beginning when returning embeddings.
        skip_end (int, optional): Number of tokens to skip from the end when returning embeddings.

    Returns:
        Tuple[torch.Tensor, List[str]]: Token embeddings of the subset of tokens and the corresponding tokens.
    """

    # Check for CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Move model to appropriate device
    model = model.to(device)

    if batch_size > 8192:  # Ensure batch size is within limit
        raise ValueError("Batch size is too large. Please use a batch size of 8192 or less.")

    # Tokenize the input text without adding special tokens
    tokenized_text = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens  
    )
    tokens = tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"].squeeze(0).tolist())

    # Move tokenized inputs to device
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}

    # Batch process the input
    outputs = []
    input_length = tokenized_text["input_ids"].size(1)
    for i in range(0, input_length, batch_size):
        start = i
        end = min(i + batch_size, input_length)

        # Subset tokenized inputs for the current batch
        batch_inputs = {k: v[:, start:end] for k, v in tokenized_text.items()}

        # Compute embeddings with no gradient computation
        with torch.no_grad():
            model_output = model(**batch_inputs)

        outputs.append(model_output.last_hidden_state)

    # Concatenate outputs along the token dimension
    all_embeddings = torch.cat(outputs, dim=1)

    # Apply skip_beginning and skip_end to the embeddings and tokens
    total_tokens = all_embeddings.size(1)
    if skip_beginning + skip_end > total_tokens:
        raise ValueError("The combination of skip_beginning and skip_end is greater than the number of tokens.")

    start_idx = skip_beginning
    end_idx = total_tokens - skip_end if skip_end > 0 else None

    subset_embeddings = all_embeddings[:, start_idx:end_idx, :]
    subset_tokens = tokens[start_idx:end_idx]

    return subset_embeddings, subset_tokens

def split_long_text_with_overlap(
    text: str,
    tokenizer,
    max_tokens: int = 4096,
    overlap_tokens: int = 128
):
    """
    Splits text into multiple chunks with overlap if total_tokens > max_tokens.

    Args:
        text (str): The input text to be split.
        tokenizer: A HuggingFace tokenizer.
        max_tokens (int): Max tokens per chunk (excluding overlap).
        overlap_tokens (int): Overlap size in tokens between consecutive chunks.

    Returns:
        List[dict]: Each dict contains:
            {
              "chunk_text": str,
              "skip_beginning": int,
              "skip_end": int,
              "start_token_idx": int,
              "end_token_idx": int
            }
    """
    # 1) Tokenize once for the full text
    tokenized = tokenizer(
        text,
        return_offsets_mapping=False,
        add_special_tokens=False
    )
    all_input_ids = tokenized["input_ids"]
    total_tokens = len(all_input_ids)

    print(f"[split_long_text_with_overlap] Total tokens in text: {total_tokens}")
    print(f"[split_long_text_with_overlap] max_tokens={max_tokens}, overlap_tokens={overlap_tokens}")

    # 2) If no splitting needed
    if total_tokens <= max_tokens:
        print("[split_long_text_with_overlap] No splitting needed; only 1 chunk.")
        return [{
            "chunk_text": text,
            "skip_beginning": 0,
            "skip_end": 0,
            "start_token_idx": 0,
            "end_token_idx": total_tokens
        }]

    # 3) Otherwise, split with overlap
    splitted_data = []
    step_size = max_tokens - overlap_tokens
    current_start = 0
    chunk_index = 0

    while current_start < total_tokens:
        chunk_end = min(current_start + max_tokens, total_tokens)

        # Slice token IDs directly
        chunk_ids = all_input_ids[current_start:chunk_end]

        splitted_data.append({
            "chunk_text": None,  # Placeholder for text (will reconstruct later)
            "skip_beginning": 0,
            "skip_end": 0,
            "start_token_idx": current_start,
            "end_token_idx": chunk_end
        })


        chunk_index += 1
        current_start += step_size

    # 4) Reconstruct text for each chunk and print tokens
    for i, chunk in enumerate(splitted_data):
        chunk_ids = all_input_ids[chunk["start_token_idx"]:chunk["end_token_idx"]]
        chunk["chunk_text"] = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Debug print for reconstructed tokens
        tokens_debug = tokenizer.convert_ids_to_tokens(chunk_ids)
        # print(f"[split_long_text_with_overlap] Chunk {i} tokens: {tokens_debug}")

    # 5) Summary print
    # print(f"[split_long_text_with_overlap] Generated {len(splitted_data)} chunks total.\n")

    return splitted_data




def create_chunk_embeddings(
    model,
    tokenizer,
    splitted_text: List[dict],
    batch_size: int = 4096
) -> List[dict]:
    """
    For each chunk in splitted_text, run text_to_token_embeddings. 
    Returns a list of dict with tokens and embeddings 
    (similar structure to text_to_token_embeddings output).

    Args:
        model: A HuggingFace model.
        tokenizer: A HuggingFace tokenizer.
        splitted_text (List[dict]): Output from split_long_text_with_overlap.
        batch_size (int, optional): Max tokens to process at once in text_to_token_embeddings.

    Returns:
        List[dict]: Each list entry has:
            {
              "tokens": [...],
              "embeddings": torch.Tensor,
              "start_token_idx": int,
              "end_token_idx": int
            }
    """


    results = []
    for chunk_info in splitted_text:
        # Run the standard text_to_token_embeddings on each chunk
        embeddings, tokens = text_to_token_embeddings(
            model=model,
            tokenizer=tokenizer,
            text=chunk_info["chunk_text"],
            batch_size=batch_size,
            skip_beginning=0,
            skip_end=0
        )

        results.append({
            "tokens": tokens,
            "embeddings": embeddings,  # shape [1, n_tokens, hidden_dim]
            "start_token_idx": chunk_info["start_token_idx"],
            "end_token_idx": chunk_info["end_token_idx"]
        })

    return results





def combine_chunk_embeddings_half_and_half(chunk_embeddings_data: List[dict]) -> Tuple[torch.Tensor, List[str]]:
    """
    Combine chunks by handling overlaps, where overlapping sections are split
    between chunks. For an overlap of size N, we take the first N/2 tokens from 
    the current chunk and the remaining tokens from the next chunk.
    """
    if not chunk_embeddings_data:
        print("[DEBUG] No chunks provided.")
        return torch.tensor([]), []

    final_tokens = []
    final_embeds = []

    def find_overlap(tokens1, tokens2, similarity_threshold=0.7):
        """
        Find the overlapping tokens between two sequences, comparing at a token level.
        
        Args:
            tokens1 (list): Tokens from the first chunk.
            tokens2 (list): Tokens from the second chunk.
            similarity_threshold (float): Threshold for token similarity to be considered a match.

        Returns:
            int: Size of the overlap in tokens.
        """
        max_possible = min(len(tokens1), len(tokens2))

        for size in range(max_possible, 0, -1):
            slice1 = tokens1[-size:]
            slice2 = tokens2[:size]
            
            # Compare token lists on a per-token basis
            similar_tokens = [
                SequenceMatcher(None, token1, token2).ratio() >= similarity_threshold
                for token1, token2 in zip(slice1, slice2)
            ]

            # If all tokens in the slice are similar, we found an overlap
            if all(similar_tokens):
                print(f"[DEBUG] Found overlap of size {size} (similarity threshold={similarity_threshold}):")
                print(f"[DEBUG] tokens1[-{size}:]={slice1}, tokens2[:{size}]={slice2}")
                return size

        print(f"[DEBUG] No significant overlap found between tokens1={tokens1[-16:]} and tokens2={tokens2[:16]}")
        return 0

    # Process the first chunk
    first_chunk = chunk_embeddings_data[0]
    current_tokens = first_chunk["tokens"]
    current_embeds = first_chunk["embeddings"].squeeze(0)

    print(f"[DEBUG] Chunk 0: tokens={current_tokens}")
    print(f"[DEBUG] Chunk 0 token count: {len(current_tokens)}")
    final_tokens.extend(current_tokens)
    final_embeds.append(current_embeds)

    # Process subsequent chunks
    for i in range(1, len(chunk_embeddings_data)):
        prev_chunk = chunk_embeddings_data[i - 1]
        current_chunk = chunk_embeddings_data[i]

        prev_tokens = prev_chunk["tokens"]
        current_tokens = current_chunk["tokens"]
        current_embeds = current_chunk["embeddings"].squeeze(0)

        print(f"[DEBUG] Chunk {i-1} tokens: {prev_tokens}")
        print(f"[DEBUG] Chunk {i} tokens: {current_tokens}")

        # Find overlap with the previous chunk
        overlap_size = find_overlap(prev_tokens, current_tokens)
        print(f"[DEBUG] Overlap size between Chunk {i-1} and Chunk {i}: {overlap_size}")

        if overlap_size > 0:
            # Retain the first portion of the overlap in the previous chunk
            keep_prev_tokens = len(final_tokens) - (overlap_size // 2)
            final_tokens = final_tokens[:keep_prev_tokens]
            final_embeds[-1] = final_embeds[-1][:keep_prev_tokens]

            print(f"[DEBUG] Keeping tokens from previous chunk up to index {keep_prev_tokens}: {final_tokens}")

            # Add only the non-overlapping portion of the current chunk
            start_idx = overlap_size - (overlap_size // 2)
            print(f"[DEBUG] Adding tokens from current chunk starting at index {start_idx}: {current_tokens[start_idx:]}")
            final_tokens.extend(current_tokens[start_idx:])
            final_embeds.append(current_embeds[start_idx:])
        else:
            # No overlap; add the entire current chunk
            print(f"[DEBUG] No overlap detected. Adding entire current chunk.")
            final_tokens.extend(current_tokens)
            final_embeds.append(current_embeds)


    # Combine embeddings into a single tensor
    try:
        final_embeddings = torch.cat(final_embeds, dim=0).unsqueeze(0)
    except Exception as e:
        print(f"[ERROR] Error during tensor concatenation: {e}")
        return torch.tensor([]), final_tokens

    # Debugging final outputs
    print(f"[DEBUG] Final number of tokens: {len(final_tokens)}")
    print(f"[DEBUG] Final tokens: {final_tokens}")
    print(f"[DEBUG] Final embeddings shape: {final_embeddings.shape}")

    return final_embeddings, final_tokens






def long_text_to_token_embedding_dataframe(
    text: str,
    model,
    tokenizer,
    max_tokens=512,
    overlap_tokens=50,
    batch_size=128,
    normalize=True
) -> pd.DataFrame:
    """
    Splits `text` into chunks (with overlap), embeds each chunk,
    combines them into a single (embeddings, tokens), filters out special tokens,
    and returns a DataFrame of token-level embeddings — no pooling at this stage.

    Args:
        text (str): The input text to embed.
        model: HuggingFace model for embeddings.
        tokenizer: HuggingFace tokenizer.
        max_tokens (int): Maximum tokens per chunk (excluding overlap).
        overlap_tokens (int): Number of overlapping tokens between chunks.
        batch_size (int): Batch size for embedding chunks.
        normalize (bool): Whether to normalize embeddings.

    Returns:
        pd.DataFrame: DataFrame with columns ["Token", "Embedding"] excluding special tokens.
    """


    # A) Split text into chunks
    print("[INFO] Splitting text into chunks with overlap...")
    splitted = split_long_text_with_overlap(
        text=text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens
    )

    # B) Embed each chunk
    print("[INFO] Creating chunk embeddings...")
    chunk_embs_data = create_chunk_embeddings(
        model=model,
        tokenizer=tokenizer,
        splitted_text=splitted,
        batch_size=batch_size
    )

    # C) Combine chunk embeddings (half-and-half overlap)
    print("[INFO] Combining chunk embeddings with half-and-half overlap...")
    combined_embeddings, combined_tokens = combine_chunk_embeddings_half_and_half(
        chunk_embeddings_data=chunk_embs_data,

    )

    # D) Build a DataFrame with (Token, Embedding), excluding special tokens
    print("[INFO] Building DataFrame and filtering out special tokens...")
    final_list = []
    combined_embeddings_cpu = combined_embeddings.squeeze(0).cpu()

    # Get the set of all special tokens
    special_tokens = set(tokenizer.all_special_tokens)

    # Counters for debugging
    total_tokens = len(combined_tokens)
    skipped_tokens = 0

    for i, token in enumerate(combined_tokens):
        if token in special_tokens:
            skipped_tokens += 1
            continue  # Skip special tokens

        # Convert to float32 before going to NumPy to avoid ScalarType BFloat16
        emb = combined_embeddings_cpu[i].float().numpy()
        final_list.append({"Token": token, "Embedding": emb})

    df = pd.DataFrame(final_list)

    print(f"[INFO] Total tokens processed: {total_tokens}")
    print(f"[INFO] Special tokens skipped: {skipped_tokens}")
    print(f"[INFO] Tokens included in DataFrame: {len(df)}")

    # E) Optionally normalize embeddings
    if normalize and not df.empty:
        print("[INFO] Normalizing embeddings...")
        emb_matrix = np.stack(df["Embedding"].values)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
        emb_matrix = emb_matrix / norms
        df["Embedding"] = list(emb_matrix)

    print("[INFO] DataFrame creation complete.\n")
    return df


def normalize_embeddings(embeddings):
    """
    Normalize each embedding vector to have unit norm.

    Args:
        embeddings (np.ndarray): Embedding vectors.

    Returns:
        np.ndarray: Normalized embedding vectors.
    """
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / norms

def create_token_embedding_dataframe(model, tokenizer, text, batch_size=4096, skip_beginning=0, skip_end=0, normalize=True):
    """
    Given a text, use the text_to_token_embeddings function to generate a DataFrame
    with tokens and their corresponding embeddings, excluding special tokens.

    Args:
        model: HuggingFace model object.
        tokenizer: HuggingFace tokenizer object.
        text (str): Input text to be tokenized and processed.
        batch_size (int, optional): Maximum number of tokens to process in one batch.
        skip_beginning (int, optional): Number of tokens to skip from the beginning when returning embeddings.
        skip_end (int, optional): Number of tokens to skip from the end when returning embeddings.
        normalize (bool, optional): Whether to normalize embeddings to unit norm. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing tokens and their embeddings.
    """
    # Generate token embeddings and tokens using text_to_token_embeddings
    embeddings, tokens = text_to_token_embeddings(
        model=model,
        tokenizer=tokenizer,
        text=text,
        batch_size=batch_size,
        skip_beginning=skip_beginning,
        skip_end=skip_end
    )

    # Convert embeddings to a list of NumPy arrays for compatibility with Pandas
    # embeddings = [embedding.cpu().numpy() for embedding in embeddings.squeeze(0)]
    embeddings = [embedding.cpu().to(torch.float32).numpy() for embedding in embeddings.squeeze(0)]

    if len(tokens) != len(embeddings):
        raise ValueError("Mismatch between number of tokens and embeddings.")

    # Filter out special tokens
    special_tokens = tokenizer.all_special_tokens
    tokens, embeddings = zip(*[
        (token, embedding) for token, embedding in zip(tokens, embeddings)
        if token not in special_tokens
    ])

    # Normalize embeddings if required
    if normalize:
        print("[INFO] Normalizing embeddings to unit norm.")
        embeddings = normalize_embeddings(np.stack(embeddings))

    # Create a DataFrame
    df = pd.DataFrame({
        "Token": tokens,
        "Embedding": list(embeddings)
    })

    # Debugging: Print some details for verification
    print("\n[DEBUG] DataFrame created (excluding special tokens):")
    print(df.head())

    return df

def create_pooled_embeddings(df, pooling_method="mean"):
    """
    Create pooled embeddings for the tokens in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing tokens and their embeddings.
        pooling_method (str): Pooling method to use ("mean", "max", "min", "sum", "median").

    Returns:
        dict: A dictionary containing:
            - "Token": List of tokens in the DataFrame.
            - "FullText": The concatenated text of all tokens.
            - "Embedding": The pooled embedding vector.
    """
    # Extract embeddings from the DataFrame
    embeddings = np.stack(df["Embedding"].values)

    print(f"[INFO] Number of embeddings to pool: {len(embeddings)}")

    # Apply the specified pooling method
    if pooling_method == "mean":
        pooled_embedding = np.mean(embeddings, axis=0)
        print("[INFO] Applied mean pooling.")
    elif pooling_method == "max":
        pooled_embedding = np.max(embeddings, axis=0)
        print("[INFO] Applied max pooling.")
    elif pooling_method == "min":
        pooled_embedding = np.min(embeddings, axis=0)
        print("[INFO] Applied min pooling.")
    elif pooling_method == "sum":
        pooled_embedding = np.sum(embeddings, axis=0)
        print("[INFO] Applied sum pooling.")
    elif pooling_method == "median":
        pooled_embedding = np.median(embeddings, axis=0)
        print("[INFO] Applied median pooling.")
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

    
    tokens = df["Token"].values.tolist()
    # print(f"[INFO] Tokens list:")
    # print(tokens)

    print("\n[INFO] Pooled embedding (first 10 dimensions):")
    print(pooled_embedding[:10])

    return {
        "Token": tokens,
        "Embedding": pooled_embedding
    }

def calculate_embedding_distance_between_dfs(pooled_result1, pooled_result2, distance_metric="cosine"):
    """
    Calculate the distance between pooled embeddings from two DataFrames.

    Args:
        pooled_result1 (dict): Result from create_pooled_embeddings for the first DataFrame.
        pooled_result2 (dict): Result from create_pooled_embeddings for the second DataFrame.
        distance_metric (str): Distance metric to use ("cosine", "euclidean", "manhattan").

    Returns:
        dict: A dictionary containing:
            - "Tokens1": Tokens from the first DataFrame.
            - "Tokens2": Tokens from the second DataFrame.
            - "Distance": Calculated distance between the embeddings.
    """
    embedding1 = pooled_result1["Embedding"]
    embedding2 = pooled_result2["Embedding"]

    if distance_metric == "cosine":
        distance = cosine(embedding1, embedding2)
        # print("[INFO] Cosine distance calculated.")
    elif distance_metric == "euclidean":
        distance = euclidean(embedding1, embedding2)
        # print("[INFO] Euclidean distance calculated.")
    elif distance_metric == "manhattan":
        distance = cityblock(embedding1, embedding2)
        # print("[INFO] Manhattan distance calculated.")
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # print(f"\n[INFO] Tokens from first DataFrame:")
    # print(" ".join(pooled_result1["Token"]))
    # print(f"\n[INFO] Number of Tokens from first DataFrame: {len(pooled_result1['Token'])}")
    # print(f"\n[INFO] Tokens from second DataFrame:")
    # print(" ".join(pooled_result2["Token"]))
    # print(f"\n[INFO] Number of Tokens from second DataFrame: {len(pooled_result2['Token'])}")
    print(f"\n[INFO] Distance ({distance_metric}): {distance}")

    return {
        "Tokens1": pooled_result1["Token"],
        "Tokens2": pooled_result2["Token"],
        "Distance": distance
    }

def calculate_embedding_similarity_between_dfs(pooled_result1, pooled_result2, similarity_metric="cosine"):
    """
    Calculate the similarity between pooled embeddings from two DataFrames.

    Args:
        pooled_result1 (dict): Result from create_pooled_embeddings for the first DataFrame.
        pooled_result2 (dict): Result from create_pooled_embeddings for the second DataFrame.
        similarity_metric (str): Similarity metric to use ("cosine", "dot_product", "pearson", "manhattan").

    Returns:
        dict: A dictionary containing:
            - "Tokens1": Tokens from the first DataFrame.
            - "Tokens2": Tokens from the second DataFrame.
            - "Similarity": Calculated similarity between the embeddings.
    """
    embedding1 = pooled_result1["Embedding"]
    embedding2 = pooled_result2["Embedding"]

    if similarity_metric == "cosine":
        similarity = 1 - cosine(embedding1, embedding2)
        # print("[INFO] Cosine similarity calculated.")
    elif similarity_metric == "dot_product":
        similarity = np.dot(embedding1, embedding2)
        # print("[INFO] Dot product similarity calculated.")
    elif similarity_metric == "pearson":
        similarity = np.corrcoef(embedding1, embedding2)[0, 1]
        # print("[INFO] Pearson correlation similarity calculated.")
    elif similarity_metric == "manhattan":
        distance = cityblock(embedding1, embedding2)
        similarity = 1 / (1 + distance)  # Transform Manhattan distance into similarity
        # print("[INFO] Manhattan similarity calculated.")
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    # print(f"\n[INFO] Tokens from first DataFrame:")
    # print(", ".join(pooled_result1["Token"]))
    # print(f"\n[INFO] Tokens from second DataFrame:")
    # print(", ".join(pooled_result2["Token"]))
    print(f"\n[INFO] Similarity ({similarity_metric}): {similarity}")

    return {
        "Tokens1": pooled_result1["Token"],
        "Tokens2": pooled_result2["Token"],
        "Similarity": similarity
    }

def count_tokens(tokenizer, text):
    """
    Count the number of tokens in the text using the tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer object.
        text: The input text (string) to be tokenized.

    Returns:
        int: The number of tokens in the text.
    """
    tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return len(tokenized_text.input_ids[0])

def get_span_annotations_from_text(text, chunks):
    """
    Given a list of chunks (strings) and the original text, return the span annotations
    (start and end indices of chunks) within the original text.

    Args:
        text (str): The combined original text.
        chunks (List[str]): List of pre-chunked text segments.

    Returns:
        List[Tuple[int, int]]: Character spans for each chunk in the original text.
    """
    span_annotations = []
    current_index = 0

    for chunk in chunks:
        # Find the exact start and end indices of the chunk in the original text
        start = text.find(chunk, current_index)
        if start == -1:
            raise ValueError(f"Chunk '{chunk}' not found in the text starting at index {current_index}")
        end = start + len(chunk)
        span_annotations.append((start, end))
        current_index = end  # Update current index to avoid overlapping matches

    return span_annotations

def char_to_token_spans(tokenizer, text, char_spans):
    """
    Converts character-based span annotations to token-based span annotations.

    Args:
        tokenizer: HuggingFace tokenizer object.
        text (str): The original text.
        char_spans (List[Tuple[int, int]]): List of character-based span annotations (start, end).

    Returns:
        List[Tuple[int, int]]: List of token-based span annotations (start_token, end_token).
    """
    tokenized_text = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False
    )
    offset_mapping = tokenized_text['offset_mapping'][0].tolist()  # Character positions for each token

    # print("\nTokenized text and offsets:")
    # tokens = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])
    # for i, (token, (start, end)) in enumerate(zip(tokens, offset_mapping)):
    #     print(f"Token {i}: '{token}' (Start: {start}, End: {end})")

    token_spans = []
    for start_char, end_char in char_spans:
        start_token = None
        end_token = None

        # Find the token indices corresponding to the character span
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if start_token is None and token_start <= start_char < token_end:
                start_token = i
            if token_start < end_char <= token_end:
                end_token = i + 1  # End token is exclusive
                break

        # Handle edge cases where the span partially overlaps tokens
        if start_token is None:
            start_token = 0  # Default to start of the sequence
        if end_token is None:
            end_token = len(offset_mapping)  # Default to end of the sequence

        token_spans.append((start_token, end_token))

    return token_spans

def late_chunking(token_embeddings, token_spans, max_length=None, batch_size=128):
    
    """
    Performs late chunking by pooling token embeddings for each token-based span, 
    dynamically optimizing for CUDA or CPU.

    Args:
        token_embeddings (torch.Tensor): Token-level embeddings for the document.
        token_spans (List[Tuple[int, int]]): Token-based span annotations (start_token, end_token).
        max_length (int, optional): Maximum allowed sequence length for processing.
        batch_size (int, optional): Batch size for GPU-based processing.

    Returns:
        List[np.ndarray]: A list of pooled embeddings for each span.
    """
    # Check for CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    token_embeddings = token_embeddings.to(device)

    # Ensure token_embeddings has correct dimensions
    if len(token_embeddings.shape) == 3:  # [batch_size, num_tokens, hidden_dim]
        token_embeddings = token_embeddings[0]  # Remove batch dimension
    num_tokens, hidden_dim = token_embeddings.shape

    print(f"Using device: {device}")
    print(f"Number of tokens in embeddings: {num_tokens}")
    print(f"Token spans: {token_spans}")

    # CUDA + Batching Optimization
    if use_cuda:
        print("CUDA detected. Using batch processing on GPU.")
        pooled_embeddings = []

        for i in range(0, len(token_spans), batch_size):
            batch_spans = token_spans[i:i + batch_size]

            # Create a mask for batch spans
            batch_mask = torch.zeros(len(batch_spans), num_tokens, dtype=torch.bool, device=device)
            for idx, (start, end) in enumerate(batch_spans):
                if start < end <= num_tokens:
                    batch_mask[idx, start:end] = True

            # Perform pooling using the mask
            span_embeddings = token_embeddings.unsqueeze(0) * batch_mask.unsqueeze(-1)
            span_sums = span_embeddings.sum(dim=1)
            span_counts = batch_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_batch = (span_sums / span_counts).cpu().numpy()  # Normalize and move to CPU
            pooled_embeddings.extend(pooled_batch)
    else:
        # CPU + Parallelization Optimization
        print("CUDA not available. Using joblib parallelization on CPU.")

        def process_span(start_token, end_token):
            if start_token >= end_token or start_token >= num_tokens:
                return np.zeros(hidden_dim)  # Invalid span returns a zero vector
            span_embeddings = token_embeddings[start_token:end_token, :]
            return span_embeddings.mean(dim=0).cpu().numpy()

        pooled_embeddings = Parallel(n_jobs=-1)(
            delayed(process_span)(start, end) for start, end in token_spans
        )

    print(f"Final pooled embeddings: {len(pooled_embeddings)} spans processed.")
    return pooled_embeddings

def clean_up():
    # Clear CUDA memory
    print("\nCleaning up...")
    torch.cuda.empty_cache()

    # Wait for threads to finish naturally
    for thread in threading.enumerate():
        if thread.name != "MainThread":
            print(f"Joining thread: {thread.name}")
            thread.join(timeout=1)  # Attempt to join the thread

    print("Cleanup completed.")