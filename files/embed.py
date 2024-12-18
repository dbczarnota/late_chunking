import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import numpy as np
from joblib import Parallel, delayed
import threading


def text_to_token_embeddings(model, tokenizer, text, batch_size=4096):
    """
    Given a model and tokenizer from HuggingFace, return token embeddings of the input text,
    dynamically optimizing for CUDA or CPU.

    Args:
        model: HuggingFace model object.
        tokenizer: HuggingFace tokenizer object.
        text (str): Input text to be tokenized and processed.
        batch_size (int, optional): Maximum number of tokens to process in one batch.

    Returns:
        torch.Tensor: Token embeddings of the input text.
    """
    # Check for CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Move model to appropriate device
    model = model.to(device)

    if batch_size > 8192:  # Ensure batch size is within limit
        raise ValueError("Batch size is too large. Please use a batch size of 8192 or less.")

    # Tokenize the input text
    tokenized_text = tokenizer(text, return_tensors="pt")
    tokens = tokenized_text.tokens()

    if len(tokens) > batch_size:
        raise ValueError("Text is too long. Ensure it contains no more than batch_size tokens.")

    # Move tokenized inputs to device
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}

    # Batch process the input
    outputs = []
    for i in range(0, len(tokens), batch_size):
        start = i
        end = min(i + batch_size, len(tokens))

        # Subset tokenized inputs for the current batch
        batch_inputs = {k: v[:, start:end] for k, v in tokenized_text.items()}

        # Compute embeddings with no gradient computation
        with torch.no_grad():
            model_output = model(**batch_inputs)

        outputs.append(model_output.last_hidden_state)

    # Concatenate outputs along the token dimension
    model_output = torch.cat(outputs, dim=1)

    return model_output

def count_tokens(tokenizer, text):
    """
    Count the number of tokens in the text using the tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer object.
        text: The input text (string) to be tokenized.

    Returns:
        int: The number of tokens in the text.
    """
    tokenized_text = tokenizer(text, return_tensors="pt")
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