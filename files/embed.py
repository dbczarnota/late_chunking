import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
import torch



def text_to_token_embeddings(model, tokenizer, text, batch_size=4096):
    """
    Given a model and tokenizer from HuggingFace, return token embeddings of the input text.
    """

    if batch_size > 8192: # no of tokens
        raise ValueError("Batch size is too large. Please use a batch size of 8192 or less.")

    tokenized_text = tokenizer(text, return_tensors="pt")
    tokens = tokenized_text.tokens()

    if len(tokens) > batch_size:
        raise ValueError("Text is too long. Ensure it contains no more than batch_size tokens.")
    
    # Batch in sizes of batch_size
    outputs = []
    for i in range(0, len(tokens), batch_size):
        
        start = i
        end   = min(i + batch_size, len(tokens))

        # subset huggingface tokenizer outputs to i : i + batch_size
        batch_inputs = {k: v[:, start:end] for k, v in tokenized_text.items()}

        with torch.no_grad():
            model_output = model(**batch_inputs)

        outputs.append(model_output.last_hidden_state)

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

def get_span_annotations(chunks):
    """
    Given a list of chunks (strings) return the span annotations (start and end indices of chunks).
    """
    span_annotations = []
    start = 0
    for chunk in chunks:
        end = start + len(chunk)
        span_annotations.append((start, end))
        start = end
    return span_annotations