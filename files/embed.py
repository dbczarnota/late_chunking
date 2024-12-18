import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
import torch



def document_to_token_embeddings(model, tokenizer, document, batch_size=4096):
    """
    Given a model and tokenizer from HuggingFace, return token embeddings of the input text document.
    """

    if batch_size > 8192: # no of tokens
        raise ValueError("Batch size is too large. Please use a batch size of 8192 or less.")

    tokenized_document = tokenizer(document, return_tensors="pt")
    tokens = tokenized_document.tokens()
    
    # Batch in sizes of batch_size
    outputs = []
    for i in range(0, len(tokens), batch_size):
        
        start = i
        end   = min(i + batch_size, len(tokens))

        # subset huggingface tokenizer outputs to i : i + batch_size
        batch_inputs = {k: v[:, start:end] for k, v in tokenized_document.items()}

        with torch.no_grad():
            model_output = model(**batch_inputs)

        outputs.append(model_output.last_hidden_state)

    model_output = torch.cat(outputs, dim=1)
    return model_output