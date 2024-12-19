from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from rich import print
from transformers import AutoTokenizer, AutoModel
from files.embed import text_to_token_embeddings, late_chunking, char_to_token_spans, get_span_annotations_from_text, clean_up
from files.handle_weaviate import connect_to_weaviate, add_to_weaviate
from files.sentence_chunkers import split_to_sentences

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3', trust_remote_code=True)
    model = AutoModel.from_pretrained('BAAI/bge-m3', trust_remote_code=True)

    # # Sample pre-chunked text input
    # pre_chunked_text = [
    #     "Transformers are a powerful tool for NLP tasks.",
    #     "They enable the use of pre-trained models.",
    #     "These models can be fine-tuned for specific applications."
    # ]

    # Alternatively, use a single text input and split it into sentences
    text = "Transformers are a powerful tool for NLP tasks. They enable the use of pre-trained models. These models can be fine-tuned for specific applications."
    sentence_split_regex = r'(?<=[.!?]) +'
    token_limit = 150  # Adjust token limit as needed
    pre_chunked_text = split_to_sentences(sentence_split_regex, text, tokenizer, token_limit)

    # Combine chunks into a single text for consistent processing
    combined_text = " ".join(pre_chunked_text)
    print(f"Combined Text: {combined_text}")

    # Get character spans for pre-chunked text
    char_spans = get_span_annotations_from_text(combined_text, pre_chunked_text)
    print("\nCharacter spans:")
    for i, span in enumerate(char_spans):
        print(f"Chunk {i + 1}: (Start: {span[0]}, End: {span[1]})")

    # Convert character spans to token spans
    print("\nConverting character spans to token spans...")
    token_spans = char_to_token_spans(tokenizer, combined_text, char_spans)
    print("Token spans:", token_spans)

    # Generate token embeddings for the combined text
    print("\nGenerating token embeddings...")
    token_embeddings = text_to_token_embeddings(model, tokenizer, combined_text, batch_size=512)
    print("Token embeddings generated with shape:", token_embeddings.shape)

    # Apply late chunking
    print("\nApplying late chunking...")
    token_embeddings = token_embeddings[0]  # Unpack batch dimension
    sentence_embeddings = late_chunking(token_embeddings, token_spans)
    
    # Print sentence embeddings
    print("\nSentence embeddings (first few numbers):")
    for i, embedding in enumerate(sentence_embeddings):
        print(f"Sentence {i + 1} embedding (first 5 values): {embedding[:5]}")

    # Connect to Weaviate
    print("\nConnecting to Weaviate...")
    client = connect_to_weaviate("test_late_chunking", delete_existing=False)

    # Upload results to Weaviate
    print("\nUploading to Weaviate...")
    add_to_weaviate(client, "test_late_chunking", pre_chunked_text, sentence_embeddings)
    client.close()
    
    # Cleanup
    try:
        clean_up()
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()