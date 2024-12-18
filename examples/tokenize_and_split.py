from pathlib import Path
import sys
from rich import print

# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from transformers import AutoTokenizer, AutoModel
from files.embed import text_to_token_embeddings, late_chunking, char_to_token_spans
from files.sentence_chunkers import split_to_sentences
import re





def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3', trust_remote_code=True)
    model = AutoModel.from_pretrained('BAAI/bge-m3', trust_remote_code=True)

    # Sample text input
    text = "Transformers are a powerful tool for NLP tasks. They enable the use of pre-trained models. These models can be fine-tuned for specific applications."

    # Get token embeddings
    print("Generating token embeddings...")
    token_embeddings = text_to_token_embeddings(model, tokenizer, text, batch_size=512)
    print("Token embeddings generated with shape:", token_embeddings.shape)

    # Split text into sentences
    print("Splitting text into sentences...")
    sentence_split_regex = r'(?<=[.!?])\s+'  # Regex to split by punctuation followed by whitespace
    token_limit = 50  # Example token limit per sentence

    sentences, char_spans = split_to_sentences(sentence_split_regex, text, tokenizer, token_limit)
    print("\nSentences and their spans:")
    for i, (sentence, span) in enumerate(zip(sentences, char_spans)):
        print(f"{i + 1}: '{sentence}' (Start: {span[0]}, End: {span[1]})")

    # Convert character spans to token spans
    print("\nConverting character spans to token spans...")
    token_spans = char_to_token_spans(tokenizer, text, char_spans)
    print("Token spans:", token_spans)

    # Apply late chunking
    print("\nApplying late chunking...")
    token_embeddings = token_embeddings[0]  # Unpack batch dimension
    sentence_embeddings = late_chunking(token_embeddings, token_spans)

    # Print sentence embeddings
    print("\nSentence embeddings (first few numbers):")
    for i, embedding in enumerate(sentence_embeddings):
        print(f"Sentence {i + 1} embedding (first 5 values): {embedding[:5]}")




if __name__ == "__main__":
    main()
