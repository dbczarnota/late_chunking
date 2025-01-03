from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from rich import print
from transformers import AutoTokenizer, AutoModel
from files.context_aware_chunker import ContextAwareChunker
from files.embed import clean_up

# A longer text excerpt (public domain - "Alice's Adventures in Wonderland")
text = """Alice was beginning to get very tired of sitting by Initialize the tokenizer and the ContextAwareChunker.her sister on the bank, and of having nothing to do: | once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), 
whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, 
when suddenly a White Rabbit with pink eyes ran close by her. Initialize the tokenizer and the ContextAwareChunker.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say 
to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that 
she ought to have wondered at this, but at the time it all seemed quite natural); 
but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, 
and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen 
a rabbit with either a waistcoat-pocket, or a watch to take out of it, 
and burning with curiosity, she ran across the field after it, and fortunately was just in time 
to see it pop down a large rabbit-hole under the hedge.
"""

# Initialize the tokenizer and the ContextAwareChunker
# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
# model = AutoModel.from_pretrained("BAAI/bge-m3", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
model     = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

chunker = ContextAwareChunker(
    tokenizer=tokenizer,
    model=model,
    max_sentence_length=80,   # Maximum tokens per initially split sentence
    min_sentence_length=20,   # Minimum tokens per final 'long' sentence
    sentence_split_regex= r'(?<=[.?!|])(?=\s+|\Z)|\n{1,2}(?=\S)',  # Regex to split on punctuation followed by space
    context_group_token_limit=250,            # Each context group can have up to 60 tokens
    context_group_overlap_size=100            # Each subsequent group overlaps the previous by 20 tokens
)

# Clean the text (optional, depending on your needs)
cleaned_text = chunker.clean_text(text)

# Split into long sentences
long_sentences = chunker.split_to_long_sentences(cleaned_text)
print(f'Long Sentences:\n{long_sentences}\n')

# Prepare context groups
groups = chunker.prepare_context_groups(long_sentences)

# Print the resulting groups
for i, group in enumerate(groups, start=1):
    print(f"\n--- Group {i} ---\n{group}\n")

# Create token embeddings
print("\nCreating token embeddings for context groups...")
token_embeddings_with_tokens = chunker.create_token_embeddings(groups)

# Debugging output for token embeddings and tokens
for i, (tokens, embedding) in enumerate(token_embeddings_with_tokens, start=1):
    print(f"\n--- Embedding {i} ---\nShape: {embedding.shape}\nFirst 5 values: {embedding[0][:5]}")
    print(f"Group {i} Tokens (First 5): {tokens[:5]}\nGroup {i} Tokens (Last 5): {tokens[-5:]}")

# Combine group tables
print("\nCombining group tables into a single table...")
combined_table = chunker.combine_group_tables(token_embeddings_with_tokens)

# Print some rows of the combined table for debugging
print("\nCombined Table (First 5 rows):")
print(combined_table.head())

# Print summary of the combined table
print("\nCombined Table Summary:")
print(combined_table.info())

# Generate pooled embeddings using span annotations and the Combined Table
print("\nGenerating pooled embeddings...")
span_annotations = [(0, 5), (5, 10), (10, 15)]  # Replace with actual token spans from your text
pooled_embeddings = chunker.generate_pooled_embeddings(span_annotations, combined_table)

# Debugging output for pooled embeddings
print("\nPooled Embeddings (First 3):")
for i, (embedding, tokens) in enumerate(pooled_embeddings[:3]):
    print(f"Pooled Embedding {i + 1} (First 5 values): {embedding[:5]}")
    print(f"Tokens in Pooled Embedding {i + 1}: {tokens}")

# Compare distances between chunks and retrieve tokens
print("\nComparing distances between chunks...")
span_annotations = [(0, 5), (5, 10), (10, 15)]  # Replace with actual token spans from your text
distances = chunker.compare_chunk_distances(combined_table, span_annotations)

# Print distances and tokens
print("\nChunk Distances and Tokens:")
for key, value in distances.items():
    print(f"{key}:")
    print(f"  Distance: {value['distance']:.4f}")
    print(f"  Tokens in Chunk 1: {value['tokens1']}")
    print(f"  Tokens in Chunk 2: {value['tokens2']}")


# Cleanup
try:
    clean_up()
except Exception as e:
    print(f"Error during cleanup: {e}")
