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
text = """Artificial Intelligence (AI) has revolutionized how we interact with technology, making machines capable of learning, adapting, and solving complex problems. From powering recommendation systems to enhancing healthcare diagnostics, AI is reshaping the world and our daily lives. Its rapid evolution raises questions about how it will continue to integrate into society and influence creativity and productivity.

Interestingly, creativity is not confined to machines. Cooking, for example, remains a deeply human art form that blends technique, tradition, and imagination. The kitchen is a space where simple ingredients transform into meals that bring people together, evoke memories, and nourish both body and soul. Experimentation with flavors and methods makes cooking a timeless pursuit of creativity.

Just as food sustains us, plants sustain the Earth. They are the foundation of life, providing oxygen, food, and a connection to nature. Tending to plants, whether through gardening or simply caring for houseplants, brings a sense of calm and responsibility. Plants remind us of the delicate balance of ecosystems and our role in preserving them for future generations."""

# Initialize the tokenizer and the ContextAwareChunker
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
model = AutoModel.from_pretrained("BAAI/bge-m3", trust_remote_code=True)

chunker = ContextAwareChunker(
    tokenizer=tokenizer,
    model=model,
    max_sentence_length=80,   # Maximum tokens per initially split sentence
    min_sentence_length=15,   # Minimum tokens per final 'long' sentence
    sentence_split_regex= r'(?<=[.?!|])(?=\s+|\Z)|\n{1,2}(?=\S)',  # Regex to split on punctuation followed by space
    context_group_token_limit=250,            # Each context group can have up to 250 tokens
    context_group_overlap_size=50,             # Each subsequent group overlaps the previous by 50 tokens
    pooling_method="mean",
    similarity_metric = "cosine"
)

# Compare sentence distances
print("\n[bold green]Comparing sentence distances:[/bold green]")
distance_results = chunker.compare_sentence_distances(text)



# Cleanup
try:
    clean_up()
except Exception as e:
    print(f"Error during cleanup: {e}")
