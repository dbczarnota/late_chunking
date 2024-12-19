from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from rich import print
from transformers import AutoTokenizer, AutoModel
from files.context_aware_chunker import ContextAwareChunker

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
text = "This is a short sentence. Here's another short one. This sentence is quite long enough on its own."

chunker = ContextAwareChunker(tokenizer)

final_sentences = chunker.split_to_long_sentences(text)

print(final_sentences)