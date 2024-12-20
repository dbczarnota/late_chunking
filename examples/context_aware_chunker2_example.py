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
text = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: | once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, \"and what is the use of a book,\" thought Alice \"without pictures or conversations?\"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), 
whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, 
when suddenly a White Rabbit with pink eyes ran close by her. 

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say 
to itself, \"Oh dear! Oh dear! I shall be late!\" (when she thought it over afterwards, it occurred to her that 
she ought to have wondered at this, but at the time it all seemed quite natural); 
but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, 
and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen 
a rabbit with either a waistcoat-pocket, or a watch to take out of it, 
and burning with curiosity, she ran across the field after it, and fortunately was just in time 
to see it pop down a large rabbit-hole under the hedge.

Artificial Intelligence (AI) is the simulation of human intelligence in machines that are designed to think, learn, and make decisions. It powers a wide range of technologies, from virtual assistants like Siri and Alexa to self-driving cars and advanced robotics. By leveraging algorithms and vast datasets, AI can analyze patterns, predict outcomes, and automate tasks, transforming industries such as healthcare, finance, and education. As AI evolves, it continues to unlock new possibilities, sparking both excitement and ethical discussions about its impact on society.
"""

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
    context_group_overlap_size=50             # Each subsequent group overlaps the previous by 50 tokens
)

# Compare sentence distances
print("\n[bold green]Comparing sentence distances:[/bold green]")
distance_results = chunker.compare_sentence_distances(text)

# Print the results
for i, result in enumerate(distance_results, start=1):
    print(f"\n[bold yellow]Sentence {i}:[/bold yellow] {result['current_sentence']}")
    if "previous" in result and result["previous"] is not None:
        print(f"Distance to previous: {result['previous']['distance']:.4f}")
        print(f"Tokens: {result['previous']['tokens1']} => {result['previous']['tokens2']}")
    else:
        print("No previous sentence comparison available.")

    if "next" in result and result["next"] is not None:
        print(f"Distance to next: {result['next']['distance']:.4f}")
        print(f"Tokens: {result['next']['tokens1']} => {result['next']['tokens2']}")
    else:
        print("No next sentence comparison available.")

# Cleanup
try:
    clean_up()
except Exception as e:
    print(f"Error during cleanup: {e}")
