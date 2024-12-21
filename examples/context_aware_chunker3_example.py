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


text = """Artificial Intelligence (AI) has revolutionized how we interact with technology, making machines capable of learning, adapting, and solving complex problems. From powering recommendation systems to enhancing healthcare diagnostics, AI is reshaping the world and our daily lives. Its rapid evolution raises questions about how it will continue to integrate into society and influence creativity and productivity.

Interestingly, creativity is not confined to machines. Cooking, for example, remains a deeply human art form that blends technique, tradition, and imagination. The kitchen is a space where simple ingredients transform into meals that bring people together, evoke memories, and nourish both body and soul. Experimentation with flavors and methods makes cooking a timeless pursuit of creativity.

Just as food sustains us, plants sustain the Earth. They are the foundation of life, providing oxygen, food, and a connection to nature. Tending to plants, whether through gardening or simply caring for houseplants, brings a sense of calm and responsibility. Plants remind us of the delicate balance of ecosystems and our role in preserving them for future generations.
Motorsport is a captivating world where speed, precision, and engineering excellence converge to create a thrilling spectacle. From Formula 1 to MotoGP, rally racing to endurance events like Le Mans, motorsport encompasses a wide array of disciplines, each offering unique challenges and excitement.

At its core, motorsport is a celebration of human ingenuity and athleticism. Drivers and riders push themselves and their vehicles to the limit, relying on split-second decision-making and unmatched reflexes. Behind the scenes, teams of engineers and mechanics work tirelessly to optimize performance, crafting machines that are marvels of technology and innovation.

The history of motorsport is rich with iconic moments, legendary racers, and groundbreaking advancements. It has shaped the automotive industry, driving forward innovations in safety, aerodynamics, and fuel efficiency. The roar of engines, the thrill of overtaking, and the camaraderie of the racing community make motorsport a global phenomenon that continues to captivate millions of fans.
"""

# Load tokenizer and model
model_name = "jinaai/jina-embeddings-v3"
# model_name = "BAAI/bge-m3"  # Replace with the appropriate model name
# model_name = "jinaai/jina-embeddings-v2-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Initialize ContextAwareChunker with appropriate parameters
chunker = ContextAwareChunker(
    tokenizer=tokenizer,
    model=model,
 
)

# Generate chunks
chunks = chunker.create_chunks(text)

# Print the chunks
print("\nGenerated Chunks:")
for i, (chunk, ratio) in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk[:100]}...\nSplit Ratio: {ratio}\n")


# Cleanup
try:
    clean_up()
except Exception as e:
    print(f"Error during cleanup: {e}")
