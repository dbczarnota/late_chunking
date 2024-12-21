from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pdfrom
from langchain_openai.embeddings import OpenAIEmbeddings
from scipy.spatial.distance import cosine
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path
from rich import print
from files.embed import count_tokens, create_token_embedding_dataframe, create_pooled_embeddings, calculate_embedding_distance_between_dfs, calculate_embedding_similarity_between_dfs



# Load tokenizer and model
model_name = "jinaai/jina-embeddings-v3"
# model_name = "BAAI/bge-m3"  # Replace with the appropriate model name
# model_name = "jinaai/jina-embeddings-v2-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

#initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Define the text for which to create token embeddings
text = """Artificial Intelligence (AI) is the simulation of human intelligence by machines, enabling them to perform tasks like learning, reasoning, problem-solving, and decision-making. It powers applications such as virtual assistants, recommendation systems, autonomous vehicles, and advanced robotics. By leveraging vast datasets and sophisticated algorithms, AI continues to transform industries, enhance productivity, and improve everyday life, though it also raises ethical considerations about privacy, bias, and the role of humans in an increasingly automated world."""
# text = """It powers applications such as virtual assistants, recommendation systems, autonomous vehicles, and advanced robotics."""
# text = """By leveraging vast datasets and sophisticated algorithms, AI continues to transform industries, enhance productivity, and improve everyday life, though it also raises ethical considerations about privacy, bias, and the role of humans in an increasingly automated world."""
# text = "Good"


print(f'Token count {count_tokens(tokenizer, text)}')
# Generate the DataFrame
print("Generating token embeddings...")
df = create_token_embedding_dataframe(model, tokenizer, text, batch_size=128, skip_beginning=0, skip_end=0, normalize=True)#41; 62
pooled_result = create_pooled_embeddings(df, pooling_method="mean")
openaiembed = embeddings.embed_query(text)



# Define just one token to check if the context is included in the previous one
text = """Cooking is the art and science of preparing food by combining ingredients and applying heat or other techniques to create meals. It is a universal activity that blends creativity and tradition, varying widely across cultures. From simple home-cooked dishes to gourmet recipes, cooking allows people to explore flavors, textures, and nutritional balance. Beyond sustenance, cooking fosters connection, as shared meals bring people together. Whether it's baking, grilling, steaming, or sautéing, the process of cooking is a rewarding way to express care and creativity."""
# text = "Evil"
# text = """It powers applications such as virtual assistants, recommendation systems, autonomous vehicles, and advanced robotics."""
# text = "Artificial Intelligence (AI)"
# text = """Sztuczna inteligencja (AI) to symulacja ludzkiej inteligencji przez maszyny, umożliwiająca im wykonywanie zadań takich jak uczenie się, rozumowanie, rozwiązywanie problemów i podejmowanie decyzji. Zasila aplikacje takie jak wirtualni asystenci, systemy rekomendacyjne, pojazdy autonomiczne i zaawansowana robotyka. Wykorzystując obszerne zbiory danych i zaawansowane algorytmy, AI wciąż transformuje różne branże, zwiększa produktywność i poprawia codzienne życie, choć jednocześnie rodzi pytania etyczne dotyczące prywatności, uprzedzeń oraz roli człowieka w coraz bardziej zautomatyzowanym świecie."""


print(f'Token count {count_tokens(tokenizer, text)}')
# Generate the DataFrame
print("Generating token embeddings...")
df = create_token_embedding_dataframe(model, tokenizer, text, batch_size=128, normalize=True)
pooled_result2 = create_pooled_embeddings(df, pooling_method="mean")


openaiembed2 = embeddings.embed_query(text)


distance = calculate_embedding_distance_between_dfs(pooled_result, pooled_result2)
similarity = calculate_embedding_similarity_between_dfs(pooled_result, pooled_result2)
#OpenAI distance
similarity_openai = 1 - cosine(openaiembed, openaiembed2)
print(f'Similarity between OpenAI embeddings: {similarity_openai}')