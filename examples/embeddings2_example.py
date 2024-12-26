from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModel
from rich import print
from scipy.spatial.distance import cosine

# Adjust paths if needed
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

# Import from your embed.py
from files.embed import (
    long_text_to_token_embedding_dataframe,
    create_pooled_embeddings,
    count_tokens,
    calculate_embedding_distance_between_dfs,
    calculate_embedding_similarity_between_dfs
)

def main():
    ########################################
    # 1. Load tokenizer & model
    ########################################
    model_name = "jinaai/jina-embeddings-v3"  # or "BAAI/bge-m3", etc.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    ########################################
    # 2. Example texts to compare
    ########################################
    text_ai = """Artificial Intelligence (AI) is a transformative technology that leverages algorithms to perform tasks traditionally requiring human intelligence. These tasks include problem-solving, decision-making, natural language understanding, and pattern recognition. AI spans a spectrum of subfields, including machine learning, deep learning, computer vision, and natural language processing.

                Machine learning, a core subset, involves training models on data to make predictions or automate processes. Deep learning, its advanced counterpart, employs neural networks to analyze large datasets, enabling breakthroughs in fields like healthcare, autonomous vehicles, and finance.

                AI also excels in natural language processing (NLP), powering tools like chatbots and translation services. Advances in generative AI have created new possibilities for content creation, from generating art and music to drafting essays.

                Despite its potential, AI raises ethical concerns about bias, privacy, and job displacement. Responsible AI development emphasizes transparency, fairness, and inclusivity. As AI continues evolving, it holds the promise to revolutionize industries and improve lives, shaping the future of human-computer interaction."""
    text_cooking = """Cooking is both an art and a science, blending creativity with technique to transform raw ingredients into delicious meals. It serves as a universal language, connecting people across cultures and traditions through shared flavors and culinary practices. From baking bread to crafting elaborate dishes, cooking offers endless opportunities for experimentation and self-expression.

                    Key to successful cooking is understanding ingredients—their flavors, textures, and how they interact. Techniques such as sautéing, roasting, steaming, and grilling allow chefs to manipulate these qualities, bringing out the best in each component. Seasoning, a critical aspect, involves balancing salt, acid, fat, and heat to create harmonious dishes.

                    Home cooking promotes health and well-being, as it enables control over ingredients and portion sizes. It fosters a sense of accomplishment and nurtures relationships, whether through family dinners or festive gatherings.

                    Modern cooking is enriched by global influences and evolving technologies, such as precision tools and innovative recipes. Yet, the heart of cooking remains unchanged: it’s about nourishing both body and soul, turning simple ingredients into something extraordinary."""

    print(f"\n[bold green]Token count for AI text:[/bold green] {count_tokens(tokenizer, text_ai)}")
    tokenized_text = tokenizer(
        text_ai,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False  
    )
    print(f"[bold green]Tokens for AI text:[/bold green] {tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'].squeeze(0).tolist())}")
    print(f"[bold green]Token count for cooking text:[/bold green] {count_tokens(tokenizer, text_cooking)}")
    tokenized_text = tokenizer(
        text_cooking,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False  
    )
    print(f"[bold green]Tokens for cooking text:[/bold green] {tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'].squeeze(0).tolist())}")

    ########################################
    # 3. Embed AI text (split-and-merge)
    ########################################
    print("\n[bold yellow]Creating token embeddings for AI text...[/bold yellow]")
    df_ai = long_text_to_token_embedding_dataframe(
        text=text_ai,
        model=model,
        tokenizer=tokenizer,
        max_tokens=128,     # chunk size
        overlap_tokens=16,  # overlap
        batch_size=128,
        normalize=True
    )
    # Now pool to get a single vector
    pooled_ai = create_pooled_embeddings(df_ai, pooling_method="mean")

    ########################################
    # 4. Embed cooking text (split-and-merge)
    ########################################
    print("\n[bold yellow]Creating token embeddings for cooking text...[/bold yellow]")
    df_cooking = long_text_to_token_embedding_dataframe(
        text=text_cooking,
        model=model,
        tokenizer=tokenizer,
        max_tokens=128,
        overlap_tokens=16,
        batch_size=128,
        normalize=True
    )
    # Pool for a single vector
    pooled_cooking = create_pooled_embeddings(df_cooking, pooling_method="mean")

    ########################################
    # 5. Calculate distance and similarity
    ########################################
    distance_info = calculate_embedding_distance_between_dfs(pooled_ai, pooled_cooking)
    similarity_info = calculate_embedding_similarity_between_dfs(pooled_ai, pooled_cooking)

    print(f"\n[bold magenta]Distance between AI and Cooking texts:[/bold magenta] {distance_info['Distance']:.4f}")
    print(f"[bold magenta]Similarity between AI and Cooking texts:[/bold magenta] {similarity_info['Similarity']:.4f}")

if __name__ == "__main__":
    main()
