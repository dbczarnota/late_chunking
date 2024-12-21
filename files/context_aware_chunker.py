from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from ragsyslib.files.engine_debugger import EngineDebugger
from files.sentence_chunkers import split_to_sentences

import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine
from rich import print
import torch


class ContextAwareChunker:
    def __init__(
        self,
        tokenizer,
        model,
        pooling_method="mean",
        similarity_metric="cosine",
        max_sentence_length=500,
        min_sentence_length=15,
        sentence_split_regex=r'(?<=[.?!|])(?=\s+|\Z)|\n{1,2}(?=\S)',
        context_group_token_limit=8000,
        context_group_overlap_size=150,
        normalize_embeddings=False,
        num_neighbors=3,
        current_chunk_size=1,
        softmin_chunk_size=50,
        max_chunk_size=1000,
        min_prev_next_ratio=2.5,
        hard_split_prev_next_ratio=5.0,
        debugger=EngineDebugger("CONTEXT_AWARE_CHUNKER")
    ):
        """
        Initializes the ContextAwareChunker with the necessary parameters.

        Parameters:
        - tokenizer: A HuggingFace tokenizer object.
        - sentence_split_regex (str): The regex pattern used to split text into sentences.
        - max_sentence_length (int): Maximum allowed tokens per sentence for initial splitting.
        - min_sentence_length (int): Minimum token length threshold for final "long" sentences.
        - context_group_token_limit (int): Maximum number of tokens allowed in each context group.
        - context_group_overlap_size (int): Number of tokens by which consecutive groups overlap.
        - normalize_embeddings (bool): Whether to normalize the embeddings or not.
        - num_neighbors (int): Number of previous and next sentences to pool for distance comparison.
        - current_chunk_size (int): Number of sentences to consider as the current chunk.
        - debugger: Optional debugger or logger.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.pooling_method = pooling_method
        self.similarity_metric = similarity_metric
        self.max_sentence_length = max_sentence_length
        self.min_sentence_length = min_sentence_length
        self.sentence_split_regex = sentence_split_regex
        self.context_group_token_limit = context_group_token_limit
        self.context_group_overlap_size = context_group_overlap_size
        self.normalize_embeddings = normalize_embeddings
        self.num_neighbors = num_neighbors
        self.current_chunk_size = current_chunk_size
        self.softmin_chunk_size = softmin_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_prev_next_ratio = min_prev_next_ratio
        self.hard_split_prev_next_ratio = hard_split_prev_next_ratio
        self.debugger = debugger

        if self.debugger:
            self.debugger.debug("init", "Initializing ContextAwareChunker.")
    
    
    def text_to_token_embeddings(self, text, skip_beginning=0, skip_end=0):
        """
        Given a model and tokenizer from HuggingFace, return token embeddings of the input text,
        dynamically optimizing for CUDA or CPU, with the option to return embeddings for a subset of tokens.

        Args:
            model: HuggingFace model object.
            tokenizer: HuggingFace tokenizer object.
            text (str): Input text to be tokenized and processed.
            batch_size (int, optional): Maximum number of tokens to process in one batch.
            skip_beginning (int, optional): Number of tokens to skip from the beginning when returning embeddings.
            skip_end (int, optional): Number of tokens to skip from the end when returning embeddings.

        Returns:
            Tuple[torch.Tensor, List[str]]: Token embeddings of the subset of tokens and the corresponding tokens.
        """


        # Check for CUDA availability
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Using device: {device}")

        # Move model to appropriate device
        model = self.model.to(device)

        if self.context_group_token_limit > 8192:  # Ensure batch size is within limit
            raise ValueError("Batch size is too large. Please use a batch size of 8192 or less.")

        # Tokenize the input text
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"].squeeze(0).tolist())

        # Move tokenized inputs to device
        tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}

        # Batch process the input
        outputs = []
        for i in range(0, tokenized_text["input_ids"].size(1), self.context_group_token_limit):
            start = i
            end = min(i + self.context_group_token_limit, tokenized_text["input_ids"].size(1))

            # Subset tokenized inputs for the current batch
            batch_inputs = {k: v[:, start:end] for k, v in tokenized_text.items()}

            # Compute embeddings with no gradient computation
            with torch.no_grad():
                model_output = model(**batch_inputs)

            outputs.append(model_output.last_hidden_state)

        # Concatenate outputs along the token dimension
        all_embeddings = torch.cat(outputs, dim=1)

        # Apply skip_beginning and skip_end to the embeddings
        if skip_beginning + skip_end >= all_embeddings.size(1):
            raise ValueError("The combination of skip_beginning and skip_end is greater than or equal to the number of tokens.")

        subset_embeddings = all_embeddings[:, skip_beginning:all_embeddings.size(1) - skip_end, :]
        subset_tokens = tokens[skip_beginning:len(tokens) - skip_end if skip_end > 0 else None]

        return subset_embeddings, subset_tokens
    
    def count_tokens(self, tokenizer, text):
        """
        Count the number of tokens in the text using the tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer object.
            text: The input text (string) to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return len(tokenized_text.input_ids[0])

    def split_to_sentences(self, sentence_split_regex, text, tokenizer, token_limit):
        """
        Splits the input text into sentences and ensures each sentence does not exceed the token limit.

        Parameters:
        - sentence_split_regex (str): The regex pattern to split sentences.
        - text (str): The input text to be split.
        - tokenizer: HuggingFace tokenizer object.
        - token_limit (int): Maximum allowed tokens per sentence.

        Returns:
        - List[str]: A list of processed sentences.
        """
        text = text.strip()
        single_sentences_list = list(filter(None, re.split(sentence_split_regex, text)))

        # single_sentences_list = re.split(sentence_split_regex, text)
        print(f"Single sentences list: {single_sentences_list}")
        processed_sentences = []

        for sentence in single_sentences_list:
            while self.count_tokens(tokenizer, sentence) > token_limit:
                tokenized_sentence = tokenizer(sentence, return_tensors="pt")
                split_point = token_limit

                # Find the approximate split point based on token_limit
                sub_sentence = tokenizer.decode(tokenized_sentence.input_ids[0][:split_point], skip_special_tokens=True)
                processed_sentences.append(sub_sentence.strip())

                # Update the sentence with the remaining tokens
                sentence = tokenizer.decode(tokenized_sentence.input_ids[0][split_point:], skip_special_tokens=True)

            processed_sentences.append(sentence.strip())

        return processed_sentences


    def clean_text(self, text): 
        """
        Cleans the text by removing additional newlines, tabs, and unnecessary special characters.
        Converts multiple newlines into a single newline.

        Parameters:
        text (str): The text to be cleaned.

        Returns:
        str: The cleaned text.
        """
        # Strip leading and trailing whitespace
        text = text.strip()
        
        # Replace multiple newlines and tabs with a single space
        text = re.sub(r'\r', '', text)
        
        # Replace multiple newlines and tabs with a single space
        text = re.sub(r'[\t]+', ' ', text)

        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)

        # replace all new lines with pipe
        text = re.sub(r'\n', '| ', text)

        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)



        return text   

    def normalize_embeddings_fc(self, embeddings):
        """
        Normalize embeddings to have unit norm.

        Args:
            embeddings (Union[torch.Tensor, np.ndarray]): The embeddings to normalize.

        Returns:
            torch.Tensor or np.ndarray: Normalized embeddings.
        """
        if isinstance(embeddings, torch.Tensor):
            # Move tensor to CPU if necessary
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()

            # Squeeze batch dimension if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)

            # Normalize tensor embeddings directly
            norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            return embeddings / norms

        elif isinstance(embeddings, np.ndarray):
            # Squeeze batch dimension if present
            if embeddings.ndim == 3:
                embeddings = embeddings.squeeze(0)

            # Normalize numpy embeddings
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            return embeddings / norms

        else:
            raise TypeError(f"Embeddings must be either a numpy array or a torch tensor. Got type: {type(embeddings)}")

   
    def split_to_long_sentences(self, text):
        """
        Splits the text into sentences using split_to_sentences, joins consecutive short sentences,
        and returns long sentences along with their token counts and token span annotations.

        Returns:
        - List[Tuple[str, int, Tuple[int, int]]]: A list of tuples where each tuple contains:
        - Sentence/chunk (str)
        - Token count (int)
        - Token span (Tuple[int, int]) in the tokenized representation of the original text
        """
        # Step 1: Split text into sentences
        sentences = split_to_sentences(
            self.sentence_split_regex, 
            text, 
            self.tokenizer, 
            self.max_sentence_length
        )

        joined_sentences = []
        buffer = []
        buffer_length = 0
        buffer_token_start = 0  # Start token index for the current buffer
        token_spans = []

        # Step 2: Tokenize the full text for span annotations
        tokenized_full_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenized_full_text.tokens()
        offsets = tokenized_full_text.offset_mapping[0]

        # Step 3: Iterate through the sentences
        for s in sentences:
            s_length = self.count_tokens(self.tokenizer, s)
            tokenized_sentence = self.tokenizer(s, add_special_tokens=False, return_tensors="pt")
            sentence_token_count = tokenized_sentence.input_ids.size(1)

            # Add the sentence to the buffer
            buffer.append(s)
            buffer_length += sentence_token_count

            # If the buffer length exceeds max_sentence_length, split it
            if buffer_length > self.max_sentence_length:
                chunk = []
                chunk_length = 0
                chunk_start_token = buffer_token_start

                while buffer and chunk_length + self.count_tokens(self.tokenizer, buffer[0]) <= self.max_sentence_length:
                    sentence = buffer.pop(0)
                    chunk.append(sentence)
                    sentence_length = self.count_tokens(self.tokenizer, sentence)
                    chunk_length += sentence_length

                # Determine token span for the chunk
                chunk_end_token = chunk_start_token + chunk_length
                joined_sentences.append((" ".join(chunk), chunk_length, (chunk_start_token, chunk_end_token)))

                # Update buffer and token start for the next chunk
                buffer_length -= chunk_length
                buffer_token_start = chunk_end_token

            # If the buffer length meets or exceeds the minimum sentence length, finalize it
            if buffer_length >= self.min_sentence_length:
                joined_sentence = " ".join(buffer)
                sentence_length = self.count_tokens(self.tokenizer, joined_sentence)
                token_start = buffer_token_start
                token_end = buffer_token_start + sentence_length

                joined_sentences.append((joined_sentence, sentence_length, (token_start, token_end)))

                buffer = []
                buffer_length = 0
                buffer_token_start = token_end

        # If anything remains in the buffer at the end, append it as a separate sentence
        if buffer:
            joined_sentence = " ".join(buffer)
            sentence_length = self.count_tokens(self.tokenizer, joined_sentence)
            token_start = buffer_token_start
            token_end = buffer_token_start + sentence_length

            joined_sentences.append((joined_sentence, sentence_length, (token_start, token_end)))
            
        print("\n[bold blue]Long Sentences:[/bold blue]")
        for i, (sentence, token_count, _) in enumerate(joined_sentences):
            print(f"Sentence {i + 1}: {sentence} (Tokens: {token_count})")
            
        return joined_sentences


    def prepare_context_groups(self, long_sentences):
        """
        Given a list of sentences (from split_to_long_sentences), divides them into context groups.

        Each group:
        - Cannot exceed `self.context_group_token_limit` tokens (hard limit).
        - Should overlap with the next group by approximately `self.context_group_overlap_size` tokens.
        - Avoid splitting sentences.

        Returns:
        List[Tuple[str, int]]: A list of tuples, each containing:
                            (group_text, overlap_with_next_group_tokens)
        """
        groups = []
        current_group = []
        current_group_length = 0

        for i, (sentence, sentence_length, sentence_span_annotations) in enumerate(long_sentences):
            # Check if adding the sentence would exceed the group limit
            if current_group_length + sentence_length > self.context_group_token_limit:
                # Finalize current group before adding this new sentence
                full_group_text = " ".join(s for s, _ in current_group)

                # Extract overlap from the tail of the current_group
                overlap_sentences = []
                overlap_tokens = self.context_group_overlap_size

                # We'll move backwards from the end of current_group
                temp_group = current_group[:]
                overlap_collected = 0
                while overlap_tokens > 0 and temp_group:
                    last_sentence, last_length = temp_group.pop()
                    if last_length <= overlap_tokens:
                        overlap_sentences.insert(0, (last_sentence, last_length))
                        overlap_tokens -= last_length
                        overlap_collected += last_length
                    else:
                        # Can't fully add this sentence to overlap because it exceeds the overlap limit
                        # We only take full sentences for overlap
                        break

                print(f"Finalized group {len(groups)} with total tokens: {current_group_length}")
                print(f"Overlap selected for next group: {overlap_collected} tokens")

                # Append the finalized group
                groups.append((full_group_text, overlap_collected))

                # Now start a new group from the overlap sentences
                current_group = overlap_sentences
                current_group_length = sum(l for _, l in current_group)

                # Add the current sentence that triggered the new group
                current_group.append((sentence, sentence_length))
                current_group_length += sentence_length
            else:
                # Just add the sentence to the current group
                current_group.append((sentence, sentence_length))
                current_group_length += sentence_length

        # Handle the last group if any sentences remain
        if current_group:
            print(f"Finalized last group {len(groups)} with total tokens: {current_group_length}")
            print("No next group, overlap: 0 tokens")
            groups.append((" ".join(s for s, _ in current_group), 0))

        return groups

        
    def create_token_embeddings(self, groups):
        """
        Create token embeddings for each context group while handling overlaps.

        Approach:
        - For the first group: skip_end = half overlap; skip_beginning = 0
        - For subsequent groups: skip_end = half overlap;
        skip_beginning = number of leading tokens that appear in the previous group's adjusted tokens,
        stopping as soon as we find a token not present in the previous group's adjusted tokens.
        """

        results = []
        previous_adjusted_tokens = []

        for i, (group_text, overlap_with_next_tokens) in enumerate(groups):
            # Half of current group's overlap at the end
            skip_end = overlap_with_next_tokens // 2

            # Tokenize the current group
            tokenized = self.tokenizer(group_text, return_offsets_mapping=True, add_special_tokens=False)
            tokens = tokenized.tokens()

            if i == 0:
                # First group:
                skip_beginning = 0
            else:
                skip_beginning = 0
            # Find the longest suffix of previous_adjusted_tokens that matches a prefix of tokens
            max_possible_overlap = min(len(previous_adjusted_tokens), len(tokens))
            for n in range(max_possible_overlap, 0, -1):
                if previous_adjusted_tokens[-n:] == tokens[:n]:
                    skip_beginning = n
                    break

            # Debug:
            print(f"Processing Group {i+1}/{len(groups)}:")
            print(f"Full Group Text: {group_text}")
            print(f"Overlap with next group (full): {overlap_with_next_tokens} tokens")
            print(f"Skip Beginning: {skip_beginning}")
            print(f"Skip End (half current overlap): {skip_end}")
            print(f"Original Tokens ({len(tokens)}): {tokens}")

            group_embeddings = self.text_to_token_embeddings(

                text=group_text,

                skip_beginning=skip_beginning,
                skip_end=skip_end,
            )

            # Normalize embeddings if specified
            if self.normalize_embeddings:
                group_embeddings = self.normalize_embeddings_fc(group_embeddings[0])

            # Adjust tokens based on skipping
            adjusted_tokens = tokens[skip_beginning: len(tokens) - skip_end if skip_end > 0 else None]
            print(f"Adjusted Tokens ({len(adjusted_tokens)}): {adjusted_tokens}\n")

            results.append((adjusted_tokens, group_embeddings))
            previous_adjusted_tokens = adjusted_tokens  # Update for the next iteration

        return results



    def combine_group_tables(self, token_embeddings_with_tokens):
        """
        Combines the token embeddings and their corresponding tokens from multiple groups into a single table.

        Parameters:
        - token_embeddings_with_tokens (List[Tuple[List[str], torch.Tensor]]): 
            A list where each element is a tuple containing:
            - tokens: List of tokens in the group.
            - embeddings: Corresponding embeddings (torch.Tensor).

        Returns:
        - pd.DataFrame: A DataFrame where each row corresponds to a token and its embedding.
        """
        combined_data = []
        for group_index, (tokens, embeddings) in enumerate(token_embeddings_with_tokens):
            # Handle tuple case for embeddings
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]  # Extract the actual tensor part

            # Remove the batch dimension if present
            if len(embeddings.shape) == 3 and embeddings.size(0) == 1:
                embeddings = embeddings.squeeze(0)  # Squeeze the batch dimension
            
            # Iterate through tokens and their corresponding embeddings
            for token_index, token in enumerate(tokens):
                if token_index >= embeddings.size(0):
                    raise IndexError(
                        f"Token index {token_index} out of bounds for embeddings with size {embeddings.size(0)}"
                    )
                embedding = embeddings[token_index]
                combined_data.append({
                    "Group": group_index,
                    "Token": token,
                    # "Embedding": embedding.cpu().numpy()  # Convert to NumPy for compatibility with Pandas
                    "Embedding": embedding.cpu().to(torch.float32).numpy()
                })

        # Convert to a Pandas DataFrame for easier manipulation and visualization
        df = pd.DataFrame(combined_data)
        return df


    def pool_embeddings(self, embeddings):
        """
        Pools the given embeddings using the class-defined pooling method.

        Args:
            embeddings (np.ndarray): A 2D array of embeddings to pool (shape: [num_tokens, embedding_dim]).

        Returns:
            np.ndarray: The pooled embedding (1D array).
        """
        if self.pooling_method == "mean":
            return np.mean(embeddings, axis=0)
        elif self.pooling_method == "max":
            return np.max(embeddings, axis=0)
        elif self.pooling_method == "min":
            return np.min(embeddings, axis=0)
        elif self.pooling_method == "sum":
            return np.sum(embeddings, axis=0)
        elif self.pooling_method == "median":
            return np.median(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")


    def generate_pooled_embeddings(self, span_annotations, combined_table):
        """
        Generates pooled embeddings for spans of tokens.

        Args:
            span_annotations (List[Tuple[int, int]]): List of (start_token, end_token) indices for spans.
            combined_table (pd.DataFrame): DataFrame containing token information and embeddings.

        Returns:
            List[Tuple[np.ndarray, List[str]]]: A list of tuples where each tuple contains:
                - The pooled embedding (np.ndarray).
                - The list of tokens in the span.
        """
        pooled_results = []

        for start_token, end_token in span_annotations:
            # Extract token data and embeddings for the span
            span_data = combined_table.iloc[start_token:end_token]
            span_embeddings = np.stack(span_data["Embedding"].values)

            # Use the helper method to pool embeddings
            pooled_embedding = self.pool_embeddings(span_embeddings)

            tokens = span_data["Token"].tolist()
            pooled_results.append((pooled_embedding, tokens))

        return pooled_results



    def compare_chunk_distances(self, combined_table, span_annotations):
        pooled_results = self.generate_pooled_embeddings(span_annotations, combined_table)

        embedding1, tokens1 = pooled_results[0]
        embedding2, tokens2 = pooled_results[1]

        if self.similarity_metric == "cosine":
            distance = cosine(embedding1, embedding2)
        elif self.similarity_metric == "euclidean":
            distance = np.linalg.norm(embedding1 - embedding2)
        elif self.similarity_metric == "manhattan":
            distance = np.sum(np.abs(embedding1 - embedding2))
        elif self.similarity_metric == "dot_product":
            distance = np.dot(embedding1, embedding2)
        elif self.similarity_metric == "pearson":
            distance = np.corrcoef(embedding1, embedding2)[0, 1]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return {
            "chunk1_chunk2": {
                "distance": distance,
                "tokens1": tokens1,
                "tokens2": tokens2,
            }
        }


    def compare_sentence_distances(self, text):
        """
        Compare distances between sentences using embeddings.

        Args:
            text (str): Input text to process and compare.

        Returns:
            List[Dict]: Each entry contains the sentence, its distances to the pooled previous and next sentences,
                        the corresponding tokens, and now also a "sentence_embedding".
        """
        # Step 1: Split text into long sentences
        long_sentences = self.split_to_long_sentences(text)

        # Step 2: Prepare context groups from long_sentences
        context_groups = self.prepare_context_groups(long_sentences)

        # Step 3: Generate token embeddings for context groups
        token_embeddings_with_tokens = self.create_token_embeddings(context_groups)

        # Step 4: Combine group tables
        combined_table = self.combine_group_tables(token_embeddings_with_tokens)

        # Debugging: Check the structure of combined_table
        print(f"Combined table sample:\n{combined_table.head()}")

        # Step 5: Compare distances for each sentence
        results = []

        for i in range(len(long_sentences)):
            # Extract token span for the current sentence
            sentence_span = long_sentences[i][2]
            start_token, end_token = sentence_span

            # Filter combined_table rows using the token span
            tokens_for_original_sentence = combined_table.iloc[start_token:end_token]["Token"].tolist()

            # Debug: Check if tokens were correctly retrieved
            if not tokens_for_original_sentence:
                print(f"[Warning] No tokens found for Sentence {i} with Span: {sentence_span}")

            result = {
                "original_sentence": tokens_for_original_sentence,  # Tokens of the original sentence
                "sentence_number": i,
                "current_sentence": None,
                "previous": None,
                "next": None,
                "prev_next_ratio": None,
                # We'll add "sentence_embedding" below
            }

            # Pool current chunk of sentences
            start_idx = max(0, i)
            end_idx = min(len(long_sentences), i + self.current_chunk_size)
            current_spans = [long_sentences[j][2] for j in range(start_idx, end_idx)]
            pooled_current_span = (current_spans[0][0], current_spans[-1][1])

            current_sentence = " ".join([long_sentences[j][0] for j in range(start_idx, end_idx)])
            result["current_sentence"] = current_sentence

            # -----------------------------
            # (A) Single-sentence embedding
            # -----------------------------
            single_sentence_data = combined_table.iloc[start_token:end_token]
            single_sentence_embeddings = np.stack(single_sentence_data["Embedding"].values)
            single_sentence_pooled = self.pool_embeddings(single_sentence_embeddings)
            result["sentence_embedding"] = single_sentence_pooled
            # End of the single-sentence embedding addition

            # Pool previous sentences if available
            if start_idx > 0:
                prev_start_idx = max(0, start_idx - self.num_neighbors)
                previous_spans = [long_sentences[j][2] for j in range(prev_start_idx, start_idx)]
                pooled_previous_span = (previous_spans[0][0], previous_spans[-1][1])
                distances = self.compare_chunk_distances(combined_table, [pooled_previous_span, pooled_current_span])

                result["previous"] = {
                    "distance": distances['chunk1_chunk2']['distance'],
                    "tokens1": distances['chunk1_chunk2']['tokens1'],
                    "tokens2": distances['chunk1_chunk2']['tokens2']
                }

            # Pool next sentences if available
            if end_idx < len(long_sentences):
                next_end_idx = min(len(long_sentences), end_idx + self.num_neighbors)
                next_spans = [long_sentences[j][2] for j in range(end_idx, next_end_idx)]
                pooled_next_span = (next_spans[0][0], next_spans[-1][1])
                distances = self.compare_chunk_distances(combined_table, [pooled_current_span, pooled_next_span])

                result["next"] = {
                    "distance": distances['chunk1_chunk2']['distance'],
                    "tokens1": distances['chunk1_chunk2']['tokens1'],
                    "tokens2": distances['chunk1_chunk2']['tokens2']
                }

            # Calculate prev/next ratio if both distances are available
            prev_distance = result['previous']['distance'] if result['previous'] else None
            next_distance = result['next']['distance'] if result['next'] else None

            if prev_distance is not None and next_distance is not None:
                result["prev_next_ratio"] = prev_distance / next_distance

            results.append(result)

        # Informative Output Block
        print("\n[Summary of Results]")
        for res in results:
            print(f"\nSentence {res['sentence_number']}: {res['original_sentence']}")
            print(f"Current Chunk: {res['current_sentence']}")
            if res['previous']:
                print(f"  Distance to Previous: {res['previous']['distance']:.4f}")
                print(f"  Tokens (Prev -> Current): {res['previous']['tokens1']} => {res['previous']['tokens2']}")
            else:
                print("  No pooled previous comparison available.")
            if res['next']:
                print(f"  Distance to Next: {res['next']['distance']:.4f}")
                print(f"  Tokens (Current -> Next): {res['next']['tokens1']} => {res['next']['tokens2']}")
            else:
                print("  No pooled next comparison available.")
            if res["prev_next_ratio"] is not None:
                print(f"  Prev/Next Distance Ratio: {res['prev_next_ratio']:.4f}")
            else:
                print("  Prev/Next Distance Ratio: N/A")

        return results



    def create_chunks(self, text):
        """
        Create chunks from text based on softmin_chunk_size, max_chunk_size, and prev/next ratios.
        Ensures chunks start with the correct sentences after splits.

        Args:
            text (str): The input text to chunk.

        Returns:
            List[Tuple[str, str, np.ndarray]]:
                [
                (chunk_text, starting_point_ratio, chunk_embedding),
                ...
                ]
        """
        # Step 1: Compare sentence distances (same as before)
        distance_results = self.compare_sentence_distances(text)

        chunks = []
        current_chunk = []
        current_chunk_embeddings = []
        current_token_count = 0
        current_max_ratio = -float("inf")
        starting_point_ratio = "N/A"  # Default for the first chunk

        for i, result in enumerate(distance_results):
            sentence = result["current_sentence"]
            sentence_embedding = result["sentence_embedding"]  # The single-sentence embedding
            prev_next_ratio = result["prev_next_ratio"]
            token_count = self.count_tokens(self.tokenizer, sentence)

            # Add sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_embeddings.append(sentence_embedding)
            current_token_count += token_count

            # Preserve your existing ratio logic
            if prev_next_ratio is not None and prev_next_ratio > current_max_ratio:
                current_max_ratio = prev_next_ratio

            # Check for hard split
            if prev_next_ratio is not None and prev_next_ratio >= self.hard_split_prev_next_ratio:
                # Finalize the current chunk
                chunk_text = " ".join(current_chunk)
                # Use pool_embeddings on the stacked embeddings instead of np.mean
                chunk_embedding = self.pool_embeddings(np.stack(current_chunk_embeddings))

                chunks.append((chunk_text, starting_point_ratio, chunk_embedding))

                # Start a new chunk with the current sentence
                current_chunk = [sentence]
                current_chunk_embeddings = [sentence_embedding]
                current_token_count = token_count
                current_max_ratio = -float("inf")
                starting_point_ratio = f"{prev_next_ratio:.4f}"
                continue

            # Check for softmin_chunk_size
            if current_token_count >= self.softmin_chunk_size:
                if (
                    current_token_count >= self.max_chunk_size
                    or (prev_next_ratio is not None and prev_next_ratio > self.min_prev_next_ratio)
                ):
                    # Finalize the current chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_embedding = self.pool_embeddings(np.stack(current_chunk_embeddings))

                    chunks.append((chunk_text, starting_point_ratio, chunk_embedding))

                    # Start a new chunk
                    current_chunk = [sentence]
                    current_chunk_embeddings = [sentence_embedding]
                    current_token_count = token_count
                    current_max_ratio = -float("inf")
                    starting_point_ratio = f"{prev_next_ratio:.4f}"

        # Add any remaining text as the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_embedding = self.pool_embeddings(np.stack(current_chunk_embeddings))
            chunks.append((chunk_text, starting_point_ratio, chunk_embedding))

        # Print the resulting chunks, just like in your original code
        print("\nChunks:")
        for chunk, ratio, embedding in chunks:
            print(f"Chunk: {chunk[:100]}...\nStarting Point Split Ratio: {ratio}")
            print(f"Embedding (first 5 dims): {embedding[:5]}\n")

        return chunks

