from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from ragsyslib.files.engine_debugger import EngineDebugger
from files.sentence_chunkers import split_to_sentences, count_tokens
from files.embed import text_to_token_embeddings
import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine
from rich import print


class ContextAwareChunker:
    def __init__(
        self,
        tokenizer,
        model,
        pooling_method="mean",
        similarity_metric = "cosine",
        max_sentence_length: int = 500,
        min_sentence_length: int = 20,
        sentence_split_regex: str = r'(?<=[.?!])\s+',
        context_group_token_limit: int = 200,
        context_group_overlap_size: int = 50,
        debugger = EngineDebugger("CONTEXT_AWARE_CHUNKER")
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
        - debugger: Optional debugger or logger.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.pooling_method = pooling_method
        self.similarity_metric = similarity_metric
        self.sentence_split_regex = sentence_split_regex
        self.max_sentence_length = max_sentence_length
        self.min_sentence_length = min_sentence_length
        self.context_group_token_limit = context_group_token_limit
        self.context_group_overlap_size = context_group_overlap_size
        self.debugger = debugger

        if self.debugger:
            self.debugger.debug("init", "Initializing ContextAwareChunker.")
  
    
    
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
            s_length = count_tokens(self.tokenizer, s)
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

                while buffer and chunk_length + count_tokens(self.tokenizer, buffer[0]) <= self.max_sentence_length:
                    sentence = buffer.pop(0)
                    chunk.append(sentence)
                    sentence_length = count_tokens(self.tokenizer, sentence)
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
                sentence_length = count_tokens(self.tokenizer, joined_sentence)
                token_start = buffer_token_start
                token_end = buffer_token_start + sentence_length

                joined_sentences.append((joined_sentence, sentence_length, (token_start, token_end)))

                buffer = []
                buffer_length = 0
                buffer_token_start = token_end

        # If anything remains in the buffer at the end, append it as a separate sentence
        if buffer:
            joined_sentence = " ".join(buffer)
            sentence_length = count_tokens(self.tokenizer, joined_sentence)
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

            group_embeddings = text_to_token_embeddings(
                model=self.model,
                tokenizer=self.tokenizer,
                text=group_text,
                batch_size=self.context_group_token_limit,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
            )

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
                    "Embedding": embedding.cpu().numpy()  # Convert to NumPy for compatibility with Pandas
                })

        # Convert to a Pandas DataFrame for easier manipulation and visualization
        df = pd.DataFrame(combined_data)
        return df


    def generate_pooled_embeddings(self, span_annotations, combined_table):
        pooled_results = []

        for start_token, end_token in span_annotations:
            span_data = combined_table.iloc[start_token:end_token]
            span_embeddings = np.stack(span_data["Embedding"].values)

            if self.pooling_method == "mean":
                pooled_embedding = np.mean(span_embeddings, axis=0)
            elif self.pooling_method == "max":
                pooled_embedding = np.max(span_embeddings, axis=0)
            elif self.pooling_method == "min":
                pooled_embedding = np.min(span_embeddings, axis=0)
            elif self.pooling_method == "sum":
                pooled_embedding = np.sum(span_embeddings, axis=0)
            elif self.pooling_method == "median":
                pooled_embedding = np.median(span_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling_method}")

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


    def compare_sentence_distances(self, text, num_neighbors=3, current_chunk_size=1):
        """
        Compare distances between sentences using embeddings.

        Args:
            text (str): Input text to process and compare.
            num_neighbors (int): Number of previous and next sentences to pool.
            current_chunk_size (int): Number of sentences to consider as the current chunk.

        Returns:
            List[Dict]: Each entry contains the sentence, its distances to the pooled previous and next sentences,
            and the corresponding tokens.
        """
        # Step 1: Split text into long sentences
        long_sentences = self.split_to_long_sentences(text)

        # Step 2: Prepare context groups from long sentences
        context_groups = self.prepare_context_groups(long_sentences)

        # Step 3: Generate token embeddings for context groups
        token_embeddings_with_tokens = self.create_token_embeddings(context_groups)

        # Step 4: Combine group tables
        combined_table = self.combine_group_tables(token_embeddings_with_tokens)

        # Step 5: Compare distances for each sentence
        results = []

        for i in range(len(long_sentences)):
            result = {
                "current_sentence": None,
                "previous": None,
                "next": None
            }

            # Pool current chunk of sentences
            start_idx = max(0, i)
            end_idx = min(len(long_sentences), i + current_chunk_size)
            current_spans = [long_sentences[j][2] for j in range(start_idx, end_idx)]
            pooled_current_span = (current_spans[0][0], current_spans[-1][1])

            current_sentence = " ".join([long_sentences[j][0] for j in range(start_idx, end_idx)])
            result["current_sentence"] = current_sentence

            if start_idx == end_idx - 1:
                print(f"\n[bold yellow]Current Sentence ({start_idx}):[/bold yellow] {current_sentence}")
            else:
                print(f"\n[bold yellow]Current Sentence (Chunk {start_idx}-{end_idx - 1}):[/bold yellow] {current_sentence}")

            # Pool previous sentences if available
            if start_idx > 0:
                prev_start_idx = max(0, start_idx - num_neighbors)
                previous_spans = [long_sentences[j][2] for j in range(prev_start_idx, start_idx)]
                pooled_previous_span = (previous_spans[0][0], previous_spans[-1][1])
                distances = self.compare_chunk_distances(combined_table, [pooled_previous_span, pooled_current_span])

                result["previous"] = {
                    "distance": distances['chunk1_chunk2']['distance'],
                    "tokens1": distances['chunk1_chunk2']['tokens1'],
                    "tokens2": distances['chunk1_chunk2']['tokens2']
                }

                print(f"Distance to pooled previous: {distances['chunk1_chunk2']['distance']:.4f}")
                print(f"Tokens: {distances['chunk1_chunk2']['tokens1']} => {distances['chunk1_chunk2']['tokens2']}")
            else:
                print("No pooled previous sentence comparison available.")

            # Pool next sentences if available
            if end_idx < len(long_sentences):
                next_end_idx = min(len(long_sentences), end_idx + num_neighbors)
                next_spans = [long_sentences[j][2] for j in range(end_idx, next_end_idx)]
                pooled_next_span = (next_spans[0][0], next_spans[-1][1])
                distances = self.compare_chunk_distances(combined_table, [pooled_current_span, pooled_next_span])

                result["next"] = {
                    "distance": distances['chunk1_chunk2']['distance'],
                    "tokens1": distances['chunk1_chunk2']['tokens1'],
                    "tokens2": distances['chunk1_chunk2']['tokens2']
                }

                print(f"Distance to pooled next: {distances['chunk1_chunk2']['distance']:.4f}")
                print(f"Tokens: {distances['chunk1_chunk2']['tokens1']} => {distances['chunk1_chunk2']['tokens2']}")
            else:
                print("No pooled next sentence comparison available.")

            prev_distance = result['previous']['distance'] if result['previous'] else None
            next_distance = result['next']['distance'] if result['next'] else None

            if prev_distance is not None and next_distance is not None:
                print(f"Distance to pooled previous / Distance to pooled next: {prev_distance / next_distance:.4f}")

            results.append(result)

        return results
