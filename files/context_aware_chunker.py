from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from ragsyslib.files.engine_debugger import EngineDebugger
from files.sentence_chunkers import split_to_sentences, count_tokens
from files.embed import text_to_token_embeddings

import re


class ContextAwareChunker:
    def __init__(
        self,
        tokenizer,
        model,
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
        and returns long sentences along with their token counts.

        Returns:
        - List[Tuple[str, int]]: A list of tuples where each tuple contains a sentence/chunk
        and its token count.
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

        # Step 2: Iterate through the sentences
        for s in sentences:
            s_length = count_tokens(self.tokenizer, s)

            # Add the sentence to the buffer
            buffer.append(s)
            buffer_length += s_length

            # If the buffer length exceeds max_sentence_length, split it
            if buffer_length > self.max_sentence_length:
                chunk = []
                chunk_length = 0

                while buffer and chunk_length + count_tokens(self.tokenizer, buffer[0]) <= self.max_sentence_length:
                    sentence = buffer.pop(0)
                    chunk.append(sentence)
                    chunk_length += count_tokens(self.tokenizer, sentence)

                joined_sentences.append((" ".join(chunk), chunk_length))
                buffer_length -= chunk_length

            # If the buffer length meets or exceeds the minimum sentence length, finalize it
            if buffer_length >= self.min_sentence_length:
                joined_sentence = " ".join(buffer)
                joined_sentences.append((joined_sentence, buffer_length))
                buffer = []
                buffer_length = 0

        # If anything remains in the buffer at the end, append it as a separate sentence
        if buffer:
            joined_sentence = " ".join(buffer)
            joined_sentences.append((joined_sentence, buffer_length))

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

        for i, (sentence, sentence_length) in enumerate(long_sentences):
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
