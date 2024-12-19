from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory of the current script
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from ragsyslib.files.engine_debugger import EngineDebugger
from files.sentence_chunkers import split_to_sentences, count_tokens

import re


class ContextAwareChunker:
    def __init__(
        self,
        tokenizer,
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
        List[str]: A list of context groups.
        """
        groups = []
        current_group = []
        current_group_length = 0

        # Iterate through the sentences and their lengths
        for i, (sentence, sentence_length) in enumerate(long_sentences):
            # Check if adding the sentence exceeds the group limit
            if current_group_length + sentence_length > self.context_group_token_limit:
                # Finalize the current group
                groups.append(" ".join(current_group))

                # Start a new group with overlap
                overlap_sentences = []
                overlap_tokens = self.context_group_overlap_size
                while overlap_tokens > 0 and current_group:
                    last_sentence = current_group.pop()
                    last_length = count_tokens(self.tokenizer, last_sentence)
                    if last_length <= overlap_tokens:
                        overlap_sentences.insert(0, last_sentence)
                        overlap_tokens -= last_length
                    else:
                        break  # Stop if the sentence cannot fit entirely into the overlap

                current_group = overlap_sentences
                current_group_length = sum(count_tokens(self.tokenizer, s) for s in current_group)

            # Add current sentence to the group
            current_group.append(sentence)
            current_group_length += sentence_length

        # Add the final group if any sentences remain
        if current_group:
            groups.append(" ".join(current_group))

        return groups
