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
        debugger = EngineDebugger("CONTEXT_AWARE_CHUNKER")
    ):
        """
        Initializes the ContextAwareChunker with the necessary parameters.

        Parameters:
        - tokenizer: A HuggingFace tokenizer object.
        - sentence_split_regex (str): The regex pattern used to split text into sentences.
        - max_sentence_length (int): Maximum allowed tokens per sentence when initially splitting.
        - sentence_length (int): Minimum token length threshold for final chunks.
        - debugger: Optional debugger or logger.
        """
        self.tokenizer = tokenizer
        self.sentence_split_regex = sentence_split_regex
        self.max_sentence_length = max_sentence_length
        self.min_sentence_length = min_sentence_length
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
        # Replace multiple newlines and tabs with a single space
        text = re.sub(r'\r', '', text)
        
        # Replace multiple newlines and tabs with a single space
        text = re.sub(r'[\t]+', ' ', text)

        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)

        # replace all new lines with pipe
        text = re.sub(r'\n', ' | ', text)

        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)

        # Strip leading and trailing whitespace
        text = text.strip()

        return text   
    
    
    def split_to_long_sentences(self, text):
        """
        Splits the text into sentences using split_to_sentences, then joins consecutive short sentences.

        Returns:
        - List[str]: A list of sentences/chunks where each is at least `self.sentence_length` tokens long.
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

            # If adding this sentence to the current buffer doesn't reach the threshold,
            # accumulate it in the buffer.
            if buffer_length + s_length < self.min_sentence_length:
                buffer.append(s)
                buffer_length += s_length
            else:
                # If we already have something in the buffer, finalize it
                if buffer:
                    joined_sentences.append(" ".join(buffer))
                    buffer = []
                    buffer_length = 0

                # Check if the current sentence alone is short
                if s_length < self.min_sentence_length:
                    # Start a new buffer with this sentence
                    buffer.append(s)
                    buffer_length = s_length
                else:
                    # This sentence alone meets the requirement
                    joined_sentences.append(s)

        # If anything remains in the buffer at the end, append it
        if buffer:
            joined_sentences.append(" ".join(buffer))

        return joined_sentences