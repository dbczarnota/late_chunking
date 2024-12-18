
import re
from typing import List
from ragsyslib.files.engine_debugger import EngineDebugger
from .embed import count_tokens




def split_to_sentences(sentence_split_regex, text, tokenizer, token_limit):
    """
    Splits the input text into sentences and ensures each sentence does not exceed the token limit.

    Parameters:
    - sentence_split_regex (str): The regex pattern to split sentences.
    - text (str): The input text to be split.
    - tokenizer: HuggingFace tokenizer object.
    - token_limit (int): Maximum allowed tokens per sentence.

    Returns:
    - Tuple[List[str], List[Tuple[int, int]]]: A list of processed sentences and their span annotations.

    This function splits the text into sentences, checks their token count, and splits them further if needed.
    It also returns the start and end indices of each processed sentence in the original text.
    """
    single_sentences_list = re.split(sentence_split_regex, text)
    processed_sentences = []
    span_annotations = []
    current_index = 0

    for sentence in single_sentences_list:
        while count_tokens(tokenizer, sentence) > token_limit:
            tokenized_sentence = tokenizer(sentence, return_tensors="pt")
            tokens = tokenized_sentence.tokens()
            split_point = token_limit

            # Find the approximate split point based on token_limit
            sub_sentence = tokenizer.decode(tokenized_sentence.input_ids[0][:split_point], skip_special_tokens=True)
            start_index = text.find(sub_sentence, current_index)
            end_index = start_index + len(sub_sentence)
            processed_sentences.append(sub_sentence.strip())
            span_annotations.append((start_index, end_index))

            # Update the sentence with the remaining tokens
            sentence = tokenizer.decode(tokenized_sentence.input_ids[0][split_point:], skip_special_tokens=True)
            current_index = end_index

        start_index = text.find(sentence, current_index)
        end_index = start_index + len(sentence)
        processed_sentences.append(sentence.strip())
        span_annotations.append((start_index, end_index))
        current_index = end_index

    return processed_sentences, span_annotations