
import re
import spacy
from spacy.tokens import Doc
from typing import List
from ragsyslib.files.engine_debugger import EngineDebugger




def split_to_sentences(sentence_split_regex, text: str) -> List[str]:
    """
    Splits the input text into sentences and processes short sentences.

    Parameters:
    - text (str): The input text to be split.

    Returns:
    - List[str]: A list of processed sentences.

    This function splits the text into sentences using the provided regex pattern and then processes
    the sentences to combine shorter ones based on the sentence length threshold.
    """
    debugger = EngineDebugger("split_to_sentences")
    single_sentences_list = re.split(sentence_split_regex, text)
    filtered_sentences = []
    buffer_sentence = ""
    i = 0
    while i < len(single_sentences_list):
        sentence = single_sentences_list[i]
        if buffer_sentence:
            combined_sentence = buffer_sentence + " " + sentence
        else:
            combined_sentence = sentence
        if len(combined_sentence.split()) < self.sentence_len and i != len(single_sentences_list) - 1:
            buffer_sentence = combined_sentence
        else:
            filtered_sentences.append(combined_sentence)
            buffer_sentence = ""
        i += 1
    if buffer_sentence:
        filtered_sentences.append(buffer_sentence)
    debugger.log_list("split_sentences", f"Filtered and combined sentences: {filtered_sentences}")
    return filtered_sentences
