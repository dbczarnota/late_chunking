
import re
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
    - List[str]: A list of processed sentences.
    """
    single_sentences_list = re.split(sentence_split_regex, text)
    processed_sentences = []

    for sentence in single_sentences_list:
        while count_tokens(tokenizer, sentence) > token_limit:
            tokenized_sentence = tokenizer(sentence, return_tensors="pt")
            split_point = token_limit

            # Find the approximate split point based on token_limit
            sub_sentence = tokenizer.decode(tokenized_sentence.input_ids[0][:split_point], skip_special_tokens=True)
            processed_sentences.append(sub_sentence.strip())

            # Update the sentence with the remaining tokens
            sentence = tokenizer.decode(tokenized_sentence.input_ids[0][split_point:], skip_special_tokens=True)

        processed_sentences.append(sentence.strip())

    return processed_sentences


