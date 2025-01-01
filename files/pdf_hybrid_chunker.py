from pathlib import Path
import sys
# Add the parent directory to the Python path
script_dir = Path(__file__).resolve().parent  # Get the directory of the current script
parent_dir = script_dir.parent  # Get the parent directory to the Python path
sys.path.append(str(parent_dir))  # Add the parent directory to the Python path

from ragsyslib.files.engine_debugger import EngineDebugger

import re
from difflib import SequenceMatcher
import ast
from rich import print
import tiktoken

class PdfHybridChunker:
    """
    A class for processing text data from files, specifically tailored for use in PDF hybrid chunking tasks.
    This class provides methods to parse text files into dictionaries and add token size metadata to them.
    """
    def __init__(
        self,
        encoder=tiktoken.get_encoding("cl100k_base"),
        debugger=EngineDebugger("CONTEXT_AWARE_CHUNKER"),
        tokens_threshold_for_points=100,
        min_tokens = 30,
        max_tokens = 800,
        expected_tokens = 400,
    ):
        """
        Initializes the PdfHybridChunker instance.

        Args:
            encoder (tiktoken.Encoding): An encoder instance from tiktoken for tokenizing text.
            debugger (EngineDebugger): A debugger instance for logging and debugging purposes.
            tokens_threshold_for_points (int): Threshold for token size in scoring conditions.
        """
        self.encoder = encoder  # Encoder instance for tokenizing text.
        self.debugger = debugger  # Debugger instance for logging operations.
        self.tokens_threshold_for_points = tokens_threshold_for_points  # Token size threshold for scoring conditions.
        self.min_tokens = min_tokens  # Minimum number of tokens for a chunk to be considered valid.
        self.max_tokens = max_tokens  # Maximum number of tokens for a chunk to be considered valid.
        self.expected_tokens = expected_tokens  # Expected number of tokens for a chunk to be considered valid.

        if self.debugger:
            self.debugger.debug("init", "Initializing ContextAwareChunker.")
    
    def print_top_chunks(self, dicts_list, top_n=5):
        """
        Prints the top N chunks with the highest scores.

        Args:
            dicts_list (list): List of dictionaries with calculated scores.
            top_n (int): Number of top chunks to print.
        """
        # Sort the list by the 'sum' score in descending order
        sorted_chunks = sorted(dicts_list, key=lambda x: x.get("score", {}).get("sum", 0), reverse=True)

        print(f"Top {top_n} chunks with the highest scores:")
        for i, chunk in enumerate(sorted_chunks[:top_n]):
            print(f"Rank {i+1}:")
            print(f"Element Number: {chunk.get('metadata', {}).get('element_number', 'N/A')}")
            print(f"Text: {chunk.get('text', 'No text available')}")
            print(f"Score: {chunk.get('score')}")
            print("-" * 50)
    
    def parse_txt_to_dicts(self, file_path):
        """
        Reads a text file containing dictionaries (one dictionary per line) and converts it into a list of dictionaries.

        Args:
            file_path (str): Path to the text file.

        Returns:
            list: A list of dictionaries parsed from the file.
        
        Example input file:
            {"text": "Sample text 1", "metadata": {"author": "John"}}
            {"text": "Sample text 2", "metadata": {"author": "Jane"}}
        
        Errors during parsing will be logged to the console using the `rich` library.
        """
        dicts_list = []  # List to store the parsed dictionaries.

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace.
                if line:  # Skip empty lines.
                    try:
                        # Safely evaluate the dictionary string using `ast.literal_eval`.
                        dict_obj = ast.literal_eval(line)
                        dicts_list.append(dict_obj)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing line: {line}\n{e}")

        return dicts_list
    
    def add_token_size_to_dicts(self, dicts_list):
        """
        Adds token size metadata for each dictionary in the provided list.
        
        This method calculates the number of tokens in the "text" field of each dictionary using the provided encoder
        and adds a `token_size` field under the dictionary's `metadata` key.

        Args:
            dicts_list (list): List of dictionaries containing at least a "text" key.

        Returns:
            list: The updated list of dictionaries with token size metadata added.

        Example:
            Input dictionary:
            {"text": "Sample text", "metadata": {"author": "John"}}

            Output dictionary:
            {"text": "Sample text", "metadata": {"author": "John", "token_size": 3}}
        """
        for d in dicts_list:
            if "text" in d:  # Ensure the dictionary has a "text" field.
                # Encode the "text" field to calculate the number of tokens.
                tokens = self.encoder.encode(d["text"])
                # Add or update the "token_size" field in the "metadata" dictionary.
                d.setdefault("metadata", {})["token_size"] = len(tokens)

        return dicts_list

    def tokenize(self, text):
        """
        Splits text into a list of words. 
        For simplicity, we remove punctuation except for basic diacritics 
        and split on whitespace.
        """
        # Lowercase (optionally)
        text = text.lower()
        # Remove typical punctuation with a regex
        text = re.sub(r'[^\wÀ-ž\s-]', '', text, flags=re.UNICODE)
        # Split on whitespace
        return text.split()

    def word_similarity(self, word_a, word_b):
        """
        Returns a similarity score between two words (0.0 to 1.0).
        Uses difflib.SequenceMatcher for demonstration.
        """
        return SequenceMatcher(None, word_a, word_b).ratio()

    def find_chunk_in_elements(self, dicts_list, chunk_text, similarity_threshold=0.8):
        """
        Attempts to find the chunk_text within the list of element dicts.
        
        :param elements: list of dicts, each containing 'text' and 'element_number'
        :param chunk_text: the chunk of text we want to locate
        :param similarity_threshold: how lenient we are with fuzzy matching (word-by-word)
        :return: a dict with 
        {
            'start_element_number': int or None,
            'end_element_number': int or None,
            'match_factor': float (0.0 to 1.0)
        }
        or None if no sufficiently good match is found
        """
        
        # 1. Flatten the elements into a list of (word, element_number).
        #    We'll keep index references so that we can later map them back.
        flattened_words = []   # list of (word, element_number)
        
        for elem in dicts_list:
            text_content = elem.get('text', '')
            elem_number = elem.get('metadata', {}).get('element_number', None)
            # If 'element_number' is not in metadata, fallback to top-level:
            if elem_number is None:
                elem_number = elem.get('element_number')
            
            tokens = self.tokenize(text_content)
            for w in tokens:
                flattened_words.append((w, elem_number))
        
        # 2. Tokenize the chunk
        chunk_tokens = self.tokenize(chunk_text)
        if not chunk_tokens:
            return None
        
        chunk_length = len(chunk_tokens)
        
        # Edge case: if chunk_length is longer than total words, we can’t match
        if chunk_length > len(flattened_words):
            return None
        
        # 3. Sliding window to find best match
        best_score = 0.0
        best_start_index = None
        
        # We'll store the best match factor
        # "match_factor" can be defined as the average similarity across the matched words.
        for start_idx in range(len(flattened_words) - chunk_length + 1):
            # For each position, compare chunk_tokens with the flattened slice
            slice_segment = flattened_words[start_idx : start_idx + chunk_length]
            
            # Compute word-by-word similarity
            total_similarity = 0.0
            for i, chunk_word in enumerate(chunk_tokens):
                doc_word, _ = slice_segment[i]
                sim = self.word_similarity(chunk_word, doc_word)
                total_similarity += sim
            
            average_similarity = total_similarity / chunk_length
            
            # If this window is the best so far, record it
            if average_similarity > best_score:
                best_score = average_similarity
                best_start_index = start_idx
        
        # 4. Check if our best score is above some threshold
        #    If not, we consider it "not found"
        if best_score < similarity_threshold:
            return None
        
        # 5. Determine the start and end element_number from the best match
        best_slice = flattened_words[best_start_index : best_start_index + chunk_length]
        start_element_number = best_slice[0][1]
        end_element_number = best_slice[-1][1]
        
        # For reporting, we can simply return these. 
        # Optionally, you might want to confirm that the entire chunk is physically 
        # consecutive or check if those element_numbers are truly contiguous. 
        # But for simplicity, let's just return them directly.
        return {
            'start_element_number': start_element_number,
            'end_element_number': end_element_number,
            'match_factor': best_score
        }

    def calculate_chunk_split_scores(self, dicts_list, save_to_file=None, phrases_list=None):
        """
        Calculates a potential chunk split point score for each element in the list of dictionaries.

        A score is added to each element in the form of a dictionary under the key `score`.
        The score includes:
        - `sum`: The total score for the element.
        - Individual conditions (e.g., `title_condition`, `token_condition`, `phrases_condition`).

        Args:
            dicts_list (list): List of dictionaries, each representing an element.
            save_to_file (str, optional): Path to a file to save the dictionaries with scores.
            phrases_list (list, optional): List of phrases to apply in `apply_phrases_condition`. Defaults to None.

        Returns:
            list: Updated list of dictionaries with scores added.
        """
        print("Starting to calculate chunk split scores...")

        for d in dicts_list:
            # Initialize the score structure for each element.
            d["score"] = {
                "sum": 0
            }

        # Apply individual conditions.
        dicts_list = self.apply_title_condition(dicts_list)
        dicts_list = self.apply_token_condition(dicts_list)
        dicts_list = self.apply_first_listitem_condition(dicts_list)

        # Only apply phrases_condition if phrases_list is provided.
        if phrases_list:
            dicts_list = self.apply_phrases_condition(dicts_list, phrases_list)

        dicts_list = self.apply_consecutive_title_condition(dicts_list)

        # Calculate the total score for each element.
        for d in dicts_list:
            d["score"]["sum"] = sum(v for k, v in d["score"].items() if k != "sum")

        # Ensure the first element has the highest score.
        if dicts_list:
            max_score = max(d["score"]["sum"] for d in dicts_list[1:])  # Find the maximum score excluding the first element.
            print(f"Calculated max score: {max_score}")
            dicts_list[0]["score"]["sum"] = max_score + 1  # Set the first element's score to be higher.

        print("Finished calculating scores.")

        # Save to file if specified.
        if save_to_file:
            self.save_dicts_to_txt(dicts_list, save_to_file)

        return dicts_list

    def apply_title_condition(self, dicts_list):
        """
        Adds a score for elements based on their type.

        Args:
            dicts_list (list): List of dictionaries representing elements.

        Returns:
            list: Updated list of dictionaries with `title_condition` scores added.
        """
        print("Applying title condition...")
        for d in dicts_list:
            element_type = d.get("type", "")
            if element_type == "Title":
                d["score"]["title_condition"] = 5
            elif element_type == "NarrativeText":
                d["score"]["title_condition"] = 2
            else:
                d["score"]["title_condition"] = 1
        print("Finished applying title condition.")
        return dicts_list

    def apply_token_condition(self, dicts_list):
        """
        Adds a score based on token size conditions for sections following a "Title".

        Args:
            dicts_list (list): List of dictionaries representing elements.

        Returns:
            list: Updated list of dictionaries with `token_condition` scores added.
        """
        print("Applying token condition...")
        for i, d in enumerate(dicts_list):
            if d.get("type") == "Title":
                # Sum token sizes until the next title.
                token_sum = 0
                for j in range(i + 1, len(dicts_list)):
                    if dicts_list[j].get("type") == "Title":
                        break
                    token_sum += dicts_list[j].get("metadata", {}).get("token_size", 0)

                if token_sum > self.tokens_threshold_for_points:
                    excess_tokens = token_sum - self.tokens_threshold_for_points
                    base_points = 5  # Fixed points for exceeding the threshold
                    additional_points = 0.01 * excess_tokens  # Additional points for excess tokens
                    d["score"]["token_condition"] = base_points + additional_points
                else:
                    d["score"]["token_condition"] = 0
            else:
                # Ensure a score field is present for non-"Title" elements
                if "score" not in d:
                    d["score"] = {}
                d["score"]["token_condition"] = 0

        print("Finished applying token condition.")
        return dicts_list
    
    def apply_first_listitem_condition(self, dicts_list):
        """
        Assigns 5 points to the first list item ("ListItem") in each detected list and 0 points to all other elements.

        Args:
            dicts_list (list): List of dictionaries representing elements.

        Returns:
            list: Updated list of dictionaries with `first_listitem_condition` scores added.
        """
        print("Applying first listitem condition...")
        in_list = False
        for i, d in enumerate(dicts_list):
            if d.get("type") == "ListItem":
                if not in_list:  # First list item in the current list
                    d["score"]["first_listitem_condition"] = 5
                    print(f"Assigned 5 points to first ListItem at index {i}.")
                    in_list = True
                else:  # Subsequent list items in the same list
                    d["score"]["first_listitem_condition"] = 0
            else:
                d["score"]["first_listitem_condition"] = 0  # Ensure 0 points for non-list items
                in_list = False  # Reset when not in a list

        print("Finished applying first listitem condition.")
        return dicts_list

    def apply_phrases_condition(self, dicts_list, phrases_list, max_below=3, max_above=2):
        """
        Applies a score based on the presence of specific phrases in the elements.

        Args:
            dicts_list (list): List of dictionaries representing elements.
            phrases_list (list): List of phrases to search for.
            max_below (int): Maximum number of elements below to check for a Title.
            max_above (int): Maximum number of elements above to check for a Title.

        Returns:
            list: Updated list of dictionaries with `phrases_condition` scores added.
        """
        print("Applying phrases condition...")

        # Initialize all elements with 0 for "phrases_condition"
        for idx, d in enumerate(dicts_list):
            if "score" not in d:
                d["score"] = {}
            d["score"]["phrases_condition"] = 0
            # print(f"Initialized element at index {idx} with phrases_condition=0")

        for phrase in phrases_list:
            print(f"Searching for phrase: {phrase}")
            # Use find_chunk_in_elements to locate the phrase in the elements
            match = self.find_chunk_in_elements(dicts_list, phrase)
            if not match:
                print(f"Phrase '{phrase}' not found in elements.")
                continue

            # Extract matching details
            start_element_number = match['start_element_number']
            print(f"Phrase '{phrase}' matched at element {start_element_number}.")

            # Find the element index of the match
            match_index = next((i for i, d in enumerate(dicts_list)
                                if d.get('metadata', {}).get('element_number') == int(start_element_number)), None)

            if match_index is None:
                print(f"No matching index found for start_element_number {start_element_number}.")
                continue

            element = dicts_list[match_index]
            print(f"Primary match for phrase '{phrase}' found at index {match_index}.")

            # Determine scoring based on the type and proximity to a Title
            if element.get("type") == "Title":
                element["score"]["phrases_condition"] = 30
                print(f"Element at index {match_index} is a Title. Assigned phrases_condition=30.")
            else:
                # Check below for a Title
                title_found = False
                for i in range(1, max_below + 1):
                    index_below = match_index + i
                    if index_below < len(dicts_list) and dicts_list[index_below].get("type") == "Title":
                        dicts_list[index_below]["score"]["phrases_condition"] = 10
                        print(f"Title found below at index {index_below}. Assigned phrases_condition=10.")
                        title_found = True
                        break

                # If no Title found below, check above
                if not title_found:
                    for i in range(1, max_above + 1):
                        index_above = match_index - i
                        if index_above >= 0 and dicts_list[index_above].get("type") == "Title":
                            dicts_list[index_above]["score"]["phrases_condition"] = 10
                            print(f"Title found above at index {index_above}. Assigned phrases_condition=10.")
                            title_found = True
                            break

                # If no Title found in proximity, apply 5 points to the original element
                if not title_found:
                    element["score"]["phrases_condition"] = 5
                    print(f"No Title found near index {match_index}. Assigned phrases_condition=5 to the original element.")

        print("Finished applying phrases condition.")
        return dicts_list

    def apply_consecutive_title_condition(self, dicts_list):
        """
        Ensures that the first "Title" element in a set of consecutive titles on the same page
        inherits the maximum score of the entire set (including the full `score` dict),
        while the following titles have a score of 0.

        Args:
            dicts_list (list): List of dictionaries representing elements.

        Returns:
            list: Updated list of dictionaries with consecutive "Title" scores adjusted.
        """
        print("Applying consecutive title condition...")
        i = 0
        while i < len(dicts_list):
            current = dicts_list[i]

            if current.get("type") == "Title":
                current_page = current.get("metadata", {}).get("page_number")
                max_score = current["score"].copy()
                j = i + 1

                # Identify all consecutive titles on the same page
                while j < len(dicts_list) and dicts_list[j].get("type") == "Title" and dicts_list[j].get("metadata", {}).get("page_number") == current_page:
                    for key, value in dicts_list[j]["score"].items():
                        max_score[key] = max(max_score.get(key, 0), value)
                    j += 1

                # Assign the max score to the first title and set others to 1
                current["score"] = max_score
                # print(f"First Title on page {current_page} inherits max score: {max_score}")
                for k in range(i + 1, j):
                    dicts_list[k]["score"] = {key: 1 for key in max_score.keys()}
                    # print(f"Title element at index {k} on page {current_page} set to 0")

                # Move the index to the end of the set
                i = j
            else:
                i += 1
        print("Finished applying consecutive title condition.")
        return dicts_list

    def save_dicts_to_txt(self, dicts_list, output_file):
        """
        Saves the list of dictionaries to a text file, one dictionary per line.

        Args:
            dicts_list (list): List of dictionaries to save.
            output_file (str): Path to the output text file.
        """
        print(f"Saving dictionaries to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as file:
            for d in dicts_list:
                file.write(str(d) + '\n')
        print(f"Finished saving dictionaries to {output_file}.")

    def find_initial_split_points(self, dicts_list, multiplier=2):
        """
        Finds split points by selecting elements with the highest scores based on the expected number of chunks.

        Args:
            dicts_list (list): List of dictionaries with scores and metadata.
            multiplier (float): Multiplier to adjust the number of split points.

        Returns:
            list: A list of split points.
        """
        print("Starting initial split points calculation...")

        # Calculate total tokens and expected number of chunks
        total_tokens = sum(d.get("metadata", {}).get("token_size", 0) for d in dicts_list)
        expected_number_of_chunks = max(1, total_tokens // self.expected_tokens)  # Ensure at least one chunk

        print(f"Total tokens: {total_tokens}")
        print(f"Expected number of chunks before multiplier: {expected_number_of_chunks}")

        # Apply the multiplier to determine the number of split points
        adjusted_chunks = max(1, int(expected_number_of_chunks * multiplier))  # Ensure at least one chunk

        print(f"Adjusted number of chunks (after multiplier): {adjusted_chunks}")

        # Sort elements by score in descending order and pick top `adjusted_chunks` elements
        sorted_indices = sorted(
            range(len(dicts_list)),
            key=lambda i: dicts_list[i].get("score", {}).get("sum", 0),
            reverse=True
        )
        top_split_indices = sorted(sorted_indices[:adjusted_chunks])

        print(f"Top split indices selected: {top_split_indices}")

        return top_split_indices

    def calculate_chunk_score(self, split_point, chunk):
        """
        Calculates the score for a given chunk, adjusting for size deviation from `expected_tokens`.

        Args:
            split_point (dict): The element at the split point.
            chunk (list): The list of elements in the chunk.

        Returns:
            float: The calculated chunk score.
        """
        token_count = sum(e["metadata"].get("token_size", 0) for e in chunk)
        if token_count < self.min_tokens or token_count > self.max_tokens:
            return 0  # Score is 0 if the chunk is outside token limits

        # Calculate maximum deviation thresholds
        max_deviation_below = self.expected_tokens - self.min_tokens
        max_deviation_above = self.max_tokens - self.expected_tokens

        # Calculate scaled factor linearly
        if token_count < self.expected_tokens:
            scaled_factor = 0.3 + 0.7 * (1 - (self.expected_tokens - token_count) / max_deviation_below)
        else:
            scaled_factor = 0.3 + 0.7 * (1 - (token_count - self.expected_tokens) / max_deviation_above)

        return split_point["score"].get("sum", 0) * scaled_factor

    def print_split_details(self, dicts_list, split_points):
        """
        Prints details of all chunks defined by the split points.

        Args:
            dicts_list (list): List of dictionaries with scores and metadata.
            split_points (list): List of indices defining the split points.
        """
        print("\nSplit Points and Chunk Details:")

        for i, idx in enumerate(split_points):
            # Define the start and end of the chunk
            start = idx
            end = split_points[i + 1] if i < len(split_points) - 1 else len(dicts_list)

            # Get the chunk and calculate its score
            chunk = dicts_list[start:end]
            if not chunk:
                print(f"Warning: Chunk starting at index {start} is empty!")
                continue

            element = dicts_list[start]  # The element that starts the chunk
            chunk_score = self.calculate_chunk_score(element, chunk)

            # Debugging: Validate chunk elements
            for c in chunk:
                if "metadata" not in c:
                    print(f"Warning: Element in chunk is missing 'metadata': {c}")

            # Print chunk details
            element_number = element["metadata"].get("element_number", "N/A")
            score = element["score"].get("sum", 0)
            element_type = element.get("type", "N/A")
            token_count = sum(e["metadata"].get("token_size", 0) for e in chunk)
            text_snippet = element["text"][:100] if "text" in element else "No text available"

            # Determine the element number range for the chunk
            start_element_number = chunk[0]["metadata"].get("element_number", "N/A")
            end_element_number = chunk[-1]["metadata"].get("element_number", "N/A")

            print(f"Split Details:")
            print(f"  Element Number: {element_number}")
            print(f"  Score: {score}")
            print(f"  Chunk Score: {chunk_score}")
            print(f"  Type: {element_type}")
            print(f"  Token Count of Chunk: {token_count}")
            print(f"  Element Number Range: {start_element_number}-{end_element_number}")
            print(f"  Text Snippet: '{text_snippet}'")
            print("-" * 50)

    def no_zero_refinement(self, dicts_list, initial_split_points):
        """
        Refines split points to ensure no chunk has a score of 0.

        Args:
            dicts_list (list): List of dictionaries with scores and metadata.
            initial_split_points (list): List of initial split points.

        Returns:
            list: A list of refined split points.
        """
        refined_split_points = []

        # Iterate over the initial split points to create initial chunks
        for i, idx in enumerate(initial_split_points):
            start = idx
            end = initial_split_points[i + 1] if i < len(initial_split_points) - 1 else len(dicts_list)
            refined_split_points.append(idx)

            chunk = dicts_list[start:end]

            # Calculate the chunk score
            chunk_score = self.calculate_chunk_score(dicts_list[start], chunk)

            # If the chunk score is 0, attempt to refine it
            if chunk_score == 0:
                token_count = sum(e["metadata"].get("token_size", 0) for e in chunk)

                # If the chunk's token count is below min_tokens, merge with next or previous based on higher score
                if token_count < self.min_tokens:
                    if len(refined_split_points) > 1:
                        prev_split_idx = refined_split_points[-2]
                        # print(f'prev_split_idx: {prev_split_idx}')
                        prev_chunk = dicts_list[prev_split_idx:start]
                        next_chunk_start = end
                        # print(f'next_chunk_start: {next_chunk_start}')
                        next_chunk_end = initial_split_points[i + 2] if i + 2 < len(initial_split_points) else len(dicts_list)
                        # print(f'next_chunk_end: {next_chunk_end}')
                        next_chunk = dicts_list[next_chunk_start:next_chunk_end]
                        

                        prev_score = self.calculate_chunk_score(dicts_list[prev_split_idx], prev_chunk + chunk)
                        # print(f'prev_score: {prev_score}')
                        if next_chunk:
                            combined_next_chunk = chunk + next_chunk
                            next_score = self.calculate_chunk_score(dicts_list[start], combined_next_chunk)
                            # print(f'next_score: {next_score}')
                        else:
                            next_score = 0

                        if prev_score >= next_score:
                            refined_split_points.pop()
                            print(f"Merging chunk starting at {start} with previous chunk at {prev_split_idx} due to higher score ({prev_score:.2f} >= {next_score:.2f}).")
                        else:
                            print(f"Merging chunk starting at {start} with next chunk due to higher score ({next_score:.2f} > {prev_score:.2f}).")
                            # Adjust the start and token_count for continued processing
                            chunk = chunk + next_chunk
                            token_count = sum(e["metadata"].get("token_size", 0) for e in chunk)

                # If the chunk's token count exceeds max_tokens, split it iteratively
                elif token_count > self.max_tokens:
                    print(f"Splitting chunk starting at {start} due to high token count ({token_count}).")

                    while token_count > self.max_tokens:
                        # Sort elements within the chunk by their scores to find optimal split points
                        sorted_indices = sorted(
                            range(len(chunk)),
                            key=lambda i: chunk[i].get("score", {}).get("sum", 0),
                            reverse=True
                        )

                        split_found = False
                        for split_idx in sorted_indices:
                            split_point = start + split_idx

                            # Ensure the new split point maintains valid token counts for resulting chunks
                            left_chunk = dicts_list[start:split_point]
                            right_chunk = dicts_list[split_point:end]

                            left_tokens = sum(e["metadata"].get("token_size", 0) for e in left_chunk)
                            right_tokens = sum(e["metadata"].get("token_size", 0) for e in right_chunk)

                            if (self.expected_tokens - self.min_tokens) <= left_tokens <= (self.expected_tokens + self.max_tokens) and right_tokens > 0:
                                refined_split_points.append(split_point)
                                print(f"Added split point at {split_point}, aiming for expected token size.")
                                chunk = right_chunk
                                start = split_point
                                token_count = right_tokens
                                split_found = True
                                break

                        if not split_found:
                            print(f"Warning: Unable to split chunk starting at {start}. Leaving as is.")
                            break

            # Ensure split points are sorted and unique
        refined_split_points = sorted(set(refined_split_points))

        return refined_split_points

    def refine_split_points(self, dicts_list, initial_split_points):
        """
        Refines split points using a greedy sliding window approach with a window size of 3,
        but avoids infinite loops by not modifying split_points in-place during iteration.
        """
        print("\nInitial Average Chunk Score:")
        initial_avg_score = self.calculate_average_chunk_score(dicts_list, initial_split_points)
        print(f"  {initial_avg_score:.2f}")

        refined_split_points = initial_split_points.copy()

        # Safety net to avoid infinite loops
        max_iterations = 50
        iteration_count = 0

        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                print("Reached max_iterations - stopping refinement to avoid infinite loop.")
                break

            improved = False
            # Work on a copy so we don't change refined_split_points while iterating
            new_split_points_list = refined_split_points.copy()

            # Iterate over the split points with a sliding window of size 3
            i = 0
            while i < len(new_split_points_list) - 1:
                window_split_points = new_split_points_list[i:i + 3]

                # Skip if fewer than 2 points in the window
                if len(window_split_points) < 2:
                    i += 1
                    continue

                window_start = window_split_points[0]
                window_end = (
                    new_split_points_list[i + 2]
                    if (i + 2) < len(new_split_points_list)
                    else len(dicts_list)
                )

                # Refine the split points within this window
                refined_window = self.refine_window(
                    dicts_list, window_split_points, window_start, window_end
                )

                # If there's a change, replace that window slice in the copy
                if refined_window != window_split_points:
                    new_split_points_list[i:i + len(window_split_points)] = refined_window
                    improved = True

                # Move to the next index
                i += 1

            # Ensure split points are sorted and unique
            new_split_points_list = sorted(set(new_split_points_list))

            # If no improvement or no change after this pass, break
            if not improved or new_split_points_list == refined_split_points:
                break

            # Otherwise, update the refined list and continue
            refined_split_points = new_split_points_list

        print("\nFinal Average Chunk Score:")
        final_avg_score = self.calculate_average_chunk_score(dicts_list, refined_split_points)
        print(f"  {final_avg_score:.2f}")

        return refined_split_points

    def refine_window(self, dicts_list, split_points, window_start, window_end):
        """
        Refines split points within a sliding window of chunks.

        Args:
            dicts_list (list): List of dictionaries with scores and metadata.
            split_points (list): List of split points in the current window.
            window_start (int): The starting index of the window.
            window_end (int): The ending index of the window.

        Returns:
            list: Refined split points for the window.
        """
        # Ensure there are enough split points for refinement
        if len(split_points) < 2:
            return split_points

        best_split_points = split_points.copy()
        best_avg_score = self.calculate_average_chunk_score(dicts_list, split_points)

        # Try shifting split points within the window
        for shift in range(-1, 2):
            temp_split_points = best_split_points.copy()
            for idx in range(1, len(temp_split_points) - 1):  # Avoid first and last points
                new_split = temp_split_points[idx] + shift
                # Ensure the new split point is within window bounds
                temp_split_points[idx] = max(window_start, min(window_end, new_split))

            temp_split_points = sorted(set(temp_split_points))
            temp_avg_score = self.calculate_average_chunk_score(dicts_list, temp_split_points)

            if temp_avg_score > best_avg_score:
                best_avg_score = temp_avg_score
                best_split_points = temp_split_points

        # Try merging chunks in the window
        for idx in range(1, len(best_split_points) - 1):  # Avoid first and last points
            if idx < len(best_split_points):  # Ensure idx is within bounds
                temp_split_points = best_split_points.copy()
                temp_split_points.pop(idx)

                temp_avg_score = self.calculate_average_chunk_score(dicts_list, temp_split_points)

                if temp_avg_score > best_avg_score:
                    best_avg_score = temp_avg_score
                    best_split_points = temp_split_points

        # Try splitting chunks in the window
        for idx in range(1, len(best_split_points) - 1):  # Avoid first and last points
            if idx + 1 < len(best_split_points):  # Ensure idx + 1 is within bounds
                temp_split_points = best_split_points.copy()
                midpoint = (temp_split_points[idx] + temp_split_points[idx + 1]) // 2

                # Avoid duplicate split points
                if midpoint not in temp_split_points:
                    temp_split_points.insert(idx + 1, midpoint)

                    temp_avg_score = self.calculate_average_chunk_score(dicts_list, temp_split_points)

                    if temp_avg_score > best_avg_score:
                        best_avg_score = temp_avg_score
                        best_split_points = temp_split_points

        return best_split_points

    def calculate_average_chunk_score(self, dicts_list, split_points):
        """
        Calculates the average chunk score given split points.

        Args:
            dicts_list (list): List of dictionaries with scores and metadata.
            split_points (list): List of split points.

        Returns:
            float: The average chunk score.
        """
        total_score = 0
        chunk_count = 0

        for i, idx in enumerate(split_points):
            start = idx
            end = split_points[i + 1] if i + 1 < len(split_points) else len(dicts_list)
            chunk = dicts_list[start:end]

            if chunk:
                chunk_score = self.calculate_chunk_score(dicts_list[start], chunk)
                if chunk_score > 0:
                    total_score += chunk_score
                    chunk_count += 1

        return total_score / chunk_count if chunk_count > 0 else 0

    def find_split_points(self, dicts_list):
        """
        Finds split points by first determining initial split points and then refining them.

        Args:
            dicts_list (list): List of dictionaries with scores and metadata.

        Returns:
            list: A list of refined split points.
        """
        # Step 1: Find initial split points
        initial_split_points = self.find_initial_split_points(dicts_list)
        print("\nInitial split points and details:")
        self.print_split_details(dicts_list, initial_split_points)
        
        # Step 2: Refine the split points
        no_zero_split_points = self.no_zero_refinement(dicts_list, initial_split_points)
        
        print("\nNo Zeros Split Points:")
        self.print_split_details(dicts_list, no_zero_split_points)        
        
        
        refined_split_points = self.refine_split_points(dicts_list, no_zero_split_points)
        # refined_split_points = no_zero_split_points # Use initial split points for now
        
        print("\nFinal refined split points and details:")
        self.print_split_details(dicts_list, refined_split_points)

 

        return refined_split_points

 

def main():

    
    
    # Example usage
    chunker = PdfHybridChunker()
    file_path = './data/chunks_output_with_metadata.txt'
    dict_list = chunker.parse_txt_to_dicts(file_path)
    # print(dict_list[:10])
    # print('#'*100)
    new_dicts = chunker.add_token_size_to_dicts(dict_list)
    # print(new_dicts[:10])
    # print('#'*100)
    file_path = './data/chunks_output_with_metadata2_with_scores.txt'
    phrases_list = ['The dominant sequence transduction models are based on', '‡Work performed while at Google Research. 31st Conference on Neural Information Processing Systems (NIPS 2017),', '5 Training']
    scored_dicts = chunker.calculate_chunk_split_scores(new_dicts, file_path, phrases_list=phrases_list)
    # print(scored_dicts[:15])
    # print('#'*100)

    chunker.print_top_chunks(scored_dicts, top_n=5)
    print(chunker.find_split_points(scored_dicts))


if __name__ == "__main__":
    main()