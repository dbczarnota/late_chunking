import os
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from rich import print
import re

# Define the Chunk class using Pydantic
class Chunk(BaseModel):
    type: str = Field(..., description="The type of the chunk, e.g., NarrativeText.")
    element_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique identifier for the chunk.")
    text: str = Field(..., description="The text content of the chunk.")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the chunk.")

# Define the TextToPdfChunks class
class TextToPdfChunks:
    def __init__(
        self,
        save_format: str = "txt",  # Class-level switch for save format, defaults to 'txt'
        clean_text: bool = True  # Option to enable or disable text cleaning, defaults to True
    ):
        self.save_format = save_format
        self.clean_text = clean_text

    def _generate_metadata(self, file_directory: str, filename: str, last_modified: str, element_number: int) -> Dict[str, Any]:
        return {
            'detection_class_prob': 1.0,
            'coordinates': {'points': ()}, 
            'last_modified': last_modified,
            'filetype': 'text/plain',
            'languages': ['unknown'],
            'page_number': 1,               
            'parent_id': None,              
            'file_directory': file_directory,
            'filename': filename,
            'element_number': element_number
        }

    def _clean_text(self, text: str) -> str:
        # Remove all strange characters except '\n' and '\n\n'
        text = re.sub(r"[^\x20-\x7E\n]", "", text)  # Remove non-ASCII characters
        # Flatten more than two '\n' into '\n\n'
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _identify_list_item(self, chunks: List[str]) -> List[str]:
        bullet_points = ["-", "*", "+"]
        roman_regex = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)$", re.IGNORECASE)
        initial_classification = []

        # First pass: Identify potential list items
        for i, chunk in enumerate(chunks):
            stripped = chunk.strip()
            if (
                any(stripped.startswith(bp) for bp in bullet_points)
                or re.match(r"^\d+[\.\)]", stripped)
                or re.match(r"^[a-zA-Z][\.\)]", stripped)
                or re.match(r"^[\(\[{]?\d+[\)\]}]?", stripped)
                or roman_regex.match(stripped)
            ):
                initial_classification.append("ListItem")
                print(f"Potential ListItem identified: '{chunk}'")
            else:
                initial_classification.append("NarrativeText")
                print(f"Not ListItem (initial): '{chunk}'")

        # Helper to extract numeric or hierarchical components
        def parse_numeric_prefix(text):
            match = re.match(r"^(\d+)([\.a-z]*)", text.strip())
            if match:
                return match.groups()
            return None, None

        # Helper to check if one number follows another hierarchically
        def is_valid_hierarchy(current, next_item):
            current_num, current_suffix = current
            next_num, next_suffix = next_item
            if not current_num or not next_num:
                return False

            if int(next_num) == int(current_num) + 1 and not next_suffix:
                return True
            if current_suffix and next_suffix:
                if current_suffix.isdigit() and next_suffix.isdigit():
                    return int(next_suffix) == int(current_suffix) + 1
                if current_suffix.isalpha() and next_suffix.isalpha():
                    return ord(next_suffix) == ord(current_suffix) + 1
            return False

        # Second pass: Refine classification based on sequence
        refined_classification = []
        for i, classification in enumerate(initial_classification):
            if classification == "ListItem":
                is_preceded = i > 0 and initial_classification[i - 1] == "ListItem"
                is_followed = i < len(chunks) - 1 and initial_classification[i + 1] == "ListItem"

                current_num, current_suffix = parse_numeric_prefix(chunks[i])
                if current_num:
                    if is_followed:
                        next_num, next_suffix = parse_numeric_prefix(chunks[i + 1])
                        is_followed = is_valid_hierarchy((current_num, current_suffix), (next_num, next_suffix))
                    if is_preceded:
                        prev_num, prev_suffix = parse_numeric_prefix(chunks[i - 1])
                        is_preceded = is_valid_hierarchy((prev_num, prev_suffix), (current_num, current_suffix))

                if is_preceded or is_followed:
                    refined_classification.append("ListItem")
                    print(f"Confirmed as ListItem: '{chunks[i]}' - Preceded: {is_preceded}, Followed: {is_followed}")
                else:
                    refined_classification.append("NarrativeText")
                    print(f"Not ListItem (isolated in refinement): '{chunks[i]}'")
            else:
                refined_classification.append("NarrativeText")

        return refined_classification

    def _identify_headings(self, chunks: List[str], types: List[str]) -> List[str]:
        updated_types = types[:]
        for i, (chunk, chunk_type) in enumerate(zip(chunks, types)):
            if chunk_type != "NarrativeText":
                continue

            score = 0
            stripped = chunk.strip()

            # Length heuristic
            if 5 <= len(stripped) <= 50:
                score += 3

            # Capitalization heuristic
            if stripped.istitle() or stripped.isupper():
                score += 2

            # Surrounding blank line heuristic
            if (i == 0 or types[i - 1] != "NarrativeText") and (i == len(chunks) - 1 or types[i + 1] != "NarrativeText"):
                score += 1

            # Terminal punctuation heuristic
            if not stripped.endswith(('.', '!', '?')):
                score += 1

            # Followed by longer text heuristic
            if i < len(chunks) - 1 and len(chunks[i + 1].strip()) > len(stripped):
                score += 3

            if score >= 6:
                updated_types[i] = "Title"

        return updated_types

    def _text_file_to_chunks(self, text: str, file_directory: str, filename: str, last_modified: str) -> List[Chunk]:
        if self.clean_text:
            text = self._clean_text(text)  # Clean the text if the option is enabled
        raw_chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        types = self._identify_list_item(raw_chunks)
        types = self._identify_headings(raw_chunks, types)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            metadata = self._generate_metadata(file_directory, filename, last_modified, i)
            chunk = Chunk(
                type=types[i],
                text=chunk_text,
                metadata=metadata
            )
            chunks.append(chunk)
        return chunks

    def process_text(self, file_path: Optional[str] = None, text: Optional[str] = None, file_directory: Optional[str] = None, filename: Optional[str] = None, last_modified: Optional[str] = None, save_to_file: Optional[str] = None) -> List[Chunk]:
        if file_path:
            text, file_directory, filename, last_modified = self._read_file(file_path).values()

        if not all([text, file_directory, filename, last_modified]):
            raise ValueError("Either provide 'file_path' or all of 'text', 'file_directory', 'filename', and 'last_modified'.")

        chunks = self._text_file_to_chunks(text, file_directory, filename, last_modified)

        if save_to_file:
            if self.save_format == "txt":
                self._save_chunks_to_txt(chunks, save_to_file)
            elif self.save_format == "json":
                self._save_chunks_to_json(chunks, save_to_file)
            else:
                raise ValueError(f"Unsupported save format: {self.save_format}")

        return chunks

    def _read_file(self, file_path: str) -> Dict[str, str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        file_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        last_modified = self._get_last_modified_time(file_path)
        return {
            'text': text,
            'file_directory': file_directory,
            'filename': filename,
            'last_modified': last_modified
        }

    def _get_last_modified_time(self, file_path: str) -> str:
        last_modified_timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(last_modified_timestamp).isoformat()

    def _save_chunks_to_txt(self, chunks: List[Chunk], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"{chunk.model_dump()}\n")

    def _save_chunks_to_json(self, chunks: List[Chunk], file_path: str):
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4)

def main():
    file_path = "./data/concatenated_text.txt"
    save_path = "./data/chunks.txt"
    text_to_pdf_chunks = TextToPdfChunks(save_format="txt", clean_text=True)

    chunks = text_to_pdf_chunks.process_text(file_path=file_path, save_to_file=save_path)

    # for chunk in chunks:
    #     print(chunk.model_dump_json(indent=4))

if __name__ == "__main__":
    main()
