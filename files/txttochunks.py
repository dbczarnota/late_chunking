import os
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Define the Chunk class using Pydantic
class Chunk(BaseModel):
    type: str = Field(..., description="The type of the chunk, e.g., NarrativeText.")
    element_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique identifier for the chunk.")
    text: str = Field(..., description="The text content of the chunk.")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the chunk.")

# Define the TextToPdfChunks class
class TextToPdfChunks:
    save_format: str = "txt"  # Class-level switch for save format, defaults to 'txt'

    def __init__(self):
        pass

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

    def _text_file_to_chunks(self, text: str, file_directory: str, filename: str, last_modified: str) -> List[Chunk]:
        raw_chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            metadata = self._generate_metadata(file_directory, filename, last_modified, i)
            chunk = Chunk(
                type='NarrativeText',
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
                f.write(f"Chunk ID: {chunk.element_id}\n")
                f.write(f"Type: {chunk.type}\n")
                f.write(f"Text: {chunk.text}\n")
                f.write(f"Metadata: {chunk.metadata}\n")
                f.write("\n")

    def _save_chunks_to_json(self, chunks: List[Chunk], file_path: str):
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([chunk.dict() for chunk in chunks], f, indent=4)

def main():
    file_path = "./multimodal_RAG/data/concatenated_text.txt"
    save_path = "./multimodal_RAG/data/chunks.txt"
    text_to_pdf_chunks = TextToPdfChunks()

    # Change the class-level save format if needed
    TextToPdfChunks.save_format = "txt"  # Change to "json" if you want JSON output

    chunks = text_to_pdf_chunks.process_text(file_path=file_path, save_to_file=save_path)

    for chunk in chunks:
        print(chunk.model_dump_json(indent=4))

if __name__ == "__main__":
    main()
