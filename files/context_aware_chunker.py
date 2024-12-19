from ragsyslib.files.engine_debugger import EngineDebugger
import re


class ContextAwareChunker:
    def __init__(
        self,
        debugger = EngineDebugger("CONTEXT_AWARE_CHUNKER")
        ):
        """
        """
        self.debugger = debugger
        self.debugger.debug("init", "Initializing ContextAwareChunker")
    
    
    
    
    
    
    def clean_text(self, text): #TODO sprawdzić czy w ogóle potrzebujemy pracując z pdf
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
