from pypdf import PdfReader
from typing import List, Dict
import re

class DocumentProcessor:
    """Process PDF documents and split into chunks"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                # Add page marker for citation purposes
                text += f"\n\n--- Page {page_num + 1} ---\n\n"
                text += page_text
            
            return text
        
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        
        # Remove page markers temporarily for cleaning
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks
        
        Returns:
            List of chunks with metadata
        """
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(cleaned_text):
            # Get chunk
            end = start + self.chunk_size
            chunk = cleaned_text[start:end]
            
            # Try to break at sentence boundary
            if end < len(cleaned_text):
                # Look for sentence ending
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclamation = chunk.rfind('!')
                
                break_point = max(last_period, last_question, last_exclamation)
                
                if break_point > self.chunk_size * 0.5:  # At least 50% through chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            # Store chunk with metadata
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk.strip(),
                'start_char': start,
                'end_char': end,
                'chunk_index': chunk_id
            })
            
            chunk_id += 1
            start = end - self.chunk_overlap  # Overlap for context
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Complete pipeline: PDF → chunks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text chunks with metadata
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(raw_text)} characters")
        
        # Chunk text
        chunks = self.chunk_text(raw_text)
        print(f"Created {len(chunks)} chunks")
        
        return chunks


# Test the processor
if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Test with a sample PDF (you'll need to provide one)
    # chunks = processor.process_pdf("sample.pdf")
    # for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
    #     print(f"\nChunk {i}:")
    #     print(chunk['text'][:200] + "...")