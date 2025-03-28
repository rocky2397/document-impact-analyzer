"""
Data extraction module for handling documents from various sources.
"""
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Note: The original research used NLM-Ingestor for PDF extraction
# NLM-Ingestor (https://github.com/NIHOPA/NLM-Ingestor) provides better extraction
# for complex PDF layouts than the built-in methods below.
# To use NLM-Ingestor:
# 1. Install it with: pip install git+https://github.com/NIHOPA/NLM-Ingestor.git
# 2. Extract text with: python -m nlm_ingestor.ingestor --input ./data --output ./extracted_text

class Document:
    """Simple document class for storing extracted text and metadata"""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.page_content = content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(content=[{self.page_content[:50]}...], metadata={self.metadata})"


class DataExtractor:
    """
    Extracts text and metadata from company documents.
    Supports PDF and text files with flexible source patterns.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the DataExtractor
        
        Args:
            data_dir (str): Directory containing company documents
        """
        self.data_dir = data_dir
    
    def extract_documents(self, 
                         companies: Optional[List[str]] = None, 
                         years: Optional[List[Union[str, int]]] = None, 
                         file_pattern: str = "*.pdf") -> List[Document]:
        """
        Extract documents from the data directory
        
        Args:
            companies (list, optional): List of company names to extract
            years (list, optional): List of years to extract
            file_pattern (str): File pattern to match (default: "*.pdf")
            
        Returns:
            list: List of Document objects with extracted text and metadata
        """
        try:
            import glob
            
            # Find all matching files
            search_pattern = os.path.join(self.data_dir, "**", file_pattern)
            all_files = glob.glob(search_pattern, recursive=True)
            
            # Filter by company if specified
            if companies:
                company_pattern = r'|'.join(companies)
                all_files = [f for f in all_files if re.search(company_pattern, f, re.IGNORECASE)]
            
            # Filter by year if specified
            if years:
                year_pattern = r'|'.join([str(y) for y in years])
                all_files = [f for f in all_files if re.search(year_pattern, f)]
            
            print(f"Found {len(all_files)} documents matching criteria")
            
            # Extract text from each file
            documents = []
            for file_path in all_files:
                try:
                    # Extract metadata from path
                    file_name = os.path.basename(file_path)
                    company = self._extract_company(file_path)
                    year = self._extract_year(file_path)
                    
                    # Extract text based on file type
                    if file_path.lower().endswith('.pdf'):
                        chunks = self._extract_from_pdf(file_path)
                    elif file_path.lower().endswith('.txt'):
                        chunks = self._extract_from_text(file_path)
                    elif file_path.lower().endswith(('.csv', '.csv.gz')):
                        chunks = self._extract_from_csv(file_path)
                    else:
                        print(f"Unsupported file type: {file_path}")
                        chunks = []
                    
                    # Create Document objects with metadata
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            content=chunk,
                            metadata={
                                "source": company,
                                "year": year,
                                "file_name": file_name,
                                "file_path": file_path,
                                "chunk_id": i
                            }
                        )
                        documents.append(doc)
                    
                    print(f"Extracted {len(chunks)} chunks from {file_path}")
                except Exception as e:
                    print(f"Error extracting from {file_path}: {e}")
            
            return documents
        except Exception as e:
            print(f"Error during document extraction: {e}")
            return []
    
    def _extract_company(self, file_path: str) -> str:
        """Extract company name from file path"""
        # Try to extract from directory structure first
        dir_name = os.path.basename(os.path.dirname(file_path))
        if not dir_name.startswith('.'):  # Ignore hidden directories
            return dir_name
        
        # Fall back to filename
        file_name = os.path.basename(file_path)
        company_match = re.search(r'^([A-Za-z]+)_', file_name)
        if company_match:
            return company_match.group(1)
        
        return "UNKNOWN"
    
    def _extract_year(self, file_path: str) -> str:
        """Extract year from file path"""
        # Try to extract year from filename
        file_name = os.path.basename(file_path)
        year_match = re.search(r'(20\d{2})', file_name)
        if year_match:
            return year_match.group(1)
        
        # Fall back to file modification date
        mod_time = os.path.getmtime(file_path)
        mod_year = datetime.fromtimestamp(mod_time).year
        return str(mod_year)
    
    def _extract_from_pdf(self, file_path: str) -> List[str]:
        """Extract text chunks from PDF"""
        chunks = []
        
        try:
            # Try PyPDF2 first
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Split into smaller chunks by paragraphs
                    paragraphs = re.split(r'\n\s*\n', text)
                    for para in paragraphs:
                        if len(para.strip()) > 50:  # Only include substantial paragraphs
                            chunks.append(para.strip())
        except Exception as e:
            print(f"PyPDF2 extraction failed, trying alternative: {e}")
            
            try:
                # Try another PDF extraction library if available
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            paragraphs = re.split(r'\n\s*\n', text)
                            for para in paragraphs:
                                if len(para.strip()) > 50:
                                    chunks.append(para.strip())
            except ImportError:
                print("pdfplumber not installed, trying with subprocess")
                
                # Last resort: try pdftotext via subprocess
                try:
                    import subprocess
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix='.txt') as temp:
                        subprocess.run(['pdftotext', file_path, temp.name])
                        with open(temp.name, 'r') as f:
                            text = f.read()
                        
                        paragraphs = re.split(r'\n\s*\n', text)
                        for para in paragraphs:
                            if len(para.strip()) > 50:
                                chunks.append(para.strip())
                except Exception as e2:
                    print(f"All PDF extraction methods failed: {e2}")
        
        return chunks
    
    def _extract_from_text(self, file_path: str) -> List[str]:
        """Extract text chunks from text file"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Split by paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                if len(para.strip()) > 50:
                    chunks.append(para.strip())
        except Exception as e:
            print(f"Error extracting from text file: {e}")
        
        return chunks
    
    def _extract_from_csv(self, file_path: str) -> List[str]:
        """Extract text chunks from CSV file"""
        chunks = []
        
        try:
            import pandas as pd
            
            # Open regular or gzipped CSV
            if file_path.endswith('.gz'):
                df = pd.read_csv(file_path, compression='gzip')
            else:
                df = pd.read_csv(file_path)
            
            # Get text columns (columns with string data)
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if text_columns:
                # Concatenate text columns for each row
                for _, row in df.iterrows():
                    text_chunk = " ".join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                    if len(text_chunk.strip()) > 50:
                        chunks.append(text_chunk.strip())
            else:
                print(f"No text columns found in CSV: {file_path}")
        except Exception as e:
            print(f"Error extracting from CSV file: {e}")
        
        return chunks