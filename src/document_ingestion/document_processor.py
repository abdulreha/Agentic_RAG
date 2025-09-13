"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            print(f"Error loading URL {url}: {str(e)}")
            return []

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        try:
            loader = PyPDFDirectoryLoader(str(directory))
            return loader.load()
        except Exception as e:
            print(f"Error loading PDF directory {directory}: {str(e)}")
            return []

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            return loader.load()
        except Exception as e:
            print(f"Error loading TXT file {file_path}: {str(e)}")
            return []

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a single PDF file"""
        try:
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {str(e)}")
            return []
    
    def load_from_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load document from a single file based on its extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of loaded documents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"File does not exist: {file_path}")
            return []
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.load_from_pdf(file_path)
        elif extension in ['.txt', '.md']:
            return self.load_from_txt(file_path)
        else:
            print(f"Unsupported file type: {extension}")
            return []
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or individual files

        Args:
            sources: List of URLs, directory paths, or file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                # Handle URL
                docs.extend(self.load_from_url(src))
            else:
                # Handle local path
                path = Path(src)
                
                if path.is_file():
                    # Single file
                    docs.extend(self.load_from_file(path))
                elif path.is_dir():
                    # Directory - load all PDFs
                    docs.extend(self.load_from_pdf_dir(path))
                else:
                    print(f"Path does not exist: {src}")
        
        return docs
    
    def process_files(self, file_paths: List[str]) -> List[Document]:
        """
        Process uploaded files directly
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed document chunks
        """
        docs = []
        
        for file_path in file_paths:
            file_docs = self.load_from_file(file_path)
            docs.extend(file_docs)
        
        if not docs:
            raise ValueError("No documents could be loaded from the provided files")
        
        return self.split_documents(docs)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        if not documents:
            return []
        
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents from URLs
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = []
        
        for url in urls:
            url_docs = self.load_from_url(url)
            docs.extend(url_docs)
        
        if not docs:
            raise ValueError("No documents could be loaded from the provided URLs")
        
        return self.split_documents(docs)
    
    def process_mixed_sources(self, sources: List[str]) -> List[Document]:
        """
        Process mixed sources (URLs, files, directories)
        
        Args:
            sources: List of mixed sources
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(sources)
        
        if not docs:
            raise ValueError("No documents could be loaded from the provided sources")
        
        return self.split_documents(docs)