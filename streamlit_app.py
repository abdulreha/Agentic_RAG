"""Streamlit UI for Agentic RAG System - Production Version"""

import streamlit as st
from pathlib import Path
import sys
import time
import tempfile
import os
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Search",
    page_icon="🔍",
    layout="wide"
)

# Enhanced CSS with wider containers
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    .main .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory and return file paths"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths, temp_dir

def initialize_rag_with_files(file_paths=None, urls=None):
    """Initialize the RAG system with files or URLs"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Process documents
        if file_paths:
            documents = doc_processor.process_files(file_paths)
        elif urls:
            documents = doc_processor.process_urls(urls)
        else:
            raise ValueError("No documents provided for processing")
        
        if not documents:
            raise ValueError("No documents were processed")
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
        
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("Agentic RAG Document Search")
    st.markdown("Upload your documents or provide URLs to start searching")
    
    # Document Source Selection
    st.markdown("### Document Source")
    
    source_option = st.radio(
        "Choose your document source:",
        ["Upload Files", "Custom URLs"],
        horizontal=True
    )
    
    if source_option == "Upload Files":
        st.markdown("#### Upload Your Documents")
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=['pdf', 'txt', 'docx', 'md'],
                accept_multiple_files=True,
                help="Supported formats: PDF, TXT, DOCX, MD"
            )
            
            if uploaded_files:
                st.success(f" {len(uploaded_files)} file(s) uploaded successfully!")
                
                # Show uploaded files
                st.markdown("**Uploaded Files:**")
                for file in uploaded_files:
                    file_size = len(file.getbuffer()) / 1024  # Size in KB
                    st.markdown(f"-  {file.name} ({file_size:.1f} KB)")
                
                # Initialize button
                if st.button(" Process Uploaded Files", use_container_width=True):
                    with st.spinner("Processing uploaded files..."):
                        file_paths, temp_dir = save_uploaded_files(uploaded_files)
                        rag_system, num_chunks = initialize_rag_with_files(file_paths=file_paths)
                        
                        if rag_system:
                            st.session_state.rag_system = rag_system
                            st.session_state.initialized = True
                            st.success(f" System ready with {num_chunks} document chunks!")
                            st.rerun()
                        
                        # Clean up temp files
                        shutil.rmtree(temp_dir, ignore_errors=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Custom URLs
        st.markdown("#### Custom URLs")
        
        # URL input
        url_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/document1.pdf\nhttps://example.com/document2.html",
            height=100
        )
        
        if url_input.strip():
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            st.markdown("**URLs to process:**")
            for i, url in enumerate(urls, 1):
                st.markdown(f"{i}. {url}")
            
            if st.button(" Process Custom URLs", use_container_width=True):
                with st.spinner("Processing custom URLs..."):
                    rag_system, num_chunks = initialize_rag_with_files(urls=urls)
                    
                    if rag_system:
                        st.session_state.rag_system = rag_system
                        st.session_state.initialized = True
                        st.success(f" System ready with {num_chunks} document chunks!")
                        st.rerun()
    
    st.markdown("---")
    
    # Search interface (only show if system is initialized)
    if st.session_state.initialized:
        st.markdown("###  Search Interface")
        
        # Search interface - using columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know?"
            )
        
        with col2:
            st.write("")  # Add some space
            submit = st.button(" Search", use_container_width=True)
        
        # Process search
        if submit and question:
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    
                    # Get answer
                    result = st.session_state.rag_system.run(question)
                    
                    elapsed_time = time.time() - start_time
                    
                    # Add to history
                    st.session_state.history.append({
                        'question': question,
                        'answer': result['answer'],
                        'time': elapsed_time
                    })
                    
                    # Display answer
                    st.markdown("###  Answer")
                    st.success(result['answer'])
                    
                    # Show retrieved docs with wider layout
                    with st.expander(" Source Documents", expanded=True):
                        if 'retrieved_docs' in result and result['retrieved_docs']:
                            # Create columns for better document display
                            if len(result['retrieved_docs']) > 1:
                                # If multiple docs, show them in columns
                                num_docs = len(result['retrieved_docs'])
                                if num_docs <= 2:
                                    cols = st.columns(2)
                                else:
                                    cols = st.columns(3)
                                
                                for i, doc in enumerate(result['retrieved_docs']):
                                    col_idx = i % len(cols)
                                    with cols[col_idx]:
                                        st.markdown(f"**Document {i+1}**")
                                        st.text_area(
                                            f"Content {i+1}",
                                            doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                            height=200,
                                            disabled=True,
                                            key=f"doc_{i}",
                                            label_visibility="collapsed"
                                        )
                                        # Show metadata if available
                                        if hasattr(doc, 'metadata') and doc.metadata:
                                            with st.expander("Metadata", expanded=False):
                                                st.json(doc.metadata)
                            else:
                                # Single document - use full width
                                doc = result['retrieved_docs'][0]
                                st.markdown("**Source Document**")
                                st.text_area(
                                    "Document Content",
                                    doc.page_content,
                                    height=300,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    with st.expander("Document Metadata"):
                                        st.json(doc.metadata)
                        else:
                            st.warning("No source documents found in result")
                    
                    st.caption(f" Response time: {elapsed_time:.2f} seconds")
            else:
                st.error("System not initialized. Please load documents first.")
    
    else:
        st.info(" Please select and load your documents above to start searching")
    
    st.markdown("---")
    
    # Show history with improved layout
    if st.session_state.history:
        st.markdown("###  Recent Searches")
        
        for i, item in enumerate(reversed(st.session_state.history[-3:])):  # Show last 3
            with st.container():
                # Create columns for question and answer
                q_col, a_col = st.columns([1, 2])
                
                with q_col:
                    st.markdown(f"**Q{len(st.session_state.history)-i}:** {item['question']}")
                    st.caption(f" {item['time']:.2f}s")
                
                with a_col:
                    st.markdown(f"**A:** {item['answer'][:300]}{'...' if len(item['answer']) > 300 else ''}")
                
                st.markdown("")
    
    # Reset system button
    if st.session_state.initialized:
        st.markdown("---")
        if st.button(" Reset System", use_container_width=True):
            st.session_state.rag_system = None
            st.session_state.initialized = False
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()