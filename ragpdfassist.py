import streamlit as st
import tempfile
import os
import hashlib
import json
from datetime import datetime
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    
    .info-box {
        background-color: #1E3A8A;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #3B82F6;
    }
    
    .success-box {
        background-color: #065F46;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #10B981;
    }
    
    /* Chat history styling */
    .chat-history {
        max-height: 400px;
        overflow-y: auto;
        background-color: #1A1A1A;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use ONLY the provided context to answer the query. 
If the context doesn't contain relevant information, state that you don't know based on the current document.
Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Model Configuration - Using DeepSeek R1 for better quality responses
MODEL_NAME = "deepseek-r1:latest"  # High-quality model for better responses
EMBEDDING_MODEL_NAME = "deepseek-r1:latest"  # Using the same model for embeddings

# Pinecone Configuration
PINECONE_INDEX_NAME = "pdf-assistant-index"
PINECONE_ENVIRONMENT = "us-east-1"  # Change based on your preference
EMBEDDING_DIMENSION = 4096  # DeepSeek R1 embedding dimension (will be auto-detected)

try:
    EMBEDDING_MODEL = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    LANGUAGE_MODEL = OllamaLLM(model=MODEL_NAME, temperature=0.1)  # Lower temperature for consistency
    
    # Auto-detect embedding dimension
    test_embedding = EMBEDDING_MODEL.embed_query("test")
    EMBEDDING_DIMENSION = len(test_embedding)
    st.info(f"🔍 Auto-detected embedding dimension: {EMBEDDING_DIMENSION}")
    
except Exception as e:
    st.error(f"Error loading Ollama models: {e}")
    st.info("Please check your Ollama installation and model names. Run 'ollama list' to see available models.")
    EMBEDDING_DIMENSION = 4096  # Fallback dimension

# Initialize Pinecone
@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone connection and create index if needed"""
    try:
        # Get API key from Streamlit secrets or environment
        api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        if not api_key:
            st.error("❌ Pinecone API key not found! Please set PINECONE_API_KEY in secrets.toml or environment variable.")
            return None, None
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            st.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            # Wait for index to be ready
            st.info("⏳ Waiting for index to be ready...")
            time.sleep(15)  # Increased wait time
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=EMBEDDING_MODEL,
            text_key="text"
        )
        
        return pc, vector_store
        
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None, None

# Initialize session state
if 'pinecone_client' not in st.session_state:
    st.session_state.pinecone_client = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = set()
if 'use_pinecone' not in st.session_state:
    st.session_state.use_pinecone = True
if 'current_pdf_hash' not in st.session_state:
    st.session_state.current_pdf_hash = None
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_file_hash(file_content):
    """Generate unique hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def check_document_exists(file_hash, vector_store):
    """Check if document already exists in Pinecone"""
    try:
        # Query with the file hash as metadata filter
        results = vector_store.similarity_search(
            query="test", 
            k=1, 
            filter={"file_hash": file_hash}
        )
        return len(results) > 0
    except:
        return False

def process_uploaded_pdf(uploaded_file, use_pinecone=True):
    """Process uploaded PDF with Pinecone or fallback to InMemory"""
    file_content = uploaded_file.getbuffer()
    file_hash = get_file_hash(file_content)
    
    # Update current PDF tracking
    st.session_state.current_pdf_hash = file_hash
    st.session_state.current_pdf_name = uploaded_file.name
    
    # Clear chat history when new PDF is uploaded
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
    
    # Check if using Pinecone and document already exists
    if use_pinecone and st.session_state.vector_store:
        if check_document_exists(file_hash, st.session_state.vector_store):
            st.info("📋 Document already processed and stored in Pinecone!")
            return st.session_state.vector_store
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load PDF documents
        document_loader = PDFPlumberLoader(tmp_file_path)
        raw_documents = document_loader.load()
        
        # Optimized chunking for better performance
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for faster processing
            chunk_overlap=100,  # Increased overlap for better context
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting
        )
        document_chunks = text_processor.split_documents(raw_documents)
        
        # Add metadata to chunks for isolation
        for i, chunk in enumerate(document_chunks):
            chunk.metadata.update({
                "file_hash": file_hash,
                "file_name": uploaded_file.name,
                "processed_at": datetime.now().isoformat(),
                "chunk_id": f"{file_hash}_{i}",  # Unique chunk identifier
                "source_doc": file_hash  # Key for filtering current document
            })
        
        if use_pinecone and st.session_state.vector_store:
            # Use Pinecone vector store
            try:
                st.info("📤 Uploading document chunks to Pinecone...")
                doc_ids = st.session_state.vector_store.add_documents(document_chunks)
                st.success(f"✅ Successfully stored {len(document_chunks)} chunks in Pinecone!")
                st.session_state.processed_documents.add(file_hash)
                return st.session_state.vector_store
            except Exception as e:
                st.error(f"Error storing in Pinecone: {e}")
                st.warning("⚠️ Falling back to InMemory storage")
                vector_store = InMemoryVectorStore(EMBEDDING_MODEL)
                vector_store.add_documents(document_chunks)
                return vector_store
        else:
            # Fallback to InMemory vector store
            st.warning("⚠️ Using InMemory storage (slower performance)")
            vector_store = InMemoryVectorStore(EMBEDDING_MODEL)
            vector_store.add_documents(document_chunks)
            return vector_store
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def find_related_documents(query, vector_store, k=3):  # Reduced k for faster retrieval
    """Find relevant documents using similarity search - ISOLATED TO CURRENT PDF"""
    try:
        if st.session_state.current_pdf_hash and st.session_state.use_pinecone:
            # Filter by current PDF hash for isolation
            results = vector_store.similarity_search(
                query, 
                k=k,
                filter={"source_doc": st.session_state.current_pdf_hash}  # KEY CHANGE: Isolate to current PDF
            )
        else:
            # Fallback for InMemory storage
            results = vector_store.similarity_search(query, k=k)
            # Filter results manually for InMemory storage
            if st.session_state.current_pdf_hash:
                results = [doc for doc in results if doc.metadata.get("file_hash") == st.session_state.current_pdf_hash]
        
        return results
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

def generate_answer(user_query, context_documents):
    """Generate answer using the language model - Optimized for speed"""
    if not context_documents:
        return "No relevant context found in the current document for your query."
    
    # Limit context length for faster processing
    context_text = "\n\n".join([doc.page_content[:500] for doc in context_documents])  # Truncate for speed
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    
    try:
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})
    except Exception as e:
        return f"Error generating response: {e}"

def add_to_chat_history(user_input, ai_response):
    """Add conversation to chat history"""
    st.session_state.chat_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "user": user_input,
        "assistant": ai_response
    })

def display_chat_history():
    """Display chat history"""
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
            with st.expander(f"🕒 {chat['timestamp']} - Conversation {len(st.session_state.chat_history)-i}"):
                st.markdown(f"**👤 You:** {chat['user']}")
                st.markdown(f"**🤖 Assistant:** {chat['assistant']}")

# UI Configuration
st.title("📘 DocuMind AI Pro")
st.markdown("### Your Intelligent Document Assistant with Isolated Retrieval")
st.markdown("---")

# Initialize Pinecone
with st.spinner("Initializing Pinecone connection..."):
    pinecone_client, pinecone_vector_store = initialize_pinecone()
    
if pinecone_client and pinecone_vector_store:
    st.session_state.pinecone_client = pinecone_client
    st.session_state.vector_store = pinecone_vector_store
    st.session_state.use_pinecone = True
    
    st.markdown("""
    <div class="success-box">
        ✅ <strong>Pinecone Connected Successfully!</strong><br>
        📊 Using high-performance vector database with isolated retrieval<br>
        🔄 Documents are automatically cached to avoid reprocessing<br>
        🎯 Queries are isolated to the current document only
    </div>
    """, unsafe_allow_html=True)
else:
    st.session_state.use_pinecone = False
    st.markdown("""
    <div class="info-box">
        ⚠️ <strong>Pinecone Not Available</strong><br>
        🔄 Falling back to InMemory storage with isolation<br>
        📝 Please configure Pinecone for better performance
    </div>
    """, unsafe_allow_html=True)

# Current Document Status
if st.session_state.current_pdf_name:
    st.markdown(f"""
    <div class="success-box">
        📄 <strong>Active Document:</strong> {st.session_state.current_pdf_name}<br>
        🎯 All queries are isolated to this document only<br>
        💬 Chat History: {len(st.session_state.chat_history)} conversations
    </div>
    """, unsafe_allow_html=True)

# Configuration Section
with st.expander("⚙️ Configuration & Stats"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Database Status:**")
        if st.session_state.use_pinecone:
            st.success("🔗 Pinecone Connected")
            if st.session_state.vector_store:
                try:
                    # Get index stats
                    index_stats = st.session_state.pinecone_client.Index(PINECONE_INDEX_NAME).describe_index_stats()
                    st.write(f"📊 Total Vectors: {index_stats.get('total_vector_count', 'N/A')}")
                    if st.session_state.current_pdf_hash:
                        st.write(f"🎯 Current PDF: {st.session_state.current_pdf_name}")
                except:
                    st.write("📊 Stats unavailable")
        else:
            st.warning("💾 InMemory Storage")
    
    with col2:
        st.write("**Model Configuration:**")
        st.write(f"🤖 LLM: {MODEL_NAME}")
        st.write(f"🔤 Embeddings: {EMBEDDING_MODEL_NAME}")
        st.write(f"📏 Dimension: {EMBEDDING_DIMENSION}")
        st.write(f"⚡ Quality Optimized with DeepSeek")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis. Queries will be isolated to this document only.",
    accept_multiple_files=False
)

if uploaded_pdf:
    # Process the uploaded PDF
    with st.spinner("Processing document..."):
        current_vector_store = process_uploaded_pdf(uploaded_pdf, st.session_state.use_pinecone)
    
    if current_vector_store:
        if not st.session_state.use_pinecone:
            st.session_state.vector_store = current_vector_store
        
        st.success("✅ Document processed successfully! Ask your questions below.")
        
        # Document info
        file_size = len(uploaded_pdf.getbuffer()) / 1024  # KB
        st.info(f"📄 **{uploaded_pdf.name}** ({file_size:.1f} KB) - Ready for isolated queries!")
    
    # Display chat history
    if st.session_state.chat_history:
        display_chat_history()
    
    # Chat interface
    if st.session_state.vector_store:
        user_input = st.chat_input("Enter your question about the current document...")
        
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("Analyzing current document..."):
                start_time = time.time()
                relevant_docs = find_related_documents(user_input, st.session_state.vector_store)
                search_time = time.time() - start_time
                
                ai_response = generate_answer(user_input, relevant_docs)
                total_time = time.time() - start_time
                
                # Add to chat history
                add_to_chat_history(user_input, ai_response)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
                
                # Performance metrics
                with st.expander("📊 Query Performance"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Search Time", f"{search_time:.2f}s")
                    with col2:
                        st.metric("Total Time", f"{total_time:.2f}s")
                    with col3:
                        st.metric("Docs Found", len(relevant_docs))
                    with col4:
                        st.metric("Isolation", "✅ Active")

# Chat Management
if st.session_state.chat_history:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    with col2:
        if st.button("📥 Download Chat History"):
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "Download as JSON",
                chat_json,
                f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚀 Powered by DeepSeek R1 + Pinecone | Enhanced Performance, Isolated Retrieval & Chat History<br>
    🎯 Each query is isolated to the current document only | 🧠 High-quality responses with DeepSeek
</div>
""", unsafe_allow_html=True)