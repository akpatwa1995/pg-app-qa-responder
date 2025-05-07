"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Change to direct import
from config import OLLAMA_HOST, OLLAMA_API_KEY, PERSIST_DIRECTORY, DOCS_METADATA_DIR, DATA_CHANGES_DIR

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Configure Ollama client
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

# Define persistent directories
PERSIST_DIRECTORY = os.path.join("data", "vectors")
DOCS_METADATA_DIR = os.path.join("data", "docs_metadata")
DATA_CHANGES_DIR = os.path.join("data", "changes")
os.makedirs(DOCS_METADATA_DIR, exist_ok=True)
os.makedirs(DATA_CHANGES_DIR, exist_ok=True)

def get_document_hash(file_content: bytes) -> str:
    """Generate a unique hash for a document based on its content."""
    return hashlib.sha256(file_content).hexdigest()

def save_document_metadata(file_name: str, doc_hash: str, collection_name: str) -> None:
    """Save metadata about a processed document."""
    metadata = {
        "file_name": file_name,
        "doc_hash": doc_hash,
        "collection_name": collection_name,
        "processed_date": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(DOCS_METADATA_DIR, f"{doc_hash}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def get_existing_collection(doc_hash: str) -> Optional[Chroma]:
    """Check if a document has been previously processed and return its collection."""
    metadata_path = os.path.join(DOCS_METADATA_DIR, f"{doc_hash}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            collection_name = metadata["collection_name"]
            
            # Update last accessed time
            metadata["last_accessed"] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Return existing collection
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=OLLAMA_HOST,
                api_key=OLLAMA_API_KEY,
                temperature=0
            )
            return Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name=collection_name
            )
    return None

def list_processed_documents() -> List[Dict[str, Any]]:
    """List all processed documents with their metadata."""
    documents = []
    for metadata_file in Path(DOCS_METADATA_DIR).glob("*.json"):
        with open(metadata_file, 'r') as f:
            documents.append(json.load(f))
    return sorted(documents, key=lambda x: x["last_accessed"], reverse=True)

def track_data_change(change_type: str, details: Dict[str, Any]) -> None:
    """
    Track changes made to test data.
    
    Args:
        change_type (str): Type of change (update, delete, deprecate)
        details (Dict[str, Any]): Details of the change
    """
    change_record = {
        "timestamp": datetime.now().isoformat(),
        "type": change_type,
        "details": details
    }
    
    # Create a unique filename based on timestamp
    filename = f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    change_path = os.path.join(DATA_CHANGES_DIR, filename)
    
    with open(change_path, 'w') as f:
        json.dump(change_record, f, indent=2)

def get_data_changes() -> List[Dict[str, Any]]:
    """
    Get all tracked data changes.
    
    Returns:
        List[Dict[str, Any]]: List of change records
    """
    changes = []
    for change_file in Path(DATA_CHANGES_DIR).glob("*.json"):
        with open(change_file, 'r') as f:
            changes.append(json.load(f))
    return sorted(changes, key=lambda x: x["timestamp"], reverse=True)

def update_document_metadata(doc_hash: str, updates: Dict[str, Any]) -> None:
    """
    Update metadata for a processed document.
    
    Args:
        doc_hash (str): Document hash
        updates (Dict[str, Any]): Updates to apply
    """
    metadata_path = os.path.join(DOCS_METADATA_DIR, f"{doc_hash}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update metadata
        metadata.update(updates)
        metadata["last_modified"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def mark_data_as_deprecated(doc_hash: str, data_id: str, reason: str) -> None:
    """
    Mark specific data as deprecated in a document.
    
    Args:
        doc_hash (str): Document hash
        data_id (str): Identifier of the data to deprecate
        reason (str): Reason for deprecation
    """
    metadata_path = os.path.join(DOCS_METADATA_DIR, f"{doc_hash}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if "deprecated_data" not in metadata:
            metadata["deprecated_data"] = []
        
        metadata["deprecated_data"].append({
            "id": data_id,
            "reason": reason,
            "deprecated_at": datetime.now().isoformat()
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Track the deprecation
        track_data_change("deprecate", {
            "document": metadata["file_name"],
            "data_id": data_id,
            "reason": reason
        })

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG For PG App QA Responder",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    
    # Get file content and generate hash
    file_content = file_upload.getvalue()
    doc_hash = get_document_hash(file_content)
    
    # Check if document was previously processed
    existing_collection = get_existing_collection(doc_hash)
    if existing_collection:
        logger.info(f"Found existing collection for document: {file_upload.name}")
        return existing_collection
    
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    
    with open(path, "wb") as f:
        f.write(file_content)
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Create unique collection name
    collection_name = f"pdf_{doc_hash}"
    
    # Create vector store with persistent storage
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST,
        api_key=OLLAMA_API_KEY,
        temperature=0
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=collection_name
    )
    
    # Save document metadata
    save_document_metadata(file_upload.name, doc_hash, collection_name)
    logger.info("Vector DB created with persistent storage")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Check for data management commands
    if question.lower().startswith(("update:", "remove:", "delete:", "learn:")):
        return handle_data_management_command(question, vector_db)
    
    # Initialize LLM with remote server configuration
    llm = ChatOllama(
        model=selected_model,
        base_url=OLLAMA_HOST,
        api_key=OLLAMA_API_KEY
    )
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are Tara, an AI Test Data Manager and PG App QA Assistant. Your primary responsibilities are:

1. Test Data Management:
   - Track and manage test data (MIDs, keys, flows, status)
   - Update test data based on user requests
   - Identify deprecated or outdated test data
   - Maintain data consistency across different test environments

2. PG App QA Support:
   - Answer queries about testing processes and procedures
   - Provide guidance on test automation and tools
   - Help troubleshoot testing issues
   - Share best practices and guidelines

3. Data Updates and Maintenance:
   - Process requests to update test data (e.g., "update MID for flow xyz")
   - Handle requests to remove deprecated data
   - Track data changes and maintain history
   - Validate data consistency

4. Learned Information:
   - Prioritize and use learned information when relevant
   - Combine learned information with existing knowledge
   - Provide specific details from learned data
   - Share contact information and specific instructions when available

When generating alternative questions, consider:
- Test data identifiers (MID, flow, status)
- User details (mobile, customer ID, email)
- Test environment configurations
- Automation setup and tools
- Current and deprecated test data
- Testing best practices
- Learned information and contact details

Generate 2 different versions of the user's question to retrieve relevant documents.
Original question: {question}""",
    )

    # RAG prompt template with enhanced context handling
    template = """You are Tara, a Test Data Manager and PG App QA Assistant. Answer the question based on the following context and guidelines:

Context: {context}

Question: {question}

Guidelines:
1. Always use the provided context to answer questions, especially when it contains specific information about:
   - Merchant IDs (MIDs)
   - Merchant keys
   - Flow configurations
   - Charges and fees
   - Test data details
   - Contact information
   - Specific instructions

2. When answering questions:
   - Provide direct answers based on the context
   - Don't add unnecessary restrictions or warnings
   - If the information is in the context, share it
   - Be specific and detailed in your responses
   - Share contact information when available
   - Provide exact instructions when available

3. For test data queries:
   - Share the exact values from the context
   - Include all relevant details
   - Don't hesitate to provide specific numbers or configurations
   - Reference the source of information when relevant

4. For data management:
   - Confirm when data is updated
   - Provide clear status of changes
   - Include all relevant details in responses
   - Track and report changes accurately

5. For learned information:
   - Prioritize learned information when relevant
   - Share contact details and specific instructions
   - Provide exact information as stored
   - Don't add unnecessary warnings about sharing information
   - If the question matches learned information, provide the exact details

Remember: Your primary role is to provide accurate information from the context. If the information is available in the context, share it without adding unnecessary restrictions or warnings. This includes contact information, specific instructions, and any other details that have been learned.

Answer the question while following these guidelines and maintaining a helpful, professional tone."""

    prompt = ChatPromptTemplate.from_template(template)

    # Set up retriever with enhanced search
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(
            search_kwargs={
                "k": 5,  # Retrieve more context for better answers
                "filter": {"type": "learned_information"}  # Prioritize learned information
            }
        ), 
        llm,
        prompt=QUERY_PROMPT
    )

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def handle_data_management_command(command: str, vector_db: Chroma) -> str:
    """
    Handle data management commands from chat.
    
    Commands:
    - update: <data_id> <new_value> - Update specific data
    - remove: <data_id> - Remove specific data
    - delete: <data_id> - Delete specific data
    - learn: <text> - Add new information to knowledge base
    
    Args:
        command (str): The command string
        vector_db (Chroma): The vector database
        
    Returns:
        str: Response message
    """
    try:
        cmd_parts = command.split(":", 1)
        if len(cmd_parts) != 2:
            return "Invalid command format. Please use: update:, remove:, delete:, or learn: followed by the details."
        
        cmd_type = cmd_parts[0].lower().strip()
        cmd_data = cmd_parts[1].strip()
        
        if cmd_type == "update":
            # Parse update command
            parts = cmd_data.split(maxsplit=1)
            if len(parts) != 2:
                return "Invalid update format. Use: update: <data_id> <new_value>"
            data_id, new_value = parts
            track_data_change("update", {
                "data_id": data_id,
                "new_value": new_value,
                "timestamp": datetime.now().isoformat()
            })
            return f"Updated data for {data_id} with new value: {new_value}"
            
        elif cmd_type in ["remove", "delete"]:
            # Parse remove/delete command
            data_id = cmd_data.strip()
            track_data_change("delete", {
                "data_id": data_id,
                "timestamp": datetime.now().isoformat()
            })
            return f"Removed data for {data_id}"
            
        elif cmd_type == "learn":
            # Add new information to knowledge base
            try:
                # Create embeddings for the new text
                embeddings = OllamaEmbeddings(
                    model="nomic-embed-text",
                    base_url=OLLAMA_HOST,
                    api_key=OLLAMA_API_KEY,
                    temperature=0
                )
                
                # Create a document with the new information
                from langchain_core.documents import Document
                new_doc = Document(
                    page_content=cmd_data,
                    metadata={
                        "source": "user_input",
                        "timestamp": datetime.now().isoformat(),
                        "type": "learned_information"
                    }
                )
                
                # Add the document to the vector store
                vector_db.add_documents([new_doc])
                
                # Track the learning
                track_data_change("learn", {
                    "text": cmd_data,
                    "timestamp": datetime.now().isoformat()
                })
                
                return f"Successfully learned and stored the following information:\n{cmd_data}\n\nThis information is now part of my knowledge base and I can use it to answer future questions."
                
            except Exception as e:
                logger.error(f"Error adding to knowledge base: {e}")
                return f"Error adding information to knowledge base: {str(e)}"
            
        else:
            return "Unknown command. Available commands: update:, remove:, delete:, learn:"
            
    except Exception as e:
        logger.error(f"Error handling data management command: {e}")
        return f"Error processing command: {str(e)}"

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()
            
            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF RAG for PG App QA Query Responder", divider="gray", anchor=False)

    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        # Try to load the last processed document
        processed_docs = list_processed_documents()
        if processed_docs:
            last_doc = processed_docs[0]  # Most recently processed document
            existing_collection = get_existing_collection(last_doc['doc_hash'])
            if existing_collection:
                st.session_state["vector_db"] = existing_collection
                st.session_state["last_processed_doc"] = last_doc
    if "last_processed_doc" not in st.session_state:
        st.session_state["last_processed_doc"] = None

    with col1:
        st.markdown("### 📄 Document Upload")
        file_upload = st.file_uploader("Upload a PDF document", type="pdf")
        
        # Show processed documents
        st.markdown("### 📚 Processed Documents")
        processed_docs = list_processed_documents()
        if processed_docs:
            for doc in processed_docs:
                with st.expander(f"📄 {doc['file_name']}"):
                    st.markdown(f"**Processed:** {doc['processed_date']}")
                    st.markdown(f"**Last Accessed:** {doc['last_accessed']}")
                    
                    # Add button to load this document
                    if st.button("Load Document", key=f"load_{doc['doc_hash']}"):
                        existing_collection = get_existing_collection(doc['doc_hash'])
                        if existing_collection:
                            st.session_state["vector_db"] = existing_collection
                            st.session_state["last_processed_doc"] = doc
                            st.success(f"Loaded document: {doc['file_name']}")
                            st.rerun()
                    
                    # Show deprecated data if any
                    if "deprecated_data" in doc:
                        st.markdown("**⚠️ Deprecated Data:**")
                        for deprecated in doc["deprecated_data"]:
                            st.markdown(f"- {deprecated['id']}: {deprecated['reason']}")
                    
                    # Add buttons for data management
                    if st.button("Manage Data", key=f"manage_{doc['doc_hash']}"):
                        st.session_state["selected_doc"] = doc
                        st.session_state["show_data_management"] = True
        
        # Data Management Section
        if st.session_state.get("show_data_management"):
            st.markdown("### 🔧 Data Management")
            doc = st.session_state["selected_doc"]
            
            # Data update form
            with st.form("data_update_form"):
                st.markdown(f"**Updating data in:** {doc['file_name']}")
                data_id = st.text_input("Data ID (e.g., MID, Flow ID)")
                update_type = st.selectbox(
                    "Update Type",
                    ["Update", "Deprecate", "Remove"]
                )
                reason = st.text_area("Reason for change")
                submitted = st.form_submit_button("Submit Change")
                
                if submitted:
                    if update_type == "Deprecate":
                        mark_data_as_deprecated(doc["doc_hash"], data_id, reason)
                        st.success(f"Data {data_id} marked as deprecated")
                    elif update_type == "Remove":
                        track_data_change("remove", {
                            "document": doc["file_name"],
                            "data_id": data_id,
                            "reason": reason
                        })
                        st.success(f"Data {data_id} marked for removal")
                    else:  # Update
                        track_data_change("update", {
                            "document": doc["file_name"],
                            "data_id": data_id,
                            "reason": reason
                        })
                        st.success(f"Data {data_id} update tracked")
            
            if st.button("Close Data Management"):
                st.session_state["show_data_management"] = False
                st.rerun()
        
        # Show recent changes
        st.markdown("### 📝 Recent Changes")
        changes = get_data_changes()
        if changes:
            for change in changes[:5]:  # Show last 5 changes
                with st.expander(f"{change['type'].title()} - {change['timestamp']}"):
                    st.json(change["details"])
        else:
            st.info("No changes recorded yet.")

        if file_upload is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    vector_db = create_vector_db(file_upload)
                    st.session_state["vector_db"] = vector_db
                    st.session_state["pdf_pages"] = extract_all_pages_as_images(file_upload)
                    st.session_state["file_upload"] = file_upload
                    st.session_state["last_processed_doc"] = {
                        "file_name": file_upload.name,
                        "doc_hash": get_document_hash(file_upload.getvalue()),
                        "processed_date": datetime.now().isoformat()
                    }
                    st.success("Document processed successfully!")

    with col2:
        st.markdown("### 💬 Chat Interface")
        
        # Show current document info if available
        if st.session_state["last_processed_doc"]:
            st.info(f"Currently using document: {st.session_state['last_processed_doc']['file_name']}")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Language Model",
            available_models,
            index=0 if available_models else None
        )

        # Add a flag to track if a query is being processed
        if "is_processing" not in st.session_state:
            st.session_state["is_processing"] = False
        if "pending_prompt" not in st.session_state:
            st.session_state["pending_prompt"] = None

        # Display chat messages
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input at the bottom, disabled if processing
        prompt = None
        if not st.session_state["is_processing"]:
            prompt = st.chat_input("Ask a question about the document")
            if prompt:
                st.session_state["pending_prompt"] = prompt
                st.session_state["is_processing"] = True
                st.rerun()
        else:
            st.chat_input("Processing previous query...", disabled=True, key="disabled_chat_input")

        # Handle pending prompt if any
        if st.session_state["is_processing"] and st.session_state["pending_prompt"]:
            user_prompt = st.session_state["pending_prompt"]
            st.session_state["messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state["vector_db"] is not None:
                        response = process_question(
                            user_prompt,
                            st.session_state["vector_db"],
                            selected_model
                        )
                    else:
                        if processed_docs:
                            response = "I don't have any document loaded at the moment. You can load a previously processed document from the list on the left, or upload a new PDF to ask questions about it."
                        else:
                            response = "I'm here to help! Please feel free to ask any questions. If you'd like to ask questions about a specific document, you can upload a PDF using the upload button on the left."
                    st.markdown(response)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )
            st.session_state["is_processing"] = False
            st.session_state["pending_prompt"] = None
            st.rerun()

        # Control buttons in a row
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state["messages"] = []
                st.rerun()
        with col2:
            if st.button("Delete Current Document"):
                if st.session_state["vector_db"] is not None:
                    delete_vector_db(st.session_state["vector_db"])
                    st.session_state["vector_db"] = None
                    st.session_state["pdf_pages"] = None
                    st.session_state["file_upload"] = None
                    st.session_state["last_processed_doc"] = None
                    st.success("Document deleted successfully!")
                else:
                    st.info("No document is currently loaded.")


if __name__ == "__main__":
    main()