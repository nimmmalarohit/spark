import logging
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pdfplumber

from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import EventType, ConversationPaused, UserUtteranceReverted
from optimus.sdk.action_meta import OptimusAction
from optimus.message_components.postback_button.button import PostBackButton, utter_buttons

logger = logging.getLogger(__name__)

# Define paths
DEFAULT_PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gcp_security_guidelines.pdf")
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wiki_knowledge")

class WikiKnowledgeBase:
    """Vector database for storing and retrieving knowledge content using TF-IDF."""
    
    def __init__(self):
        """Initialize the knowledge base with TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectors = None
        self.documents = []
        self.metadata = []
        self.initialized = False
        self.creation_time = None
        self.pdf_last_modified = None
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Add documents to the knowledge base."""
        if not documents:
            return
            
        # Create or update TF-IDF vectors
        if self.initialized and len(self.documents) > 0:
            # Combine existing and new documents for vectorization
            all_docs = self.documents + documents
            self.vectors = self.vectorizer.fit_transform(all_docs)
        else:
            # First time adding documents
            self.vectors = self.vectorizer.fit_transform(documents)
            self.initialized = True
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # Update timestamps
        self.creation_time = datetime.now().timestamp()
        if os.path.exists(DEFAULT_PDF_PATH):
            self.pdf_last_modified = os.path.getmtime(DEFAULT_PDF_PATH)
        
        logger.info(f"Added {len(documents)} documents to knowledge base. Total: {len(self.documents)}")
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents in the knowledge base."""
        if not self.initialized or self.vectors is None:
            logger.warning("Knowledge base not initialized yet.")
            return []
        
        # Create query vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Return results with scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                score = similarities[idx]
                # Only include results with meaningful similarity
                if score > 0.1:
                    results.append((self.documents[idx], self.metadata[idx], float(score)))
        
        return results
    
    def save(self, file_path: str):
        """Save knowledge base to disk."""
        # Convert sparse matrix to list for JSON serialization
        vectors_list = self.vectors.toarray().tolist() if self.vectors is not None else []
        
        # Save vocabulary, document data, and timestamps
        with open(f"{file_path}.json", "w") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "vectors": vectors_list,
                "vocabulary": self.vectorizer.vocabulary_,
                "idf": self.vectorizer.idf_.tolist() if hasattr(self.vectorizer, 'idf_') else [],
                "creation_time": self.creation_time,
                "pdf_last_modified": self.pdf_last_modified
            }, f)
            
        logger.info(f"Saved knowledge base to {file_path}")
        
    def load(self, file_path: str):
        """Load knowledge base from disk."""
        if os.path.exists(f"{file_path}.json"):
            with open(f"{file_path}.json", "r") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.creation_time = data.get("creation_time")
                self.pdf_last_modified = data.get("pdf_last_modified")
                
                # Rebuild vectorizer
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.vectorizer.vocabulary_ = data["vocabulary"]
                
                if "idf" in data and data["idf"]:
                    self.vectorizer.idf_ = np.array(data["idf"])
                
                # Convert back to sparse matrix if needed
                if "vectors" in data and data["vectors"]:
                    self.vectors = np.array(data["vectors"])
                    
                self.initialized = True
                logger.info(f"Loaded knowledge base from {file_path} with {len(self.documents)} documents")
                
                # Check if PDF has been modified since knowledge base was created
                if os.path.exists(DEFAULT_PDF_PATH):
                    current_mtime = os.path.getmtime(DEFAULT_PDF_PATH)
                    if not self.pdf_last_modified or current_mtime > self.pdf_last_modified:
                        logger.info("PDF has been modified since knowledge base was created. Knowledge base should be refreshed.")
                        return False
                return True
        else:
            logger.warning(f"Knowledge base file {file_path}.json not found")
            return False
    
    def needs_update(self) -> bool:
        """Check if knowledge base needs to be updated based on PDF modification time."""
        if not self.pdf_last_modified or not os.path.exists(DEFAULT_PDF_PATH):
            return True
            
        current_mtime = os.path.getmtime(DEFAULT_PDF_PATH)
        return current_mtime > self.pdf_last_modified


class PDFContentExtractor:
    """Extracts content from PDF files and processes it for the knowledge base."""
    
    def __init__(self):
        """Initialize the PDF content extractor."""
        self.knowledge_base = WikiKnowledgeBase()
        
    def extract_from_pdf(self, pdf_path: str) -> bool:
        """Extract content from a PDF file using pdfplumber."""
        logger.info(f"Extracting content from PDF: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Process each page
                all_chunks = []
                
                for i, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {i+1}/{len(pdf.pages)}")
                    text = page.extract_text()
                    
                    # Process text into chunks
                    chunks = self._process_text(text, page_num=i+1)
                    all_chunks.extend(chunks)
                
                # Add chunks to knowledge base
                documents = []
                metadata_list = []
                
                for text, meta in all_chunks:
                    documents.append(text)
                    metadata_list.append({
                        "source": pdf_path,
                        "page": meta.get("page", 0),
                        "heading": meta.get("heading", ""),
                    })
                
                # Add to knowledge base
                self.knowledge_base.add_documents(documents, metadata_list)
                
                # Save knowledge base
                self.knowledge_base.save(KNOWLEDGE_BASE_PATH)
                
                logger.info(f"Extracted {len(documents)} chunks from PDF")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {str(e)}")
            return False
            
    def _process_text(self, text: str, page_num: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Process text into logical chunks by sections."""
        if not text:
            return []
            
        # Split by potential headers (lines followed by blank lines)
        lines = text.split('\n')
        chunks = []
        current_heading = None
        current_text = []
        
        # Process text line by line
        line_position = 0
        for line in lines:
            line_position += 1
            stripped_line = line.strip()
            
            # Check if this is a heading
            is_heading = False
            if stripped_line:
                # Common markdown headings
                if stripped_line.startswith('#'):
                    is_heading = True
                # Short lines that may be headings (but not too short)
                elif 3 <= len(stripped_line) <= 80:
                    # Check if it's followed by a blank line or if it has title case
                    if line_position < len(lines) - 1 and not lines[line_position].strip():
                        is_heading = True
                    # Title case often indicates headings
                    elif stripped_line.istitle():
                        is_heading = True
            
            # If heading found and we already have content, save the current chunk
            if is_heading and current_heading is not None and current_text:
                chunk_text = current_heading + '\n\n' + '\n'.join(current_text)
                chunk_metadata = {
                    "page": page_num,
                    "heading": current_heading,
                }
                chunks.append((chunk_text, chunk_metadata))
                current_text = []
            
            # Update heading or add to current text
            if is_heading:
                current_heading = stripped_line.lstrip('#').strip()
            elif stripped_line:
                current_text.append(stripped_line)
        
        # Add the last chunk if exists
        if current_heading and current_text:
            chunk_text = current_heading + '\n\n' + '\n'.join(current_text)
            chunk_metadata = {
                "page": page_num,
                "heading": current_heading,
            }
            chunks.append((chunk_text, chunk_metadata))
        
        # If no chunks were created, create one from the entire text
        if not chunks and text.strip():
            # Try to detect logical sections
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 1:
                # Create multiple chunks from paragraphs
                for i, para in enumerate(paragraphs):
                    if len(para.strip()) > 20:  # Only include substantial paragraphs
                        chunk_metadata = {
                            "page": page_num,
                            "heading": f"Section {i+1}",
                        }
                        chunks.append((para.strip(), chunk_metadata))
            else:
                # One chunk for the whole page
                chunk_metadata = {
                    "page": page_num,
                    "heading": "Page content",
                }
                chunks.append((text.strip(), chunk_metadata))
        
        return chunks


# Function to check if knowledge base is up to date
def is_knowledge_base_current() -> bool:
    """Check if knowledge base is up to date with the PDF."""
    kb = WikiKnowledgeBase()
    if not kb.load(KNOWLEDGE_BASE_PATH):
        return False
        
    return not kb.needs_update()


# Initialize or refresh the knowledge base
def initialize_knowledge_base(force_refresh=False) -> bool:
    """Initialize the knowledge base from the repository PDF."""
    # Check if knowledge base exists and is up to date
    if not force_refresh and is_knowledge_base_current():
        logger.info("Knowledge base is up to date.")
        return True
    
    # Knowledge base doesn't exist or needs refresh
    logger.info("Creating or refreshing knowledge base from PDF...")
    
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF file not found: {DEFAULT_PDF_PATH}")
        return False
        
    # If knowledge base file exists, delete it to ensure clean recreation
    if os.path.exists(f"{KNOWLEDGE_BASE_PATH}.json"):
        try:
            os.remove(f"{KNOWLEDGE_BASE_PATH}.json")
        except Exception as e:
            logger.warning(f"Could not remove old knowledge base file: {e}")
    
    extractor = PDFContentExtractor()
    return extractor.extract_from_pdf(DEFAULT_PDF_PATH)


# Function to search knowledge base leveraging Rasa's NLU
async def search_wiki_knowledge(tracker: Tracker) -> Tuple[str, Optional[str]]:
    """Search the wiki knowledge base using Rasa NLU information."""
    try:
        # Check if knowledge base needs updating
        if not is_knowledge_base_current():
            logger.info("PDF has been modified. Refreshing knowledge base...")
            initialize_knowledge_base(force_refresh=True)
        
        # Get Rasa's intent and entities information
        latest_message = tracker.latest_message
        intent = latest_message.get("intent", {}).get("name")
        confidence = latest_message.get("intent", {}).get("confidence", 0.0)
        entities = latest_message.get("entities", [])
        query = latest_message.get("text")
        
        # Load knowledge base
        knowledge_base = WikiKnowledgeBase()
        if not knowledge_base.load(KNOWLEDGE_BASE_PATH):
            success = initialize_knowledge_base()
            if not success:
                return None, "Knowledge base could not be initialized"
        
        # Enhance query with entity information from Rasa
        enhanced_query = query
        extracted_entities = []
        for entity in entities:
            entity_value = entity.get("value")
            if entity_value:
                extracted_entities.append(entity_value)
        
        if extracted_entities:
            enhanced_query = f"{query} {' '.join(extracted_entities)}"
        
        # Search with enhanced query
        results = knowledge_base.search(enhanced_query, k=3)
        if not results:
            return None, "No results found"
        
        # Get best result
        content, metadata, score = results[0]
        
        # Process content to extract complete information
        # Important: Don't truncate the content!
        if '\n\n' in content:
            heading, full_content = content.split('\n\n', 1)
        else:
            heading = metadata.get('heading', '')
            full_content = content
        
        # Remove any formatting characters that might cause truncation
        clean_content = full_content.replace('#', '').strip()
        
        # Return the complete content
        return clean_content, None
    except Exception as e:
        logger.error(f"Error searching wiki: {str(e)}")
        return None, f"Error: {str(e)}"


class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):
    """Enhanced default fallback action that uses knowledge base."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        latest_message = tracker.latest_message.get('text')
        logger.info(f'### User query "{latest_message}" falling back ###')
        
        # Try to find relevant information in knowledge base using Rasa NLU
        wiki_response, error = await search_wiki_knowledge(tracker)
        
        if wiki_response:
            # Found relevant information - respond with just the content
            dispatcher.utter_message(text=wiki_response)
            return []
        
        # Original fallback behavior
        draft_email_button = PostBackButton.with_intent(title="Draft email", intent="gcp_feedback")
        utter_buttons(
            dispatcher=dispatcher, 
            text="Sorry, I am not trained for this. Please share query details if it is frequently required and we will set it up. Please click \"Draft email \" button, to share query details if required.",
            buttons=[draft_email_button]
        )
        return [ConversationPaused(), UserUtteranceReverted()]


class ActionRefreshKnowledgeBase(OptimusAction, action_name="action_refresh_knowledge_base"):
    """Action to manually refresh the knowledge base."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        logger.info("Manual knowledge base refresh requested")
        
        success = initialize_knowledge_base(force_refresh=True)
        
        if success:
            dispatcher.utter_message(text="Knowledge base has been successfully refreshed with the latest PDF content.")
        else:
            dispatcher.utter_message(text="Failed to refresh knowledge base. Please check logs for details.")
            
        return []
