import logging
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pdfplumber

from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import EventType, ConversationPaused, UserUtteranceReverted
from optimus.sdk.action_meta import OptimusAction

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
        
        logger.info(f"Added {len(documents)} documents to knowledge base. Total: {len(self.documents)}")
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents in the knowledge base."""
        if not self.initialized or self.vectors is None:
            logger.warning("Knowledge base not initialized yet.")
            return []
        
        # Create query vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all documents
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]  # Sort and take last k in reverse order
        
        # Return results with scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                score = similarities[idx]
                # Only include results with some similarity
                if score > 0.1:
                    results.append((self.documents[idx], self.metadata[idx], float(score)))
        
        return results
    
    def save(self, file_path: str):
        """Save knowledge base to disk."""
        # Convert sparse matrix to list for JSON serialization
        vectors_list = self.vectors.toarray().tolist() if self.vectors is not None else []
        
        # Save vocabulary and document data
        with open(f"{file_path}.json", "w") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "vectors": vectors_list,
                "vocabulary": self.vectorizer.vocabulary_,
                "idf": self.vectorizer.idf_.tolist() if hasattr(self.vectorizer, 'idf_') else []
            }, f)
            
        logger.info(f"Saved knowledge base to {file_path}")
        
    def load(self, file_path: str):
        """Load knowledge base from disk."""
        if os.path.exists(f"{file_path}.json"):
            with open(f"{file_path}.json", "r") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                
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
                return True
        else:
            logger.warning(f"Knowledge base file {file_path}.json not found")
            return False


class PDFContentExtractor:
    """Extracts content from PDF files and processes it for the knowledge base."""
    
    def __init__(self):
        """Initialize the PDF content extractor."""
        self.knowledge_base = WikiKnowledgeBase()
        
    def extract_from_pdf(self, pdf_path: str) -> bool:
        """Extract content from a PDF file and add it to the knowledge base."""
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
            if stripped_line and stripped_line.startswith('#'):  # Markdown style heading
                is_heading = True
            elif len(stripped_line) > 0 and len(stripped_line) < 80:  # Potential heading
                next_line_idx = line_position
                if next_line_idx < len(lines) and not lines[next_line_idx].strip():
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
        
        # If no chunks were created (no clear headings), create one from the entire text
        if not chunks and text.strip():
            chunk_metadata = {
                "page": page_num,
                "heading": "Page content",
            }
            chunks.append((text.strip(), chunk_metadata))
        
        return chunks


# Initialize the knowledge base
def initialize_knowledge_base():
    """Initialize the knowledge base from the repository PDF."""
    # Check if knowledge base already exists
    kb = WikiKnowledgeBase()
    if kb.load(KNOWLEDGE_BASE_PATH):
        logger.info("Knowledge base loaded successfully from existing files.")
        return True
    
    # Knowledge base doesn't exist, create it from PDF
    logger.info("Knowledge base not found. Creating from PDF...")
    
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF file not found: {DEFAULT_PDF_PATH}")
        return False
        
    extractor = PDFContentExtractor()
    return extractor.extract_from_pdf(DEFAULT_PDF_PATH)


# Function to search knowledge base (to be used by other actions)
async def search_wiki_knowledge(query: str) -> Tuple[str, Optional[str]]:
    """Search the wiki knowledge base for relevant content."""
    try:
        knowledge_base = WikiKnowledgeBase()
        # Try to load or initialize the knowledge base
        if not knowledge_base.load(KNOWLEDGE_BASE_PATH):
            success = initialize_knowledge_base()
            if not success:
                return None, "Knowledge base could not be initialized"
            
            # Try loading again after initialization
            if not knowledge_base.load(KNOWLEDGE_BASE_PATH):
                return None, "Knowledge base initialization failed"
            
        results = knowledge_base.search(query, k=1)
        if not results:
            return None, "No results found"
            
        content, metadata, score = results[0]
        
        # Only return if confidence is high enough
        if score < 0.2:  # TF-IDF similarity threshold
            return None, "Low confidence results"
            
        # Format response
        response = f"Based on our documentation:\n\n"
        response += f"### {metadata.get('heading', 'Information')}\n\n"
        
        # Add content (remove the heading which we already added)
        content_parts = content.split("\n\n", 1)
        if len(content_parts) > 1:
            response += content_parts[1]
        else:
            response += content
        
        # Add source reference
        response += f"\n\n*Source: PDF page {metadata.get('page', '?')}*"
        
        return response, None
    except Exception as e:
        logger.error(f"Error searching wiki: {str(e)}")
        return None, f"Error: {str(e)}"


class ActionWikiSearch(OptimusAction, action_name="action_wiki_search"):
    """Action for searching wiki knowledge base."""
    
    def __init__(self):
        super().__init__()
        self.knowledge_base = WikiKnowledgeBase()
        self.knowledge_base.load(KNOWLEDGE_BASE_PATH)
        
    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        # Get user query
        query = tracker.latest_message.get('text')
        
        # Try to initialize knowledge base if not already done
        if not self.knowledge_base.initialized:
            success = initialize_knowledge_base()
            if not success:
                dispatcher.utter_message(text="I'm sorry, I couldn't access my knowledge base. Please contact the administrator.")
                return []
        
        # Search knowledge base
        results = self.knowledge_base.search(query, k=3)
        
        if not results:
            dispatcher.utter_message(text="I couldn't find any information about that in my knowledge base.")
            return []
        
        # Format response
        response = self._format_response(results)
        dispatcher.utter_message(text=response)
        
        return []
    
    def _format_response(self, results: List[Tuple[str, Dict[str, Any], float]]) -> str:
        """Format search results into a readable response."""
        best_result = results[0]
        content, metadata, score = best_result
        
        # Start with the heading
        response = f"### {metadata.get('heading', 'Information')}\n\n"
        
        # Add content (remove the heading which we already added)
        content_parts = content.split("\n\n", 1)
        if len(content_parts) > 1:
            response += content_parts[1]
        else:
            response += content
        
        # Add source reference
        response += f"\n\n*Source: PDF page {metadata.get('page', '?')}*"
        
        return response
