import logging
import json
import os
import re
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
    """Vector database for semantic content matching without hardcoded patterns."""
    
    def __init__(self):
        """Initialize the knowledge base with semantic matching capabilities."""
        # Use bigrams and character n-grams for better semantic matching
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            analyzer='word',
            min_df=1,
            max_df=0.95,
            sublinear_tf=True  # Apply sublinear tf scaling for better matching
        )
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
            # Reinitialize vectorizer with all documents to ensure consistency
            all_docs = self.documents + documents
            all_meta = self.metadata + metadata
            
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                analyzer='word',
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
            self.vectors = self.vectorizer.fit_transform(all_docs)
            self.documents = all_docs
            self.metadata = all_meta
        else:
            # First time adding documents
            self.vectors = self.vectorizer.fit_transform(documents)
            self.documents = documents
            self.metadata = metadata
            self.initialized = True
        
        # Update timestamps
        self.creation_time = datetime.now().timestamp()
        if os.path.exists(DEFAULT_PDF_PATH):
            self.pdf_last_modified = os.path.getmtime(DEFAULT_PDF_PATH)
        
        logger.info(f"Added {len(documents)} documents to knowledge base. Total: {len(self.documents)}")
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for semantically similar documents using improved matching."""
        if not self.initialized or self.vectors is None:
            logger.warning("Knowledge base not initialized yet.")
            return []
        
        # Preprocess query to improve matching
        query = self._normalize_query(query)
        
        # Create query vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate semantic similarity
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
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better semantic matching without hardcoding."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation except meaningful symbols
        query = re.sub(r'[^\w\s-.]', ' ', query)
        
        # Remove filler words that don't affect meaning
        filler_words = {'is', 'the', 'a', 'an', 'in', 'at', 'on', 'for', 'of', 'used', 'what'}
        tokens = query.split()
        tokens = [t for t in tokens if t not in filler_words]
        
        # Add focus on important technical terms if present
        meaningful_terms = []
        for token in tokens:
            if len(token) > 3:  # Likely a meaningful term
                meaningful_terms.append(token)
        
        # If we have meaningful terms, prioritize them
        if meaningful_terms:
            normalized_query = ' '.join(tokens + meaningful_terms)
        else:
            normalized_query = ' '.join(tokens)
            
        return normalized_query
    
    def save(self, file_path: str):
        """Save knowledge base to disk."""
        # Since sklearn models aren't directly JSON serializable,
        # we'll save the vocabulary and trained features separately
        
        try:
            # Save core data
            with open(f"{file_path}.json", "w") as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "creation_time": self.creation_time,
                    "pdf_last_modified": self.pdf_last_modified,
                    # Save vectorizer components that we need
                    "vocabulary": self.vectorizer.vocabulary_ if hasattr(self.vectorizer, 'vocabulary_') else {},
                    "idf": self.vectorizer.idf_.tolist() if hasattr(self.vectorizer, 'idf_') else []
                }, f)
            
            # Save vectors as numpy array
            if self.vectors is not None:
                np.save(f"{file_path}_vectors.npy", self.vectors.toarray())
                
            logger.info(f"Saved knowledge base to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
        
    def load(self, file_path: str) -> bool:
        """Load knowledge base from disk."""
        try:
            if os.path.exists(f"{file_path}.json"):
                # Load main data
                with open(f"{file_path}.json", "r") as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    self.metadata = data["metadata"]
                    self.creation_time = data.get("creation_time")
                    self.pdf_last_modified = data.get("pdf_last_modified")
                
                # Initialize vectorizer with saved vocabulary
                self.vectorizer = TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),
                    analyzer='word',
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True
                )
                if "vocabulary" in data and data["vocabulary"]:
                    self.vectorizer.vocabulary_ = data["vocabulary"]
                
                if "idf" in data and data["idf"]:
                    self.vectorizer.idf_ = np.array(data["idf"])
                
                # Load vectors
                if os.path.exists(f"{file_path}_vectors.npy"):
                    vectors_array = np.load(f"{file_path}_vectors.npy")
                    self.vectors = vectors_array
                    
                self.initialized = True
                logger.info(f"Loaded knowledge base from {file_path} with {len(self.documents)} documents")
                
                # Check if PDF has been modified since knowledge base was created
                if os.path.exists(DEFAULT_PDF_PATH):
                    current_mtime = os.path.getmtime(DEFAULT_PDF_PATH)
                    if not self.pdf_last_modified or current_mtime > self.pdf_last_modified:
                        logger.info("PDF has been modified. Knowledge base needs refresh.")
                        return False
                return True
            else:
                logger.warning(f"Knowledge base file {file_path}.json not found")
                return False
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def needs_update(self) -> bool:
        """Check if knowledge base needs to be updated based on PDF modification time."""
        if not self.pdf_last_modified or not os.path.exists(DEFAULT_PDF_PATH):
            return True
            
        current_mtime = os.path.getmtime(DEFAULT_PDF_PATH)
        return current_mtime > self.pdf_last_modified


class PDFContentExtractor:
    """Extracts content from PDFs with robust semantic grouping."""
    
    def __init__(self):
        """Initialize the PDF content extractor."""
        self.knowledge_base = WikiKnowledgeBase()
        
    def extract_from_pdf(self, pdf_path: str) -> bool:
        """Extract content from a PDF with better semantic grouping."""
        logger.info(f"Extracting content from PDF: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # First extract the full text of the PDF to get a global view
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n\n"
                
                # Identify major section boundaries
                sections = self._identify_sections(full_text)
                
                # Process each section into smaller semantic units
                all_chunks = []
                for section_title, section_text, section_type in sections:
                    chunks = self._process_section(section_title, section_text, section_type)
                    all_chunks.extend(chunks)
                
                # Add enhanced semantic units to knowledge base
                documents = []
                metadata_list = []
                
                for text, meta in all_chunks:
                    documents.append(text)
                    metadata_list.append({
                        "source": pdf_path,
                        "section": meta.get("section", ""),
                        "heading": meta.get("heading", ""),
                        "content_type": meta.get("content_type", "text")
                    })
                
                # Add to knowledge base
                self.knowledge_base.add_documents(documents, metadata_list)
                
                # Save knowledge base
                self.knowledge_base.save(KNOWLEDGE_BASE_PATH)
                
                logger.info(f"Extracted {len(documents)} semantic units from PDF")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {str(e)}")
            return False
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str, str]]:
        """Identify major sections in the document for better semantic grouping."""
        sections = []
        
        # Split by lines to find headings
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify section headings
            is_heading = False
            if line.startswith('#'):
                is_heading = True
            elif len(line) < 80 and (line.isupper() or line[0].isupper() and not line.endswith('.') and len(line.split()) <= 7):
                # Likely a heading - short text that's all caps or starts with uppercase and isn't a sentence
                is_heading = True
            
            if is_heading:
                # Save previous section if exists
                if current_section and current_content:
                    sections.append((
                        current_section, 
                        '\n'.join(current_content),
                        'section' 
                    ))
                
                # Start new section
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                # Add to current section
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections.append((
                current_section,
                '\n'.join(current_content),
                'section'
            ))
        
        # If no sections found, create one generic section
        if not sections and text.strip():
            sections.append(("Document Content", text.strip(), "content"))
        
        return sections
    
    def _process_section(self, section_title: str, section_text: str, section_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a section into smaller semantic units."""
        chunks = []
        
        # Don't break up short sections - keep as a single semantic unit
        if len(section_text.split()) < 100:
            full_text = f"{section_title}\n\n{section_text}"
            chunks.append((full_text, {
                "section": section_title,
                "heading": section_title,
                "content_type": section_type
            }))
            return chunks
        
        # For longer sections, try to break into subsections
        subsections = []
        current_subsection = None
        current_content = []
        
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this might be a subsection heading
            is_heading = False
            if len(line) < 80 and not line.endswith('.') and len(line.split()) <= 10:
                # Look for title-case or all caps patterns
                if line.istitle() or line.isupper() or all(w[0].isupper() for w in line.split() if w):
                    is_heading = True
            
            if is_heading:
                # Save previous subsection
                if current_subsection and current_content:
                    subsections.append((current_subsection, '\n'.join(current_content)))
                
                # Start new subsection
                current_subsection = line
                current_content = []
            else:
                # Add to current subsection
                current_content.append(line)
        
        # Add final subsection
        if current_subsection and current_content:
            subsections.append((current_subsection, '\n'.join(current_content)))
        
        # Create semantic chunks from subsections
        if subsections:
            for subsection_title, subsection_content in subsections:
                # Combine section and subsection for better context
                full_text = f"{section_title}: {subsection_title}\n\n{subsection_content}"
                chunks.append((full_text, {
                    "section": section_title,
                    "heading": subsection_title,
                    "content_type": "subsection"
                }))
        else:
            # No subsections found, use the whole section
            full_text = f"{section_title}\n\n{section_text}"
            chunks.append((full_text, {
                "section": section_title,
                "heading": section_title,
                "content_type": section_type
            }))
        
        return chunks


# Initialize or refresh the knowledge base
def initialize_knowledge_base(force_refresh=False) -> bool:
    """Initialize the knowledge base from the repository PDF."""
    # Check if knowledge base exists and is up to date
    kb = WikiKnowledgeBase()
    
    if not force_refresh and kb.load(KNOWLEDGE_BASE_PATH) and not kb.needs_update():
        logger.info("Knowledge base is up to date.")
        return True
    
    # Knowledge base doesn't exist or needs refresh
    logger.info("Creating or refreshing knowledge base from PDF...")
    
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF file not found: {DEFAULT_PDF_PATH}")
        return False
    
    # Clean up old files if they exist
    for ext in ['.json', '_vectors.npy']:
        old_file = f"{KNOWLEDGE_BASE_PATH}{ext}"
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                logger.info(f"Removed old knowledge base file: {old_file}")
            except Exception as e:
                logger.warning(f"Could not remove old file {old_file}: {e}")
    
    # Create new knowledge base
    extractor = PDFContentExtractor()
    return extractor.extract_from_pdf(DEFAULT_PDF_PATH)


# Search function that leverages Rasa NLU information
async def search_wiki_knowledge(tracker: Tracker) -> Tuple[str, Optional[str]]:
    """Search the knowledge base with better semantic understanding."""
    try:
        # Check if knowledge base needs updating
        kb = WikiKnowledgeBase()
        if not kb.load(KNOWLEDGE_BASE_PATH) or kb.needs_update():
            logger.info("Knowledge base needs refresh. Updating...")
            initialize_knowledge_base(force_refresh=True)
            # Load again after refresh
            if not kb.load(KNOWLEDGE_BASE_PATH):
                return None, "Knowledge base could not be initialized"
        
        # Get query from user message
        query = tracker.latest_message.get('text')
        
        # Get entities from Rasa NLU to enhance query
        entities = tracker.latest_message.get("entities", [])
        entity_values = [e.get("value") for e in entities if e.get("value")]
        
        # Create an enhanced query if we have entities
        if entity_values:
            enhanced_query = f"{query} {' '.join(entity_values)}"
            # Try enhanced query first
            results = kb.search(enhanced_query, k=3)
            
            # If no good results, fall back to original query
            if not results or results[0][2] < 0.2:  # Low confidence
                results = kb.search(query, k=3)
        else:
            # Just use original query
            results = kb.search(query, k=3)
        
        # Handle no results case
        if not results:
            return None, "No relevant information found"
        
        # Get the best match
        content, metadata, score = results[0]
        
        # Skip results with very low confidence
        if score < 0.15:
            return None, "No confident matches found"
        
        # Extract just the actual content part, not the heading
        # This ensures we're returning just the answer, not the category
        if '\n\n' in content:
            # Split into heading and content
            parts = content.split('\n\n', 1)
            # Check if the first part looks like a heading
            if len(parts[0].split()) <= 10:
                return parts[1].strip(), None
        
        # If no clear heading/content split, return the whole content
        return content.strip(), None
        
    except Exception as e:
        logger.error(f"Error searching wiki: {str(e)}")
        return None, f"Error in knowledge search: {str(e)}"


class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):
    """Enhanced fallback action with semantic knowledge search."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        latest_message = tracker.latest_message.get('text')
        logger.info(f'### User query "{latest_message}" falling back ###')
        
        # Try to find relevant information in knowledge base
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
