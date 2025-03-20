import logging
import json
import os
import re
import difflib
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter

import pdfplumber

from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import EventType, ConversationPaused, UserUtteranceReverted
from optimus.sdk.action_meta import OptimusAction
from optimus.message_components.postback_button.button import PostBackButton, utter_buttons

logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "gcp_t2_l3.pdf")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "wiki_knowledge.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class UniversalKnowledgeBase:
    """Generic knowledge base with advanced fuzzy matching for any content domain."""
    
    def __init__(self):
        self.sections = {}  # {id: {title, content, text_signature}}
        self.word_index = {}  # {normalized_word: {section_ids}}
        self.phrase_index = {}  # {phrase: {section_ids}}
        self.initialized = False
        
    def add_section(self, section_id: str, title: str, content: str):
        """Add a section with advanced indexing."""
        # Store the section
        full_text = f"{title} {content}"
        self.sections[section_id] = {
            "title": title,
            "content": content,
            "text_signature": self._create_text_signature(full_text)
        }
        
        # Index individual words
        words = self._extract_words(full_text)
        for word in words:
            normalized = self._normalize_word(word)
            if len(normalized) >= 3:  # Only index meaningful words
                if normalized not in self.word_index:
                    self.word_index[normalized] = set()
                self.word_index[normalized].add(section_id)
        
        # Index phrases (2-3 word combinations)
        word_list = [w for w in words if len(self._normalize_word(w)) >= 3]
        for i in range(len(word_list) - 1):
            # 2-word phrases
            phrase = f"{word_list[i]} {word_list[i+1]}".lower()
            if phrase not in self.phrase_index:
                self.phrase_index[phrase] = set()
            self.phrase_index[phrase].add(section_id)
            
            # 3-word phrases
            if i < len(word_list) - 2:
                phrase = f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}".lower()
                if phrase not in self.phrase_index:
                    self.phrase_index[phrase] = set()
                self.phrase_index[phrase].add(section_id)
    
    def search(self, query: str) -> List[Tuple[str, float]]:
        """Search using multi-stage matching with fuzzy word matching."""
        if not self.initialized:
            return []
        
        # Extract query words and create signature
        query_words = self._extract_words(query)
        query_signature = self._create_text_signature(query)
        
        # Stage 1: Find potential section matches based on indexed words and phrases
        candidates = set()
        
        # Check for exact phrase matches (highest priority)
        for i in range(len(query_words) - 1):
            if i < len(query_words) - 1:
                phrase = f"{query_words[i]} {query_words[i+1]}".lower()
                if phrase in self.phrase_index:
                    candidates.update(self.phrase_index[phrase])
            
            if i < len(query_words) - 2:
                phrase = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}".lower()
                if phrase in self.phrase_index:
                    candidates.update(self.phrase_index[phrase])
        
        # Add word matches (including fuzzy matching for each word)
        for word in query_words:
            normalized = self._normalize_word(word)
            if len(normalized) >= 3:
                # Exact word match
                if normalized in self.word_index:
                    candidates.update(self.word_index[normalized])
                
                # Fuzzy word matching
                if len(normalized) >= 4:  # Only do fuzzy matching for longer words
                    close_matches = self._find_fuzzy_word_matches(normalized)
                    for match in close_matches:
                        if match in self.word_index:
                            candidates.update(self.word_index[match])
        
        if not candidates:
            # Fallback: include all sections if no matches found
            candidates = set(self.sections.keys())
        
        # Stage 2: Score candidates based on text similarity
        results = []
        for section_id in candidates:
            section = self.sections[section_id]
            
            # Calculate similarity score based on multiple factors
            title_similarity = self._calculate_similarity(query, section["title"]) * 2.0  # Title matches weighted more
            content_similarity = self._calculate_similarity(query, section["content"])
            signature_similarity = self._compare_signatures(query_signature, section["text_signature"])
            
            # Combined score
            score = (title_similarity + content_similarity + signature_similarity) / 3
            
            # Add to results if score is reasonable
            if score > 0.1:
                results.append((section_id, score))
        
        # Sort by score and return best matches
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_content(self, section_id: str) -> str:
        """Get the content of a section."""
        if section_id not in self.sections:
            return ""
        return self.sections[section_id]["content"]
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    def _normalize_word(self, word: str) -> str:
        """Normalize a word for indexing and matching."""
        return word.lower().strip()
    
    def _create_text_signature(self, text: str) -> Dict[str, int]:
        """Create a frequency-based signature of the text."""
        words = self._extract_words(text)
        signature = Counter([self._normalize_word(w) for w in words if len(self._normalize_word(w)) >= 3])
        return dict(signature)
    
    def _compare_signatures(self, sig1: Dict[str, int], sig2: Dict[str, int]) -> float:
        """Compare two text signatures for similarity."""
        if not sig1 or not sig2:
            return 0.0
            
        # Calculate overlap coefficient of shared words
        common_words = set(sig1.keys()) & set(sig2.keys())
        if not common_words:
            return 0.0
            
        total1 = sum(sig1.values())
        total2 = sum(sig2.values())
        if total1 == 0 or total2 == 0:
            return 0.0
            
        # Calculate weighted similarity based on word frequencies
        similarity = 0.0
        for word in common_words:
            similarity += min(sig1.get(word, 0), sig2.get(word, 0))
            
        # Normalize
        return similarity / max(total1, total2)
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """Calculate simple similarity between query and text."""
        query = query.lower()
        text = text.lower()
        
        # Simple similarity using difflib
        return difflib.SequenceMatcher(None, query, text).ratio()
    
    def _find_fuzzy_word_matches(self, word: str) -> List[str]:
        """Find fuzzy matches for a word in the index."""
        candidates = []
        
        # Check for words with the same beginning
        for indexed_word in self.word_index.keys():
            # Words starting with the same characters
            if indexed_word.startswith(word[:min(3, len(word))]):
                candidates.append(indexed_word)
                
            # Words with high similarity
            if difflib.SequenceMatcher(None, word, indexed_word).ratio() > 0.8:
                candidates.append(indexed_word)
                
            # Edit distance for typos (basic implementation)
            if len(word) > 4 and abs(len(word) - len(indexed_word)) <= 2:
                if sum(1 for a, b in zip(word, indexed_word) if a != b) <= 2:
                    candidates.append(indexed_word)
        
        return candidates
    
    def save(self, file_path: str) -> bool:
        """Save knowledge base to file."""
        try:
            # Convert set objects to lists for JSON serialization
            serializable_word_index = {k: list(v) for k, v in self.word_index.items()}
            serializable_phrase_index = {k: list(v) for k, v in self.phrase_index.items()}
            
            data = {
                "sections": self.sections,
                "word_index": serializable_word_index,
                "phrase_index": serializable_phrase_index
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f)
                
            self.initialized = True
            logger.info(f"Saved knowledge base with {len(self.sections)} sections")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def load(self, file_path: str) -> bool:
        """Load knowledge base from file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Knowledge base file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.sections = data["sections"]
            
            # Convert lists back to sets
            self.word_index = {k: set(v) for k, v in data["word_index"].items()}
            self.phrase_index = {k: set(v) for k, v in data["phrase_index"].items()}
            
            self.initialized = len(self.sections) > 0
            logger.info(f"Loaded knowledge base with {len(self.sections)} sections")
            return self.initialized
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False


def extract_pdf_content(pdf_path: str) -> UniversalKnowledgeBase:
    """Extract content from PDF with improved section detection."""
    kb = UniversalKnowledgeBase()
    
    try:
        logger.info(f"Extracting content from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return kb
        
        # Extract all text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n\nPage {i+1}:\n\n{text}"
        
        # Split text into sections using multiple heuristics
        sections = []
        
        # Split by lines and look for headings
        lines = full_text.split('\n')
        current_title = ""
        current_content = []
        section_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and page markers
            if not line or line.startswith('Page '):
                continue
            
            # Check for potential headings - multiple patterns
            is_heading = False
            
            # Pattern 1: Markdown-style heading
            if line.startswith('#') or line.startswith('==='):
                is_heading = True
            
            # Pattern 2: Short line with special formatting (all caps, etc.)
            elif len(line) < 60 and (line.isupper() or line[0].isupper()) and i < len(lines) - 1:
                if not lines[i+1].strip() or not lines[i+1][0].isupper():
                    is_heading = True
            
            # Pattern 3: Numbered headings
            elif re.match(r'^\d+\.\s+[A-Z]', line):
                is_heading = True
            
            if is_heading:
                # Save previous section if it exists
                if current_title and current_content:
                    content = '\n'.join(current_content)
                    sections.append((current_title, content))
                    section_count += 1
                
                # Clean heading
                current_title = line.strip('# =')
                current_content = []
            else:
                # Add to current content
                current_content.append(line)
        
        # Add the last section
        if current_title and current_content:
            content = '\n'.join(current_content)
            sections.append((current_title, content))
        
        # Add sections to knowledge base
        for i, (title, content) in enumerate(sections):
            kb.add_section(f"section_{i}", title, content)
        
        logger.info(f"Extracted {len(sections)} sections from PDF")
        
        # If no sections found, use entire text
        if not sections:
            kb.add_section("section_0", "PDF Content", full_text)
        
        kb.initialized = True
        
    except Exception as e:
        logger.error(f"Error extracting PDF content: {str(e)}")
    
    return kb


def initialize_knowledge_base(force_refresh=False) -> UniversalKnowledgeBase:
    """Initialize or refresh the knowledge base."""
    kb = UniversalKnowledgeBase()
    
    # Try to load existing knowledge base
    if not force_refresh and os.path.exists(KNOWLEDGE_BASE_PATH):
        if kb.load(KNOWLEDGE_BASE_PATH):
            return kb
    
    # Check if PDF exists
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF file not found: {DEFAULT_PDF_PATH}")
        return kb
    
    # Extract content from PDF
    logger.info("Creating new knowledge base from PDF")
    kb = extract_pdf_content(DEFAULT_PDF_PATH)
    
    # Save knowledge base
    if kb.initialized:
        kb.save(KNOWLEDGE_BASE_PATH)
    
    return kb


async def search_knowledge_base(query: str) -> Tuple[str, Optional[str]]:
    """Search with intelligent content matching."""
    try:
        # Get knowledge base
        kb = initialize_knowledge_base()
        
        if not kb.initialized:
            logger.error("Knowledge base initialization failed")
            return None, "Knowledge base not available"
        
        # Advanced search
        results = kb.search(query)
        
        if not results:
            logger.info(f"No results found for query: {query}")
            return None, "No matching information found"
        
        # Get the best match
        section_id, score = results[0]
        
        logger.info(f"Query: '{query}' matched with section_id: {section_id}, score: {score:.3f}")
        
        # Get section content
        content = kb.get_content(section_id)
        
        # If score is too low, don't trust the result
        if score < 0.15:
            logger.info(f"Match score too low ({score:.3f}) for query: {query}")
            return None, "No confident matches found"
        
        # Return the content directly
        return content.strip(), None
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return None, f"Error: {str(e)}"


class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):
    """Enhanced fallback with universal knowledge matching."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        query = tracker.latest_message.get('text')
        logger.info(f'### User query "{query}" falling back ###')
        
        # Try to find information in knowledge base
        content, error = await search_knowledge_base(query)
        
        if content:
            # Found relevant information - return just the content
            logger.info(f"Answering query with content: {content[:100]}...")
            dispatcher.utter_message(text=content)
            return []
        
        # Log the error if any
        if error:
            logger.warning(f"Knowledge base search error: {error}")
        
        # Default fallback response
        draft_email_button = PostBackButton.with_intent(title="Draft email", intent="gcp_feedback")
        utter_buttons(
            dispatcher=dispatcher, 
            text="Sorry, I am not trained for this. Please share query details if it is frequently required and we will set it up. Please click \"Draft email \" button, to share query details if required.",
            buttons=[draft_email_button]
        )
        return [ConversationPaused(), UserUtteranceReverted()]
