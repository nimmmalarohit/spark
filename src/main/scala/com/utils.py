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

class PrecisionKnowledgeBase:
    """Knowledge base optimized for precise content matching."""
    
    def __init__(self):
        self.sections = {}  # {id: {title, content}}
        self.text_chunks = {}  # {id: {chunk_text, section_id, position}}
        self.phrases = {}  # {normalized_phrase: [chunk_ids]}
        self.keywords = {}  # {keyword: [section_ids]}
        self.initialized = False
        
    def add_section(self, section_id: str, title: str, content: str):
        """Add a section and index it for precise matching."""
        self.sections[section_id] = {
            "title": title.strip(),
            "content": content.strip()
        }
        
        # Index section keywords
        words = self._extract_words(f"{title} {content}")
        for word in words:
            if len(word) >= 3:
                word = word.lower()
                if word not in self.keywords:
                    self.keywords[word] = set()
                self.keywords[word].add(section_id)
        
        # Split content into meaningful chunks
        self._index_content_chunks(section_id, title, content)
    
    def _index_content_chunks(self, section_id: str, title: str, content: str):
        """Break content into meaningful chunks for precise matching."""
        # First try to split by lines
        lines = content.split('\n')
        
        chunk_id = 0
        current_chunk = []
        current_position = 0
        
        # Process content line by line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a bullet point or numbered item
            is_list_item = bool(re.match(r'^\s*[•o\-*]\s+', line) or 
                                re.match(r'^\s*\d+\.\s+', line))
            
            # Start a new chunk for list items or if current chunk is large
            if is_list_item or len(' '.join(current_chunk)) > 200:
                if current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_key = f"{section_id}_chunk_{chunk_id}"
                    self.text_chunks[chunk_key] = {
                        "chunk_text": chunk_text,
                        "section_id": section_id,
                        "position": current_position
                    }
                    
                    # Index phrases in this chunk
                    self._index_phrases(chunk_key, chunk_text)
                    
                    chunk_id += 1
                    current_chunk = []
                
                current_position += 1
            
            # Add line to current chunk
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_key = f"{section_id}_chunk_{chunk_id}"
            self.text_chunks[chunk_key] = {
                "chunk_text": chunk_text,
                "section_id": section_id,
                "position": current_position
            }
            
            # Index phrases in this chunk
            self._index_phrases(chunk_key, chunk_text)
    
    def _index_phrases(self, chunk_id: str, text: str):
        """Index meaningful phrases in text."""
        # Create n-grams (2-4 words) from text
        words = self._extract_words(text)
        
        # Generate key phrases (2-4 word phrases)
        for n in range(2, 5):  # 2, 3, and 4-word phrases
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n]).lower()
                    if len(phrase) >= 5:  # Only meaningful phrases
                        if phrase not in self.phrases:
                            self.phrases[phrase] = []
                        self.phrases[phrase].append(chunk_id)
    
    def search(self, query: str) -> List[Tuple[str, float]]:
        """Search with multi-stage precision matching."""
        if not self.initialized:
            return []
        
        # Normalize query
        query = query.lower()
        
        # Stage 1: Try direct content matching first
        direct_matches = self._find_direct_matches(query)
        if direct_matches:
            logger.info(f"Found direct matches for query: {query}")
            return direct_matches
        
        # Stage 2: Try phrase matching
        phrase_matches = self._find_phrase_matches(query)
        if phrase_matches:
            logger.info(f"Found phrase matches for query: {query}")
            return phrase_matches
        
        # Stage 3: Fall back to keyword matching
        keyword_matches = self._find_keyword_matches(query)
        logger.info(f"Found keyword matches for query: {query}")
        return keyword_matches
    
    def _find_direct_matches(self, query: str) -> List[Tuple[str, float]]:
        """Find direct content matches for near-exact queries."""
        results = []
        
        # Check for content that contains the exact query
        for section_id, section in self.sections.items():
            title = section["title"].lower()
            content = section["content"].lower()
            
            # Calculate exact match score
            title_contains = query in title
            content_contains = query in content
            
            # If query appears exactly in content, it's a very good match
            if title_contains or content_contains:
                # Calculate more precise match score based on how much of the query is covered
                if title_contains:
                    score = 0.9  # High score for title matches
                else:
                    # Find the context of the match (surrounding text)
                    match_pos = content.find(query)
                    match_context = content[max(0, match_pos-50):min(len(content), match_pos+len(query)+50)]
                    
                    # Score based on how much of the text is the query
                    coverage = len(query) / len(match_context)
                    score = 0.7 + (coverage * 0.2)  # 0.7-0.9 based on coverage
                
                results.append((section_id, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _find_phrase_matches(self, query: str) -> List[Tuple[str, float]]:
        """Find matches based on phrases in the query."""
        # Extract phrases from query
        query_words = self._extract_words(query)
        
        # Identify query chunks that might match indexed phrases
        chunk_matches = Counter()
        query_phrases = []
        
        # Generate phrases from query (2-4 word phrases)
        for n in range(2, 5):  # 2, 3, and 4-word phrases
            if len(query_words) >= n:
                for i in range(len(query_words) - n + 1):
                    phrase = ' '.join(query_words[i:i+n]).lower()
                    if len(phrase) >= 5:  # Only meaningful phrases
                        query_phrases.append(phrase)
                        if phrase in self.phrases:
                            for chunk_id in self.phrases[phrase]:
                                chunk_matches[chunk_id] += 1
        
        # Score chunks by phrase matches
        chunk_scores = {}
        for chunk_id, match_count in chunk_matches.items():
            coverage = match_count / max(1, len(query_phrases))
            chunk_scores[chunk_id] = coverage
        
        # Get sections from top chunks
        section_scores = {}
        for chunk_id, score in chunk_scores.items():
            if score >= 0.2:  # Only reasonable matches
                chunk_info = self.text_chunks[chunk_id]
                section_id = chunk_info["section_id"]
                
                # Take best score for each section
                if section_id not in section_scores or score > section_scores[section_id]:
                    section_scores[section_id] = score
        
        # Return scored sections
        results = [(section_id, score) for section_id, score in section_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _find_keyword_matches(self, query: str) -> List[Tuple[str, float]]:
        """Find matches based on keywords in the query."""
        query_words = self._extract_words(query)
        
        # Count keyword matches per section
        section_matches = Counter()
        for word in query_words:
            if len(word) >= 3:
                word = word.lower()
                if word in self.keywords:
                    for section_id in self.keywords[word]:
                        section_matches[section_id] += 1
        
        # Score sections by keyword matches
        results = []
        for section_id, match_count in section_matches.items():
            # Normalize score by query length
            score = match_count / max(1, len(query_words))
            if score >= 0.2:  # Only include reasonable matches
                results.append((section_id, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_section_content(self, section_id: str) -> str:
        """Get the content of a section."""
        if section_id not in self.sections:
            return ""
        return self.sections[section_id]["content"]
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        return [w for w in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) 
                if len(w) >= 3 and w not in {'the', 'and', 'for', 'are', 'with'}]
    
    def save(self, file_path: str) -> bool:
        """Save knowledge base to file."""
        try:
            # Convert sets to lists for JSON serialization
            serializable_keywords = {k: list(v) for k, v in self.keywords.items()}
            
            data = {
                "sections": self.sections,
                "text_chunks": self.text_chunks,
                "phrases": self.phrases,
                "keywords": serializable_keywords
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
            self.text_chunks = data["text_chunks"]
            self.phrases = data["phrases"]
            
            # Convert lists back to sets
            self.keywords = {k: set(v) for k, v in data["keywords"].items()}
            
            self.initialized = len(self.sections) > 0
            logger.info(f"Loaded knowledge base with {len(self.sections)} sections")
            return self.initialized
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False


def extract_pdf_content(pdf_path: str) -> PrecisionKnowledgeBase:
    """Extract content from PDF with advanced section detection."""
    kb = PrecisionKnowledgeBase()
    
    try:
        logger.info(f"Extracting content from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return kb
        
        # Extract all text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            # First pass to get all content
            all_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n\n"
            
            # Save raw text for debugging
            debug_file = os.path.join(DATA_DIR, "raw_pdf_text.txt")
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(all_text)
        
        # Identify section boundaries
        sections = []
        lines = all_text.split('\n')
        
        # First pass: identify potential section headers
        potential_headers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check various heading patterns
            is_heading = False
            
            # Pattern 1: Markdown-style heading
            if line.startswith('#'):
                is_heading = True
                potential_headers.append((i, line.lstrip('#').strip()))
            
            # Pattern 2: Short line that looks like a title
            elif len(line) < 60 and line[0].isupper() and i < len(lines) - 1:
                next_line = lines[i+1].strip() if i+1 < len(lines) else ""
                if not next_line or next_line[0].isspace() or next_line[0].islower():
                    is_heading = True
                    potential_headers.append((i, line))
        
        # If no headers found, create a single section with all content
        if not potential_headers:
            kb.add_section("section_0", "PDF Content", all_text)
            return kb
        
        # Second pass: extract sections based on headers
        for j, (header_line, title) in enumerate(potential_headers):
            # Calculate section boundaries
            start_line = header_line + 1
            end_line = len(lines)
            if j < len(potential_headers) - 1:
                end_line = potential_headers[j+1][0]
            
            # Extract section content
            section_lines = lines[start_line:end_line]
            content = "\n".join(line.strip() for line in section_lines if line.strip())
            
            # Add section to knowledge base
            kb.add_section(f"section_{j}", title, content)
            sections.append((title, content))
        
        # Save sections for debugging
        debug_sections = os.path.join(DATA_DIR, "extracted_sections.txt")
        with open(debug_sections, "w", encoding="utf-8") as f:
            for title, content in sections:
                f.write(f"==== {title} ====\n\n{content}\n\n{'='*40}\n\n")
        
        kb.initialized = True
        logger.info(f"Extracted {len(sections)} sections from PDF")
        
    except Exception as e:
        logger.error(f"Error extracting PDF content: {str(e)}")
    
    return kb


def initialize_knowledge_base(force_refresh=False) -> PrecisionKnowledgeBase:
    """Initialize or refresh the knowledge base."""
    kb = PrecisionKnowledgeBase()
    
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
    """Search with precision content matching."""
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
        content = kb.get_section_content(section_id)
        
        # If score is too low, don't trust the result
        if score < 0.15:
            logger.info(f"Match score too low ({score:.3f}) for query: {query}")
            return None, "No confident matches found"
        
        # Return the content directly without any category headings
        return content.strip(), None
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return None, f"Error: {str(e)}"


class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):
    """Enhanced fallback with precision content matching."""

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
