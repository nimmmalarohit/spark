import logging
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional

import pdfplumber

from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import EventType, ConversationPaused, UserUtteranceReverted
from optimus.sdk.action_meta import OptimusAction
from optimus.message_components.postback_button.button import PostBackButton, utter_buttons

logger = logging.getLogger(__name__)

# Define paths - use absolute paths to avoid any confusion
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "gcp_security_guidelines.pdf")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "wiki_knowledge.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class SimpleKnowledgeBase:
    """Simple, robust knowledge base without complex dependencies."""
    
    def __init__(self):
        self.sections = {}  # {section_heading: section_content}
        self.keywords = {}  # {keyword: [section_headings]}
        self.initialized = False
        
    def add_section(self, heading: str, content: str):
        """Add a section with its content."""
        self.sections[heading] = content
        
        # Extract keywords from heading and content
        words = re.findall(r'\b\w+\b', heading.lower())
        for word in words:
            if len(word) > 3:  # Only use meaningful words
                if word not in self.keywords:
                    self.keywords[word] = []
                if heading not in self.keywords[word]:
                    self.keywords[word].append(heading)
        
    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """Search for matching sections using a simple keyword approach."""
        if not self.initialized:
            return []
        
        # Extract keywords from query
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_words = [w for w in query_words if len(w) > 3]  # Only meaningful words
        
        # Count keyword matches for each section
        section_scores = {}
        for word in query_words:
            if word in self.keywords:
                for heading in self.keywords[word]:
                    if heading not in section_scores:
                        section_scores[heading] = 0
                    section_scores[heading] += 1
        
        # Create results with scores
        results = []
        for heading, score in section_scores.items():
            # Calculate a normalized score
            normalized_score = score / max(len(query_words), 1)
            if normalized_score > 0.2:  # Only include reasonable matches
                results.append((heading, self.sections[heading], normalized_score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def save(self, file_path: str) -> bool:
        """Save knowledge base to disk."""
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    'sections': self.sections,
                    'keywords': self.keywords
                }, f, indent=2)
            logger.info(f"Saved knowledge base to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def load(self, file_path: str) -> bool:
        """Load knowledge base from disk."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Knowledge base file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.sections = data.get('sections', {})
                self.keywords = data.get('keywords', {})
                
            self.initialized = len(self.sections) > 0
            logger.info(f"Loaded knowledge base with {len(self.sections)} sections")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False


def extract_knowledge_from_pdf(pdf_path: str) -> SimpleKnowledgeBase:
    """Extract knowledge from PDF using a simple, robust approach."""
    kb = SimpleKnowledgeBase()
    
    try:
        logger.info(f"Extracting knowledge from {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return kb
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
            
            # Split text into sections based on headings
            sections = []
            
            # First try to find markdown-style headings
            heading_pattern = r'(?:^|\n)#+\s+(.*?)(?:\n|$)'
            headings = re.findall(heading_pattern, full_text)
            
            if headings:
                # Split by markdown headings
                chunks = re.split(heading_pattern, full_text)[1:]  # Skip first which is before any heading
                
                for i, heading in enumerate(headings):
                    if i < len(chunks):
                        sections.append((heading.strip(), chunks[i].strip()))
            else:
                # Try to find headings based on line length and characteristics
                lines = full_text.split('\n')
                i = 0
                current_heading = None
                current_content = []
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    is_heading = False
                    if line and len(line) < 80:
                        # Short line that's not part of a paragraph
                        next_line = lines[i+1].strip() if i+1 < len(lines) else ""
                        
                        # Check if followed by blank or if it looks like a title
                        if not next_line or (line.istitle() and not line.endswith('.')):
                            is_heading = True
                    
                    if is_heading:
                        # Save previous section
                        if current_heading and current_content:
                            sections.append((current_heading, '\n'.join(current_content)))
                        
                        # Start new section
                        current_heading = line
                        current_content = []
                    elif line:
                        # Add to current content
                        current_content.append(line)
                    
                    i += 1
                
                # Add final section
                if current_heading and current_content:
                    sections.append((current_heading, '\n'.join(current_content)))
            
            # Add sections to knowledge base
            for heading, content in sections:
                kb.add_section(heading, content)
            
            kb.initialized = True
            logger.info(f"Extracted {len(sections)} sections from PDF")
    
    except Exception as e:
        logger.error(f"Error extracting knowledge from PDF: {e}")
    
    return kb


def initialize_knowledge_base(force_refresh=False) -> SimpleKnowledgeBase:
    """Initialize or refresh the knowledge base."""
    kb = SimpleKnowledgeBase()
    
    # Try to load existing knowledge base
    if not force_refresh and os.path.exists(KNOWLEDGE_BASE_PATH):
        if kb.load(KNOWLEDGE_BASE_PATH):
            logger.info("Successfully loaded existing knowledge base")
            return kb
    
    # Need to create or refresh knowledge base
    logger.info("Creating new knowledge base from PDF")
    
    # Verify PDF exists
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF file not found: {DEFAULT_PDF_PATH}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Data directory: {DATA_DIR}")
        return kb
    
    # Extract knowledge from PDF
    kb = extract_knowledge_from_pdf(DEFAULT_PDF_PATH)
    
    # Save knowledge base if successful
    if kb.initialized:
        kb.save(KNOWLEDGE_BASE_PATH)
    
    return kb


async def search_knowledge_base(query: str) -> Tuple[str, Optional[str]]:
    """Search the knowledge base using a simple keyword approach."""
    try:
        # Get knowledge base
        kb = initialize_knowledge_base()
        
        if not kb.initialized:
            logger.error("Knowledge base initialization failed")
            return None, "Knowledge base could not be initialized"
        
        # Search for matching sections
        results = kb.search(query)
        
        if not results:
            return None, "No matching information found"
        
        # Get best match
        heading, content, score = results[0]
        
        # Format response - just return the content
        logger.info(f"Found match for '{query}': {heading} (score: {score})")
        return content.strip(), None
    
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return None, f"Error: {str(e)}"


class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):
    """Simple, robust fallback action with knowledge base search."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        query = tracker.latest_message.get('text')
        logger.info(f'### User query "{query}" falling back ###')
        
        # Try to find information in knowledge base
        content, error = await search_knowledge_base(query)
        
        if content:
            # Found relevant information - respond with content
            dispatcher.utter_message(text=content)
            return []
        
        # Default fallback response
        draft_email_button = PostBackButton.with_intent(title="Draft email", intent="gcp_feedback")
        utter_buttons(
            dispatcher=dispatcher, 
            text="Sorry, I am not trained for this. Please share query details if it is frequently required and we will set it up. Please click \"Draft email \" button, to share query details if required.",
            buttons=[draft_email_button]
        )
        return [ConversationPaused(), UserUtteranceReverted()]
