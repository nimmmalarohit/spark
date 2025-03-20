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
        self.contents = {}  # {section_id: content}
        self.keywords = {}  # {keyword: [section_ids]}
        self.initialized = False
        
    def add_content(self, section_id: str, title: str, content: str):
        """Add a section with its content."""
        # Store the full content including title
        full_content = f"{title}\n\n{content}" if content else title
        self.contents[section_id] = full_content
        
        # Extract keywords from title and content
        all_text = f"{title} {content}".lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        for word in words:
            if len(word) > 3:  # Only use meaningful words
                if word not in self.keywords:
                    self.keywords[word] = []
                if section_id not in self.keywords[word]:
                    self.keywords[word].append(section_id)
        
    def search(self, query: str) -> List[Tuple[str, float]]:
        """Search for matching content using keywords."""
        if not self.initialized:
            return []
        
        # Extract keywords from query
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_words = [w for w in query_words if len(w) > 3]  # Only meaningful words
        
        # Count keyword matches for each section
        section_scores = {}
        for word in query_words:
            if word in self.keywords:
                for section_id in self.keywords[word]:
                    if section_id not in section_scores:
                        section_scores[section_id] = 0
                    section_scores[section_id] += 1
        
        # Create results with scores
        results = []
        for section_id, score in section_scores.items():
            # Calculate a normalized score
            normalized_score = score / max(len(query_words), 1)
            if normalized_score > 0.2:  # Only include reasonable matches
                results.append((self.contents[section_id], normalized_score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def save(self, file_path: str) -> bool:
        """Save knowledge base to disk."""
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    'contents': self.contents,
                    'keywords': self.keywords
                }, f, indent=2)
            logger.info(f"Saved knowledge base to {file_path}")
            self.initialized = True
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
                self.contents = data.get('contents', {})
                self.keywords = data.get('keywords', {})
                
            self.initialized = len(self.contents) > 0
            logger.info(f"Loaded knowledge base with {len(self.contents)} content sections")
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
            
            # Extract content sections with their headings
            sections = []
            
            # Split text into lines for analysis
            lines = full_text.split('\n')
            
            # Process lines to find sections
            i = 0
            current_title = None
            current_content_lines = []
            section_count = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Check if this line looks like a heading
                is_heading = False
                if line and len(line) < 80:
                    # Look for title patterns like "# Something" or short lines
                    if line.startswith('#') or (len(line) < 50 and not line.endswith('.')) or line.isupper():
                        # Check if next line is blank or next line is not a continuation
                        next_line = lines[i+1].strip() if i+1 < len(lines) else ""
                        if not next_line or next_line.startswith('#'):
                            is_heading = True
                
                if is_heading:
                    # Save previous section if it exists
                    if current_title:
                        section_id = f"section_{section_count}"
                        section_count += 1
                        content = "\n".join(current_content_lines)
                        kb.add_content(section_id, current_title, content)
                    
                    # Start new section
                    current_title = line.lstrip('#').strip()
                    current_content_lines = []
                elif line:
                    # Add to current content
                    current_content_lines.append(line)
                
                i += 1
            
            # Add the last section
            if current_title:
                section_id = f"section_{section_count}"
                content = "\n".join(current_content_lines)
                kb.add_content(section_id, current_title, content)
            
            # Mark as initialized if we have content
            kb.initialized = True
            logger.info(f"Extracted {section_count+1} sections from PDF")
    
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
    
    # Extract knowledge from PDF
    kb = extract_knowledge_from_pdf(DEFAULT_PDF_PATH)
    
    # Save knowledge base if successful
    if kb.initialized:
        kb.save(KNOWLEDGE_BASE_PATH)
    
    return kb


async def search_knowledge_base(query: str) -> Tuple[str, Optional[str]]:
    """Search knowledge base and return actual answer content."""
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
        
        # Get best match - IMPORTANT: This contains the full text, not just the heading
        content, score = results[0]
        
        # Remove title/heading part if present
        content_parts = content.split('\n\n', 1)
        if len(content_parts) > 1:
            # Full content after the heading
            answer = content_parts[1].strip()
        else:
            # Just use the whole content if we can't separate
            answer = content.strip()
        
        # If the answer is very short, it might be just a heading
        # In that case, include the second best result if available
        if len(answer) < 20 and len(results) > 1:
            logger.info("First result was too short, adding second result")
            second_content, _ = results[1]
            second_parts = second_content.split('\n\n', 1)
            if len(second_parts) > 1:
                answer += "\n\n" + second_parts[1].strip()
        
        logger.info(f"Found match for '{query}' (score: {score})")
        logger.info(f"Returning answer: {answer[:100]}...")
        
        return answer, None
    
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
            # Found relevant information - respond with full content
            logger.info(f"Answering with content: {content[:100]}...")
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
