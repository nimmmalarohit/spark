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
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "gcp_t2_l3.pdf")  # Updated to match your actual PDF name
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "wiki_knowledge.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class PDFKnowledgeBase:
    """Robust knowledge base for PDF content."""
    
    def __init__(self):
        self.sections = {}  # {id: {title, content}}
        self.initialized = False
        
    def add_section(self, section_id: str, title: str, content: str):
        """Add a section to the knowledge base."""
        self.sections[section_id] = {
            "title": title,
            "content": content
        }
        
    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """Search for content matching the query."""
        if not self.initialized or not self.sections:
            logger.warning("Knowledge base not initialized or empty")
            return []
        
        # Clean query
        query = query.lower().strip()
        
        # Extract keywords (anything 3+ chars)
        keywords = [w.lower() for w in re.findall(r'\b\w{3,}\b', query)]
        
        # Calculate match scores for each section
        results = []
        for section_id, section in self.sections.items():
            title = section["title"].lower()
            content = section["content"].lower()
            
            # Calculate score based on keyword matches
            score = 0
            for keyword in keywords:
                # Higher score for title matches
                if keyword in title:
                    score += 3
                # Score for content matches
                if keyword in content:
                    score += 1
            
            # Normalize by number of keywords
            if keywords:
                score = score / len(keywords)
                
                # Only include if score is reasonable
                if score > 0.5:
                    full_content = f"{section['title']}\n\n{section['content']}"
                    results.append((section_id, full_content, score))
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def save(self, file_path: str) -> bool:
        """Save knowledge base to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.sections, f, indent=2)
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
                self.sections = json.load(f)
            
            self.initialized = len(self.sections) > 0
            logger.info(f"Loaded knowledge base with {len(self.sections)} sections")
            return self.initialized
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False


def extract_pdf_content(pdf_path: str) -> PDFKnowledgeBase:
    """Extract content from PDF with improved section detection."""
    kb = PDFKnowledgeBase()
    
    try:
        logger.info(f"Extracting content from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return kb
        
        # Debug: Log the PDF file size
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        
        # Open PDF and extract content
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages")
            
            # First pass: collect all text
            raw_text = ""
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Add page number marker to help with debugging
                    raw_text += f"\n\n--- Page {i+1} ---\n\n{text}"
            
            # Save raw text for debugging
            debug_path = os.path.join(DATA_DIR, "pdf_raw_text.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            logger.info(f"Saved raw PDF text to {debug_path}")
            
            # Break text into lines
            lines = raw_text.split('\n')
            
            # Identify potential section headings using multiple heuristics
            sections = []
            section_starts = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines and page markers
                if not line or line.startswith('--- Page'):
                    continue
                
                is_heading = False
                
                # Check for heading patterns
                if line.startswith('#'):
                    is_heading = True
                elif re.match(r'^[A-Z][A-Za-z\s]+$', line) and len(line) < 60:
                    # Capitalized short line
                    is_heading = True
                elif re.match(r'^\d+\.\s+[A-Z]', line):
                    # Numbered section
                    is_heading = True
                elif re.match(r'^[A-Z][a-z]+\s+[vV]ersion', line):
                    # Special case for "Python version" etc.
                    is_heading = True
                    
                if is_heading:
                    section_starts.append(i)
            
            # Create sections based on identified headings
            for j, start_idx in enumerate(section_starts):
                # Get the heading line
                heading = lines[start_idx].strip().lstrip('#').strip()
                
                # Get content until next heading or end
                content_lines = []
                end_idx = section_starts[j+1] if j+1 < len(section_starts) else len(lines)
                
                for k in range(start_idx+1, end_idx):
                    line = lines[k].strip()
                    if line and not line.startswith('--- Page'):
                        content_lines.append(line)
                
                content = '\n'.join(content_lines)
                
                # Add to knowledge base
                section_id = f"section_{j}"
                kb.add_section(section_id, heading, content)
                sections.append((heading, content))
            
            logger.info(f"Extracted {len(sections)} sections from PDF")
            
            # If no sections were found, create one with all content
            if not sections:
                logger.warning("No sections detected, creating one section with all content")
                kb.add_section("section_0", "PDF Content", raw_text)
            
            # Save sections for debugging
            debug_sections_path = os.path.join(DATA_DIR, "pdf_sections.txt")
            with open(debug_sections_path, 'w', encoding='utf-8') as f:
                for heading, content in sections:
                    f.write(f"==== {heading} ====\n\n{content}\n\n")
            logger.info(f"Saved extracted sections to {debug_sections_path}")
            
            kb.initialized = True
    except Exception as e:
        logger.error(f"Error extracting PDF content: {str(e)}")
    
    return kb


def initialize_knowledge_base(force_refresh=True):  # Set to True to force refresh
    """Initialize or refresh the knowledge base."""
    # Always validate PDF exists
    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error(f"PDF file not found: {DEFAULT_PDF_PATH}")
        logger.error(f"Available files in data directory: {os.listdir(DATA_DIR)}")
        return None
    
    # Try to load existing KB if not forcing refresh
    kb = PDFKnowledgeBase()
    if not force_refresh and os.path.exists(KNOWLEDGE_BASE_PATH):
        if kb.load(KNOWLEDGE_BASE_PATH):
            return kb
    
    # Extract content from PDF
    logger.info("Extracting content from PDF...")
    kb = extract_pdf_content(DEFAULT_PDF_PATH)
    
    # Save knowledge base
    if kb.initialized:
        kb.save(KNOWLEDGE_BASE_PATH)
    else:
        logger.error("Failed to initialize knowledge base from PDF")
    
    return kb


async def search_knowledge_base(query: str) -> Tuple[str, Optional[str]]:
    """Search knowledge base and return the best matching content."""
    try:
        # Get knowledge base
        kb = initialize_knowledge_base(force_refresh=False)
        
        if not kb or not kb.initialized:
            logger.error("Knowledge base not available")
            return None, "Knowledge base not available"
        
        # Search for matching content
        results = kb.search(query)
        
        # Debug
        logger.info(f"Search for '{query}' found {len(results)} results")
        for i, (section_id, content, score) in enumerate(results[:3]):
            logger.info(f"Result {i+1}: section_id={section_id}, score={score}, content preview: {content[:50]}...")
        
        if not results:
            return None, "No matching information found"
        
        # Get the best match
        _, content, score = results[0]
        
        # Format the response to be just the content part
        content_parts = content.split('\n\n', 1)
        if len(content_parts) > 1:
            # Return content without the heading
            return content_parts[1].strip(), None
        else:
            # Return full content if no clear separation
            return content.strip(), None
    
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return None, f"Error: {str(e)}"


class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):
    """Enhanced fallback with knowledge base search."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        query = tracker.latest_message.get('text')
        logger.info(f'### User query "{query}" falling back ###')
        
        # Try to find information in knowledge base
        content, error = await search_knowledge_base(query)
        
        if content:
            # Found relevant information
            logger.info(f"Responding with content: {content[:100]}...")
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
