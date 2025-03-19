def _process_text(self, text: str, page_num: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Process text into coherent sections rather than small chunks."""
    if not text:
        return []
        
    # Split into sections based on Markdown-style headings
    sections = []
    current_heading = None
    current_content = []
    current_section = ""
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is a heading (starts with # or is short and next line is empty)
        is_heading = False
        if line.startswith('#'):
            is_heading = True
        elif line and len(line) < 80 and i < len(lines) - 1 and not lines[i+1].strip():
            is_heading = True
            
        if is_heading:
            # Save previous section if it exists
            if current_heading and current_content:
                full_section = f"{current_heading}\n\n{'\n'.join(current_content)}"
                sections.append((full_section, {
                    "page": page_num,
                    "heading": current_heading,
                    "section_type": "content"
                }))
            
            # Start new section
            current_heading = line.lstrip('#').strip()
            current_content = []
        elif line:
            # Add to current section content
            current_content.append(line)
            
        i += 1
    
    # Add final section
    if current_heading and current_content:
        full_section = f"{current_heading}\n\n{'\n'.join(current_content)}"
        sections.append((full_section, {
            "page": page_num,
            "heading": current_heading,
            "section_type": "content"
        }))
    
    # If no sections were found, create one from the entire text
    if not sections and text.strip():
        sections.append((text.strip(), {
            "page": page_num,
            "heading": "Page content",
            "section_type": "content"
        }))
    
    return sections



async def search_wiki_knowledge(tracker: Tracker) -> Tuple[str, Optional[str]]:
    """Search the wiki knowledge base using Rasa NLU information."""
    try:
        # Check if knowledge base needs updating
        if not is_knowledge_base_current():
            logger.info("PDF has been modified. Refreshing knowledge base...")
            initialize_knowledge_base(force_refresh=True)
        
        # Get query from tracker
        query = tracker.latest_message.get('text')
        
        # Load knowledge base
        knowledge_base = WikiKnowledgeBase()
        if not knowledge_base.load(KNOWLEDGE_BASE_PATH):
            success = initialize_knowledge_base()
            if not success:
                return None, "Knowledge base could not be initialized"
        
        # Get Rasa's entity information to enhance query
        entities = tracker.latest_message.get("entities", [])
        extracted_entities = []
        for entity in entities:
            entity_value = entity.get("value")
            if entity_value:
                extracted_entities.append(entity_value)
        
        # Create enhanced query with entities
        enhanced_query = query
        if extracted_entities:
            enhanced_query = f"{query} {' '.join(extracted_entities)}"
        
        # Perform search with enhanced query
        results = knowledge_base.search(enhanced_query, k=3)
        
        # Try original query if enhanced query doesn't yield results
        if not results:
            results = knowledge_base.search(query, k=3)
            if not results:
                return None, "No results found"
        
        # Get best result
        content, metadata, score = results[0]
        
        # Only proceed if score is high enough
        if score < 0.1:
            return None, "Low confidence results"
        
        # Get the full section content without any formatting
        # This ensures we return complete paragraphs
        full_content = content.strip()
        
        # Remove heading markers and clean up
        clean_content = full_content
        lines = clean_content.split('\n')
        
        # If first line looks like a heading, format properly
        if lines and len(lines) > 1:
            # Keep first line as heading if it starts with # or is short
            if lines[0].startswith('#') or (len(lines[0]) < 80 and not lines[1].strip()):
                heading = lines[0].lstrip('#').strip()
                body = '\n'.join(lines[2:]).strip()  # Skip the blank line after heading
                clean_content = body
        
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
            # Found relevant information - respond with just the content in a SINGLE message
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
