async def search_wiki_knowledge(tracker: Tracker) -> Tuple[str, Optional[str]]:
    """Use Rasa's NLU information to better search the knowledge base."""
    try:
        # Get Rasa's intent confidence and entities
        latest_message = tracker.latest_message
        intent = latest_message.get("intent", {}).get("name")
        confidence = latest_message.get("intent", {}).get("confidence", 0.0)
        entities = latest_message.get("entities", [])
        query = latest_message.get("text")
        
        # Initialize knowledge base
        knowledge_base = WikiKnowledgeBase()
        if not knowledge_base.load(KNOWLEDGE_BASE_PATH):
            success = initialize_knowledge_base()
            if not success:
                return None, "Knowledge base could not be initialized"
        
        # Enhance query with entity information from Rasa's NLU
        enhanced_query = query
        extracted_entities = []
        for entity in entities:
            entity_value = entity.get("value")
            extracted_entities.append(entity_value)
        
        if extracted_entities:
            # Add extracted entities to enhance the search
            enhanced_query = f"{query} {' '.join(extracted_entities)}"
        
        # Search knowledge base with the enhanced query
        results = knowledge_base.search(enhanced_query, k=3)
        if not results:
            return None, "No results found"
        
        # Get best result
        content, metadata, score = results[0]
        
        # Process and return just the content
        content_parts = content.split("\n\n", 1)
        actual_content = content_parts[1] if len(content_parts) > 1 else content
        actual_content = actual_content.replace('#', '').strip()
        
        return actual_content, None
    except Exception as e:
        logger.error(f"Error searching wiki: {str(e)}")
        return None, f"Error: {str(e)}"





def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
    """Simplified search that relies on Rasa's NLU for preprocessing."""
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







class ActionDefaultFallback(OptimusAction, action_name="action_gcp_default_fallback"):

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        latest_message = tracker.latest_message.get('text')
        logger.info(f'### User query "{latest_message}" falling back ###')
        
        # Use tracker to get all Rasa NLU information
        wiki_response, error = await search_wiki_knowledge(tracker)
        
        if wiki_response:
            # Found relevant information in knowledge base
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





# Add to your nlu.yml file
nlu:
  - intent: ask_version
    examples: |
      - what is the python version
      - what version of python are you using
      - tell me the python version
      - python version
      - what is the version of python
      - version of python
      - what python version do you use
      - python version information
      - which python version
      - what are the versions
      - version information
      - software versions
      - what versions are being used


