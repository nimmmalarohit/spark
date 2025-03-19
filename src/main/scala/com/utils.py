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
            
        results = knowledge_base.search(query, k=3)  # Get top 3 results
        if not results:
            return None, "No results found"
        
        # Check if any result has good similarity
        best_result = results[0]
        content, metadata, score = best_result
        
        # Only return if confidence is high enough
        if score < 0.15:  # Lower threshold to catch more variations
            return None, "Low confidence results"
        
        # Format response in a cleaner way
        # Extract the actual content without the heading
        content_parts = content.split("\n\n", 1)
        actual_content = content_parts[1] if len(content_parts) > 1 else content
        
        # Clean up the content - remove any markdown formatting
        actual_content = actual_content.replace('#', '').strip()
        
        # Just return the content directly, without extra formatting
        return actual_content, None
    except Exception as e:
        logger.error(f"Error searching wiki: {str(e)}")
        return None, f"Error: {str(e)}"
