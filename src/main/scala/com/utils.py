class ActionRefreshKnowledgeBase(OptimusAction, action_name="action_refresh_knowledge_base"):
    """Action to manually refresh the knowledge base."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        logger.info("Manual knowledge base refresh requested")
        
        try:
            # Force PDF timestamp check
            if os.path.exists(f"{KNOWLEDGE_BASE_PATH}.json"):
                try:
                    os.remove(f"{KNOWLEDGE_BASE_PATH}.json")
                    logger.info("Deleted existing knowledge base file")
                except Exception as e:
                    logger.warning(f"Could not remove old knowledge base file: {e}")
            
            # Recreate the knowledge base
            success = initialize_knowledge_base(force_refresh=True)
            
            if success:
                dispatcher.utter_message(text="✅ Knowledge base has been successfully refreshed with the latest PDF content.")
            else:
                dispatcher.utter_message(text="❌ Failed to refresh knowledge base. Please check logs for details.")
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            dispatcher.utter_message(text=f"❌ Error refreshing knowledge base: {str(e)}")
            
        # Important: Return empty list to prevent leaking content from PDFs
        return []

class ActionWikiQuery(OptimusAction, action_name="action_wiki_query"):
    """Action to directly query the knowledge base."""

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        try:
            query = tracker.latest_message.get('text')
            if not query:
                dispatcher.utter_message(text="Please provide a query to search for information.")
                return []
            
            # Get response from knowledge base
            response, error = await search_wiki_knowledge(tracker)
            
            if error:
                dispatcher.utter_message(text=f"Sorry, I couldn't find information about that: {error}")
            elif response:
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(text="No relevant information found in my knowledge base.")
                
        except Exception as e:
            logger.error(f"Error in wiki query: {e}")
            dispatcher.utter_message(text="Sorry, I encountered an error while searching for information.")
            
        # Important: Return empty list to prevent content leakage
        return []


- intent: wiki_query
  examples: |
    - what does the wiki say about [query]
    - search the wiki for [topic]
    - find information about [topic]
    - tell me about [topic] from the documentation
    - what information do you have about [topic]
    - what is [topic]
    - who is [person]
    - search for [query]

- rule: Wiki Query
  steps:
  - intent: wiki_query
  - action: action_wiki_query
