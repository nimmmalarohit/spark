async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
    latest_message = tracker.latest_message.get('text')
    logger.info(f'### User query "{latest_message}" falling back ###')
    
    # Try to find relevant information in PDF knowledge base
    wiki_response, error = await search_wiki_knowledge(latest_message)
    
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
