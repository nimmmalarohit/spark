    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents in the knowledge base with better question handling."""
        if not self.initialized or self.vectors is None:
            logger.warning("Knowledge base not initialized yet.")
            return []
        
        # Preprocess query to handle variations
        # Remove common question words and normalize
        query = query.lower()
        question_starters = ["what is", "what are", "how to", "where is", "who is", "tell me about", "show me"]
        for starter in question_starters:
            if query.startswith(starter):
                query = query[len(starter):].strip()
        
        # Add common topic keywords to improve matching
        if "version" in query and "python" not in query and "spark" not in query:
            # If asking about versions generally, expand query
            expanded_query = f"{query} python spark version"
            query_vector = self.vectorizer.transform([expanded_query])
        else:
            query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all documents
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Return results with scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                score = similarities[idx]
                # Only include results with some similarity
                if score > 0.1:  # Lower threshold
                    results.append((self.documents[idx], self.metadata[idx], float(score)))
        
        return results
