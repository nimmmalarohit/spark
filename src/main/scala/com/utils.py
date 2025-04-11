import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test importing all required libraries."""
    logger.info("Testing library imports...")

    try:
        import rasa
        logger.info(f"✓ Rasa version: {rasa.__version__}")

        import rasa_sdk
        logger.info(f"✓ Rasa SDK version: {rasa_sdk.__version__}")

        import faiss
        logger.info(f"✓ FAISS imported successfully")

        import bs4
        logger.info(f"✓ BeautifulSoup4 version: {bs4.__version__}")

        import markdown
        logger.info(f"✓ Markdown version: {markdown.version}")

        import numpy as np
        logger.info(f"✓ NumPy version: {np.__version__}")

        import sentence_transformers
        logger.info(f"✓ Sentence Transformers version: {sentence_transformers.__version__}")

        logger.info("All imports successful!")
        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False


def test_sentence_transformers_faiss_integration():
    """Test the integration between sentence-transformers and FAISS."""
    logger.info("Testing sentence-transformers with FAISS integration...")

    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        # Load a small model for testing
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Create test sentences
        test_sentences = [
            "This is a test sentence for FAISS.",
            "Another example to check compatibility.",
            "Testing sentence transformers with FAISS."
        ]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = model.encode(test_sentences, convert_to_numpy=True)

        # Verify embedding dimensions
        dimension = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dimension}")

        # Create a simple FAISS index
        logger.info("Creating FAISS index...")
        index = faiss.IndexFlatL2(dimension)

        # Add vectors to the index
        logger.info("Adding vectors to index...")
        index.add(embeddings.astype(np.float32))

        # Test search functionality
        logger.info("Testing search functionality...")
        query = "Test sentence for compatibility"
        query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)

        distances, indices = index.search(query_vector, k=2)

        logger.info(f"Search results - indices: {indices}, distances: {distances}")
        logger.info("FAISS and sentence-transformers integration test successful!")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


def test_beautifulsoup():
    """Test basic BeautifulSoup functionality."""
    logger.info("Testing BeautifulSoup...")

    try:
        from bs4 import BeautifulSoup

        html = """
        <html>
            <body>
                <h1>Test Document</h1>
                <p>This is a paragraph with <span>some formatting</span>.</p>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()

        logger.info(f"Extracted text: {text.strip()}")
        logger.info("BeautifulSoup test successful!")
        return True

    except Exception as e:
        logger.error(f"BeautifulSoup test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting library compatibility tests...")

    # Test imports
    import_success = test_imports()
    if not import_success:
        logger.error("Import tests failed. Please check your environment.")
        sys.exit(1)

    # Test BeautifulSoup
    bs_success = test_beautifulsoup()
    if not bs_success:
        logger.warning("BeautifulSoup test failed.")

    # Test sentence-transformers with FAISS
    st_faiss_success = test_sentence_transformers_faiss_integration()
    if not st_faiss_success:
        logger.error("sentence-transformers and FAISS integration test failed.")
        sys.exit(1)

    logger.info("All compatibility tests passed successfully!")
