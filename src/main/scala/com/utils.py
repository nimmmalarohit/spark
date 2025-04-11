# test_library_compatibility.py
import sys
import logging
import os
import tempfile
import shutil
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bs4():
    """Test BeautifulSoup functionality."""
    try:
        from bs4 import BeautifulSoup
        
        # Sample HTML content
        html = """
        <html>
            <body>
                <h1>Python version used in GCP</h1>
                <p>We are using 3.9 as of now.</p>
                <script>console.log('This should be removed');</script>
            </body>
        </html>
        """
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script tags
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        logger.info(f"✓ BeautifulSoup successfully parsed HTML: '{text.strip()[:50]}...'")
        return True
    except Exception as e:
        logger.error(f"BeautifulSoup test failed: {e}")
        return False

def test_faiss_basic():
    """Test basic FAISS functionality."""
    try:
        import faiss
        
        # Create a simple index
        dimension = 10
        index = faiss.IndexFlatL2(dimension)
        
        # Add some vectors
        vectors = np.random.random((5, dimension)).astype('float32')
        index.add(vectors)
        
        # Search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, k=2)
        
        logger.info(f"✓ FAISS index created and searched successfully")
        logger.info(f"  Found indices: {indices[0]}, distances: {distances[0]}")
        
        # Test save/load
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "test.index")
        
        try:
            faiss.write_index(index, temp_file)
            logger.info(f"✓ FAISS index saved successfully")
            
            loaded_index = faiss.read_index(temp_file)
            logger.info(f"✓ FAISS index loaded successfully with {loaded_index.ntotal} vectors")
        finally:
            shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        logger.error(f"FAISS test failed: {e}")
        return False

def test_sentence_transformer():
    """Test sentence-transformer with safe imports."""
    try:
        # Check huggingface_hub version first
        import huggingface_hub
        logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
        
        # Try importing sentence_transformers
        from sentence_transformers import SentenceTransformer
        
        # Try loading a small model - but wrap in try/except in case it fails
        try:
            logger.info("Attempting to load a small sentence transformer model...")
            # Use 'all-MiniLM-L6-v2' which is a commonly used small model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embedding dimension
            dimension = model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded successfully. Embedding dimension: {dimension}")
            
            # Try encoding a simple sentence
            test_sentence = "This is a test sentence for FAISS."
            embedding = model.encode(test_sentence)
            logger.info(f"✓ Successfully generated embedding with shape: {embedding.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def test_integrated_workflow():
    """Test a simplified version of the core workflow."""
    try:
        # Only proceed if we have the necessary imports
        from bs4 import BeautifulSoup
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Sample HTML content
        html = """
        <html>
            <body>
                <h1>Python version used in GCP</h1>
                <p>We are using 3.9 as of now. We have plans to migrate from Python 3.9 to 3.10.</p>
                <h1>Spark version is used in GCP</h1>
                <p>We are using Spark 2.2 as of now. We have plans to migrate from Spark 2.2 to 3.3.</p>
            </body>
        </html>
        """
        
        # 1. Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator="\n")
        logger.info(f"1. Extracted text from HTML: '{text.strip()[:50]}...'")
        
        # 2. Create text chunks
        chunks = [
            text[0:100],
            text[50:150]
        ]
        logger.info(f"2. Created text chunks: {len(chunks)} chunks")
        
        # 3. Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, convert_to_numpy=True)
        logger.info(f"3. Generated embeddings with shape: {embeddings.shape}")
        
        # 4. Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        logger.info(f"4. Added embeddings to FAISS index")
        
        # 5. Search for a query
        query = "What Python version is used?"
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, k=1)
        
        logger.info(f"5. Search results:")
        logger.info(f"  Query: '{query}'")
        logger.info(f"  Best match index: {indices[0][0]}")
        logger.info(f"  Best match content: '{chunks[indices[0][0]][:50]}...'")
        logger.info(f"  Distance: {distances[0][0]}")
        
        return True
    except Exception as e:
        logger.error(f"Integrated workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting library compatibility tests...")
    
    # Track overall success
    all_tests_passed = True
    
    # Test BeautifulSoup
    logger.info("\n--- Testing BeautifulSoup ---")
    if not test_bs4():
        logger.error("❌ BeautifulSoup test failed")
        all_tests_passed = False
    
    # Test FAISS
    logger.info("\n--- Testing FAISS ---")
    if not test_faiss_basic():
        logger.error("❌ FAISS test failed")
        all_tests_passed = False
    
    # Test sentence-transformers
    logger.info("\n--- Testing sentence-transformers ---")
    if not test_sentence_transformer():
        logger.warning("⚠️ sentence-transformers test failed")
        # Continue anyway - we'll try the integrated test
    
    # Only run integrated test if individual components worked
    if all_tests_passed:
        logger.info("\n--- Testing integrated workflow ---")
        if not test_integrated_workflow():
            logger.error("❌ Integrated workflow test failed")
            all_tests_passed = False
    
    # Final status
    if all_tests_passed:
        logger.info("\n✅ All compatibility tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some compatibility tests failed")
        sys.exit(1)
