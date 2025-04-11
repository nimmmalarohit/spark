# test_library_compatibility.py
import sys
import logging
import os
import tempfile
import shutil
import numpy as np

# Set up logging with multiple handlers to ensure visibility
def setup_logging():
    # Clear any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Configure root logger
    root.setLevel(logging.INFO)
    
    # Console handler with clear formatting
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('\n>>> %(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    root.addHandler(console)
    
    # Also add a print statement handler for backup
    class PrintHandler(logging.Handler):
        def emit(self, record):
            print(f"\n*** PRINT: {record.levelname} - {record.getMessage()}")
    
    print_handler = PrintHandler()
    print_handler.setLevel(logging.INFO)
    root.addHandler(print_handler)
    
    # Test if logging is working
    root.info("LOGGING TEST - If you can see this, logging is working correctly")
    print("DIRECT PRINT TEST - If you can see this, print is working correctly")

setup_logging()
logger = logging.getLogger(__name__)

def test_bs4():
    """Test BeautifulSoup functionality."""
    print("\n=== Testing BeautifulSoup ===")
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
        
        print(f"BeautifulSoup successfully parsed HTML: '{text.strip()[:50]}...'")
        return True
    except Exception as e:
        print(f"BeautifulSoup test failed: {e}")
        return False

def test_faiss_basic():
    """Test basic FAISS functionality."""
    print("\n=== Testing FAISS ===")
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
        
        print(f"FAISS index created and searched successfully")
        print(f"Found indices: {indices[0]}, distances: {distances[0]}")
        
        # Test save/load
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "test.index")
        
        try:
            faiss.write_index(index, temp_file)
            print(f"FAISS index saved successfully")
            
            loaded_index = faiss.read_index(temp_file)
            print(f"FAISS index loaded successfully with {loaded_index.ntotal} vectors")
        finally:
            shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"FAISS test failed: {e}")
        return False

def test_sentence_transformer():
    """Test sentence-transformer with safe imports."""
    print("\n=== Testing sentence-transformers ===")
    try:
        # Check huggingface_hub version first
        import huggingface_hub
        print(f"huggingface_hub version: {huggingface_hub.__version__}")
        
        # Try importing sentence_transformers
        from sentence_transformers import SentenceTransformer
        print("Successfully imported SentenceTransformer")
        
        # Try loading a small model - but wrap in try/except in case it fails
        try:
            print("Attempting to load a small sentence transformer model...")
            # Use 'all-MiniLM-L6-v2' which is a commonly used small model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embedding dimension
            dimension = model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {dimension}")
            
            # Try encoding a simple sentence
            test_sentence = "This is a test sentence for FAISS."
            embedding = model.encode(test_sentence)
            print(f"Successfully generated embedding with shape: {embedding.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    except ImportError as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_workflow():
    """Test a simplified version of the core workflow."""
    print("\n=== Testing integrated workflow ===")
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
        print(f"1. Extracted text from HTML: '{text.strip()[:50]}...'")
        
        # 2. Create text chunks
        chunks = [
            text[0:100],
            text[50:150]
        ]
        print(f"2. Created text chunks: {len(chunks)} chunks")
        
        # 3. Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, convert_to_numpy=True)
        print(f"3. Generated embeddings with shape: {embeddings.shape}")
        
        # 4. Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        print(f"4. Added embeddings to FAISS index")
        
        # 5. Search for a query
        query = "What Python version is used?"
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, k=1)
        
        print(f"5. Search results:")
        print(f"  Query: '{query}'")
        print(f"  Best match index: {indices[0][0]}")
        print(f"  Best match content: '{chunks[indices[0][0]][:50]}...'")
        print(f"  Distance: {distances[0][0]}")
        
        return True
    except Exception as e:
        print(f"Integrated workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n==================================================")
    print("STARTING LIBRARY COMPATIBILITY TESTS")
    print("==================================================")
    
    # Track overall success
    all_tests_passed = True
    
    # Test BeautifulSoup
    if not test_bs4():
        print("❌ BeautifulSoup test failed")
        all_tests_passed = False
    
    # Test FAISS
    if not test_faiss_basic():
        print("❌ FAISS test failed")
        all_tests_passed = False
    
    # Test sentence-transformers
    if not test_sentence_transformer():
        print("⚠️ sentence-transformers test failed")
        # Continue anyway - we'll try the integrated test
    
    # Only run integrated test if individual components worked
    if all_tests_passed:
        if not test_integrated_workflow():
            print("❌ Integrated workflow test failed")
            all_tests_passed = False
    
    # Final status
    print("\n==================================================")
    if all_tests_passed:
        print("✅ All compatibility tests PASSED!")
        sys.exit(0)
    else:
        print("❌ Some compatibility tests FAILED")
        sys.exit(1)
