import os
import time
import asyncio
from typing import Dict, Any, Optional
import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import glob
import threading
import subprocess
import signal

# importing from directories
from load_data import DataHandler
from retrieval import QueryProcessor
from llm_manager import LLMManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths to data directories (up one level from src)
DATA_DIR = os.path.join('..', 'data')
PROCESSED_DATA_DIR = os.path.join('..', 'processed_data')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

app = FastAPI(title="Religious Book Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store globals in app.state
app.state.query_processor = None
app.state.gradio_process = None
app.state.backend_ready = False
app.state.llm_manager = None

class QueryRequest(BaseModel):
    query: str

class ProcessDataRequest(BaseModel):
    filename: str
    batch_size: int = 50

@app.get("/")
async def root():
    return {"message": "Religious Book Chatbot API is running. UI available at http://localhost:7860"}

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a user query and return relevant content from the religious book."""
    try:
        result = app.state.query_processor.process_query(request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verse/{chapter}/{verse}")
async def get_verse(chapter: int, verse: int, section: Optional[str] = Query(None)):
    """Get a specific verse by chapter and verse number."""
    try:
        section = section or "all"
        result = app.state.query_processor.verse_retriever.retrieve_verse(chapter, verse, section)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error retrieving verse: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{keyword}")
async def search_keyword(keyword: str, limit: Optional[int] = Query(5)):
    """Search for verses containing a specific keyword."""
    try:
        results = app.state.query_processor.verse_retriever.search_by_keyword(keyword, limit)
        return {"keyword": keyword, "results": results}
    except Exception as e:
        logger.error(f"Error searching keyword: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-data")
async def process_data(request: ProcessDataRequest):
    """Process a data file and create indices."""
    try:
        handler = DataHandler(data_dir=DATA_DIR, output_dir=PROCESSED_DATA_DIR)
        
        # Process the PDF file
        output_filename = handler.process_pdf(request.filename, request.batch_size)
        
        return {"message": f"Successfully processed {request.filename}", "output_file": output_filename}
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-available-files")
async def list_available_files():
    """List available files in the data directory."""
    try:
        # Use the absolute path to the data directory
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New API endpoint for checking if backend is ready
@app.get("/status/ready")
async def backend_ready():
    """Report if backend is fully initialized and ready."""
    return {"ready": app.state.backend_ready}

def initialize_query_processor():
    """Initialize the query processor with the correct index files."""
    # Look for processed files
    processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "processed_*.json"))
    
    if processed_files:
        # Use the most recent processed file as the basis for index filenames
        latest_file = max(processed_files, key=os.path.getmtime)
        base_name = os.path.basename(latest_file).replace("processed_", "").replace(".json", "")
        
        index_file = f"index_processed_{base_name}.json"
        keyword_file = f"keyword_index_processed_{base_name}.json"
        
        logger.info(f"Using index file: {index_file}")
        logger.info(f"Using keyword file: {keyword_file}")
    else:
        # No processed files found, use default names
        logger.warning("No processed files found. Using default index filenames.")
        index_file = "index_processed_verses.json"
        keyword_file = "keyword_index_processed_verses.json"
        
        # Create empty index files if they don't exist
        if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, index_file)):
            with open(os.path.join(PROCESSED_DATA_DIR, index_file), 'w') as f:
                f.write("{}")
        if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, keyword_file)):
            with open(os.path.join(PROCESSED_DATA_DIR, keyword_file), 'w') as f:
                f.write("{}")
    
    # Initialize query processor
    app.state.query_processor = QueryProcessor(
        data_dir=PROCESSED_DATA_DIR,
        index_file=index_file,
        keyword_index_file=keyword_file
    )

def watch_for_new_files():
    """Watch for new PDF files and process them automatically."""
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        # Get current PDF files
        pdf_files = set([f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')])
        
        for pdf_file in pdf_files:
            if not is_file_already_processed(pdf_file):
                try:
                    logger.info(f"New file detected: {pdf_file}. Processing...")
                    handler = DataHandler(data_dir=DATA_DIR, output_dir=PROCESSED_DATA_DIR)
                    handler.process_pdf(pdf_file, batch_size=50)
                    logger.info(f"Successfully processed {pdf_file}")
                    
                    # Reinitialize the query processor to use the new files
                    initialize_query_processor()
                    
                except Exception as e:
                    logger.error(f"Error processing new file {pdf_file}: {str(e)}")

def is_file_already_processed(filename):
    """Check if a PDF file has already been processed."""
    base_name = os.path.splitext(filename)[0]
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"processed_{base_name}.json")
    return os.path.exists(processed_file)

def start_gradio_ui():
    """Start the Gradio UI in a separate process."""
    try:
        # Use Python to start gradio_app.py as a separate process
        gradio_script = os.path.join(os.path.dirname(__file__), 'gradio_app.py')
        
        # Start a new process for Gradio
        process = subprocess.Popen(['python', gradio_script], 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        
        app.state.gradio_process = process
        logger.info(f"Gradio UI started as a separate process (PID: {process.pid})")
        
        return process
    except Exception as e:
        logger.error(f"Error starting Gradio UI: {e}")
        return None

# On startup event for FastAPI
@app.on_event("startup")
async def startup_event():
    """Process PDF files and start Gradio UI when FastAPI starts."""
    logger.info("Starting backend processing...")

    # Initialize LLM Manager singleton
    app.state.llm_manager = LLMManager.get_instance()
    logger.info("LLM Manager initialized")

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Check if there are PDF files to process
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    # Process all PDF files before starting UI
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in pdf_files:
            if not is_file_already_processed(pdf_file):
                try:
                    logger.info(f"Processing file: {pdf_file}...")
                    handler = DataHandler(data_dir=DATA_DIR, output_dir=PROCESSED_DATA_DIR)
                    handler.process_pdf(pdf_file, batch_size=50)
                    logger.info(f"Successfully processed {pdf_file}")
                except Exception as e:
                    logger.error(f"Error processing file {pdf_file}: {str(e)}")
    
    # Initialize query processor with newly processed files
    initialize_query_processor()
    
    # Mark backend as ready
    app.state.backend_ready = True
    logger.info("Backend initialization complete, starting Gradio UI...")
    
    # Start Gradio UI in a separate process
    gradio_process = start_gradio_ui()
    
    # Start file watcher
    watcher_thread = threading.Thread(target=watch_for_new_files, daemon=True)
    watcher_thread.start()
    logger.info("File watcher started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown."""
    logger.info("Application shutting down...")
    
    # Kill Gradio process if it exists
    if app.state.gradio_process:
        try:
            # Try to terminate gracefully first
            if hasattr(signal, 'SIGTERM'):
                app.state.gradio_process.send_signal(signal.SIGTERM)
                # Wait a bit for graceful shutdown
                try:
                    app.state.gradio_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # If it doesn't exit in time, force kill
                    app.state.gradio_process.kill()
            else:
                # On Windows, just use terminate
                app.state.gradio_process.terminate()
            
            logger.info("Gradio UI process terminated")
        except Exception as e:
            logger.error(f"Error terminating Gradio process: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)