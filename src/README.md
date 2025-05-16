# Bhagavad-Gita-Chatbot
A modular chatbot system for navigating and querying religious texts, with support for Sanskrit verses, translations, transliterations, and purports.
Features. This chatbot is made considerating all Hindu religious books in english language. 

PDF Processing: Extract structured data from religious text PDFs
Sanskrit Support: Handle both Devanagari script and transliterated Sanskrit verses
Efficient Retrieval: Index and search verses by chapter, verse, or keyword
LLM Integration: Use language models to understand natural language queries
Separate UI: Clean separation between API and UI components
Single Command Start: Run both API and UI with one command

Project Structure
project/
├── data/                  # Raw religious book PDFs
├── processed_data/        # Processed and indexed data
├── src/
│   ├── llm_interface.py   # LLM API interaction 
│   ├── load_data.py       # Data loading and processing
│   ├── retrieval.py       # Query parsing and content retrieval
├── main.py                # FastAPI application that also starts Gradio
├── gradio_app.py          # Gradio UI in a separate file
└── requirements.txt       # Project dependencies
Setup Instructions
1. Install dependencies
bashpip install -r requirements.txt
2. Set up API keys
Set environment variables for LLM API keys:
bashexport GROQ_API_KEY=your-first-api-key
export GROQ_API_KEY2=your-second-api-key
export GROQ_API_KEY3=your-third-api-key
3. Prepare data
Place your religious text PDFs in the data/ directory.
4. Start the application
bashuvicorn main:app --reload
This single command will start:

The FastAPI backend at http://localhost:8000
The Gradio UI at http://localhost:7860

Usage
Accessing the Interface
After starting the application with uvicorn main:app --reload, you can access:

The Gradio UI at http://localhost:7860
The API directly at http://localhost:8000

Example Queries

"What does Chapter 2 Verse 41 say?"
"Tell me the purport for 6.5"
"What is the translation of 12.15?"
"Search for karma"

Interface Tabs

Chat: Ask natural language questions about verses
Specific Verse: Directly access a verse by chapter and verse number
Keyword Search: Search for specific keywords in the religious text
Data Processing: Process new PDF files and add them to the system

API Endpoints

POST /query - Process a natural language query
GET /verse/{chapter}/{verse} - Get a specific verse
GET /search/{keyword} - Search for verses containing a keyword
POST /process-data - Process a data file
GET /list-available-files - List available files

Technical Details
Components

LLM Interface (llm_interface.py):

Handles communication with language model APIs
Provides key rotation for managing rate limits
Extracts structured information from user queries


Data Loading and Processing (load_data.py):

Extracts text and structure from PDFs
Identifies Sanskrit verses in both Devanagari and transliterated forms
Processes translations and purports
Creates indexable data structures


Query Processing and Retrieval (retrieval.py):

Parses natural language queries
Retrieves relevant verses and sections
Handles keyword searches


FastAPI Backend (main.py):

Provides RESTful API endpoints
Starts the Gradio UI in a separate thread
Manages core backend functionality


Gradio UI (gradio_app.py):

Provides user-friendly interface in a separate file
Communicates with the FastAPI backend via HTTP
Formats responses for readability



How It Works
The system uses a multithreaded approach:

When main.py starts (via uvicorn), it launches the FastAPI server
During FastAPI startup, it spawns a separate thread that runs the Gradio UI
Both servers run independently but are started with a single command
The Gradio UI makes HTTP requests to the FastAPI backend

Extending the System
To add support for additional religious texts:

Add PDFs to the data/ directory
Process them using the "Data Processing" tab or API endpoint
The system will automatically extract and index the content

Limitations

PDFs must follow a consistent format with clear verse markers
Free tier LLM has 30 RPM and 12000 TPM limits per API key
Long purports are handled efficiently but may be truncated in the UI display

Future Enhancements

Vector database integration for semantic search
Support for more religious text formats
Multi-language interface
Advanced filtering and comparison features