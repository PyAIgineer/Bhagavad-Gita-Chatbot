import os
import re
import json
import time
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ReligiousTextProcessor")

# Backwards compatibility wrapper for the existing codebase
class DataHandler:
    """Compatibility wrapper class to maintain the same interface for the existing codebase."""
    
    def __init__(self, data_dir="data", output_dir="processed_data"):
        """Initialize with directories and components."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get API keys from environment
        api_keys = [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("GROQ_API_KEY2"),
            os.environ.get("GROQ_API_KEY3")
        ]
        api_keys = [key for key in api_keys if key]
        
        # Initialize the processor
        self.processor = ReligiousTextProcessor(
            data_dir=data_dir,
            output_dir=output_dir,
            api_keys=api_keys
        )
        
        logger.info(f"DataHandler initialized with data_dir={data_dir}, output_dir={output_dir}")
    
    def load_pdf(self, filename: str) -> str:
        """Load text content from a PDF file using PyMuPDF (fitz)."""
        return self.processor.extract_text_from_pdf(filename)
    
    def process_pdf(self, filename: str, batch_size: int = 50):
        """Process a PDF file from loading to creating indices."""
        logger.info(f"Processing PDF file: {filename}")
        
        try:
            result = self.processor.process_file(filename)
            return os.path.basename(result["output_file"])
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            raise
    
    def extract_toc(self, filename: str):
        """Extract table of contents from PDF to understand book structure."""
        return self.processor.extract_toc(filename)
    
    def extract_verses(self, text: str, filename: str):
        """Extract verses and their components from the religious text."""
        # This is handled internally by the processor now, but we keep the method
        # for backward compatibility
        logger.warning("extract_verses called directly - this is handled by process_pdf now")
        return []
    
    def save_processed_data(self, verses, filename: str):
        """Save processed verses to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        logger.info(f"Saving {len(verses)} verses to {filepath}")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(verses, file, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
            raise

class APIKeyManager:
    """Manages API keys with rotation to respect rate limits."""
    
    def __init__(self, api_keys):
        """Initialize with list of API keys."""
        self.api_keys = deque(api_keys if api_keys else [])
        self.current_key = None
        self.call_timestamps = {key: [] for key in self.api_keys}
        self.max_rpm = 25  # 25 calls per minute to stay under 30 RPM limit
        logger.info(f"APIKeyManager initialized with {len(self.api_keys)} keys")
        
        # Preload Groq clients with disabled retries
        self.clients = {}
        try:
            from groq import Groq
            for key in self.api_keys:
                if key:
                    self.clients[key] = Groq(api_key=key, max_retries=0)
        except ImportError:
            logger.warning("Groq library not found. Cannot pre-create clients.")
    
    def get_key(self):
        """Get the next available API key respecting rate limits."""
        if not self.api_keys:
            logger.warning("No API keys available")
            return None
            
        # If we have a current key, check if it's still within rate limits
        now = time.time()
        if self.current_key:
            # Remove timestamps older than 1 minute
            self.call_timestamps[self.current_key] = [
                ts for ts in self.call_timestamps[self.current_key] 
                if now - ts < 60
            ]
            
            # If current key has capacity, keep using it
            if len(self.call_timestamps[self.current_key]) < self.max_rpm:
                return self.current_key
        
        # Try each key in rotation until we find one with capacity
        original_key = self.current_key
        for _ in range(len(self.api_keys)):
            self.api_keys.rotate(-1)  # Move to next key
            key = self.api_keys[0]
            
            # Clean up old timestamps
            self.call_timestamps[key] = [
                ts for ts in self.call_timestamps[key] 
                if now - ts < 60
            ]
            
            # Check if this key has capacity
            if len(self.call_timestamps[key]) < self.max_rpm:
                self.current_key = key
                logger.debug(f"Switched API key from {original_key} to {key}")
                return key
        
        # If all keys are at capacity, use the one with the oldest timestamp
        min_time = float('inf')
        min_key = None
        for key in self.api_keys:
            if self.call_timestamps[key]:
                oldest = min(self.call_timestamps[key])
                if oldest < min_time:
                    min_time = oldest
                    min_key = key
        
        if min_key:
            wait_time = 60 - (now - min_time) + 0.1  # Add a small buffer
            logger.info(f"All API keys at capacity. Waiting {wait_time:.2f}s for {min_key}")
            time.sleep(wait_time)
            self.call_timestamps[min_key] = [
                ts for ts in self.call_timestamps[min_key] 
                if now + wait_time - ts < 60
            ]
            self.current_key = min_key
            return min_key
        
        logger.error("Failed to get an available API key")
        return None
    
    def record_usage(self, key=None):
        """Record usage of a key to track rate limits."""
        key = key or self.current_key
        if key:
            self.call_timestamps[key].append(time.time())
            logger.debug(f"Recorded usage of API key {key}")


class LLMProcessor:
    """Handles LLM processing using Groq API."""
    
    def __init__(self, api_keys, model_name="llama3-8b-8192"):
        """Initialize with API keys and model name."""
        self.key_manager = APIKeyManager(api_keys)
        self.model_name = model_name
        self.clients = {}  # Store pre-configured clients
        
        # Import Groq only if we have API keys
        if api_keys:
            try:
                from groq import Groq
                self.Groq = Groq
                
                # Pre-create clients with max_retries=0
                for i, key in enumerate(api_keys):
                    if key:
                        self.clients[key] = Groq(api_key=key, max_retries=0)
                
                logger.info(f"LLMProcessor initialized with model {model_name} and {len(self.clients)} clients")
            except ImportError:
                logger.warning("Groq library not found. Install with 'pip install groq'")
                self.Groq = None
        else:
            self.Groq = None
            logger.warning("No API keys provided. LLM processing will be skipped.")
    
    def call_llm(self, messages, temperature=0, retries=2):
        """Call LLM with automatic key rotation and rate limiting."""
        if not self.Groq:
            logger.warning("Groq not available. Skipping LLM call.")
            return None
        
        # Check content length to avoid payload too large errors
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        if total_length > 4000:  # Conservative limit
            logger.warning(f"Content length {total_length} exceeds safe limit. Truncating.")
            # Truncate the longest message content
            longest_idx = max(range(len(messages)), key=lambda i: len(messages[i].get("content", "")))
            content = messages[longest_idx]["content"]
            messages[longest_idx]["content"] = content[:4000] 
        
        # Try all keys in rotation
        keys_tried = set()
        total_attempts = 0
        max_attempts = retries * len(self.key_manager.api_keys) + 1
        
        while len(keys_tried) < len(self.key_manager.api_keys) and total_attempts < max_attempts:
            try:
                api_key = self.key_manager.get_key()
                if not api_key:
                    logger.error("No API key available")
                    return None
                
                keys_tried.add(api_key)
                total_attempts += 1
                
                # Use the pre-configured client with max_retries=0
                client = self.clients.get(api_key)
                if not client:
                    # Fall back to creating a new client if needed
                    client = self.Groq(api_key=api_key, max_retries=0)
                    self.clients[api_key] = client
                
                logger.info(f"Trying API key {len(keys_tried)}/{len(self.key_manager.api_keys)} - Attempt {total_attempts}/{max_attempts}")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                
                self.key_manager.record_usage(api_key)
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"LLM call failed: {error_str}")
                
                # If hit rate limit (429), try next key immediately
                if "rate limit" in error_str.lower() or "429" in error_str:
                    logger.warning(f"Rate limit hit. Switching to next API key.")
                    continue
                elif "timeout" in error_str.lower() or "connection" in error_str.lower():
                    # Network errors might be temporary
                    import time
                    time.sleep(1)
                    continue
                else:
                    # For other errors
                    return None
        
        logger.warning(f"All API keys exhausted after {total_attempts} attempts. Aborting LLM call.")
        return None

class ReligiousTextProcessor:
    """Main class for processing religious texts with LLM and traditional fallbacks."""
    
    def __init__(self, data_dir="data", output_dir="processed_data", api_keys=None):
        """Initialize with directories and API keys."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize LLM processor if API keys provided
        self.llm = LLMProcessor(api_keys) if api_keys else None
        logger.info(f"ReligiousTextProcessor initialized with data_dir={data_dir}, output_dir={output_dir}")
    
    def extract_text_from_pdf(self, filename):
        """Extract text from PDF file."""
        filepath = os.path.join(self.data_dir, filename)
        logger.info(f"Extracting text from {filepath}")
        
        try:
            doc = fitz.open(filepath)
            text = ""
            total_pages = len(doc)
            
            logger.info(f"PDF has {total_pages} pages")
            for i, page in enumerate(doc):
                if i % 100 == 0:
                    logger.info(f"Processing page {i+1}/{total_pages}")
                text += page.get_text() + "\n"
            
            doc.close()
            logger.info("PDF text extraction completed")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_book_structure(self, filename):
        """Extract book structure using multiple approaches in order of reliability:
        1. PDF metadata TOC
        2. LLM analysis of TOC/index sections in text
        3. Regex pattern matching for chapter markers
        """
        logger.info(f"Extracting book structure from {filename}")
        
        # APPROACH 1: Try to get TOC from PDF metadata (previously in extract_toc method)
        try:
            filepath = os.path.join(self.data_dir, filename)
            doc = fitz.open(filepath)
            pdf_toc = doc.get_toc()
            
            if pdf_toc:
                logger.info(f"Found TOC in PDF metadata with {len(pdf_toc)} items")
                doc.close()
                
                # Process and format TOC data
                chapters = []
                for level, title, page in pdf_toc:
                    # Look for chapter information in title
                    chapter_match = re.search(r'(?:chapter|CHAPTER)[:\s-]*(\d+)', title, re.IGNORECASE)
                    chapter_num = int(chapter_match.group(1)) if chapter_match else None
                    
                    # Only add items that appear to be chapters
                    if chapter_num:
                        chapters.append({
                            "chapter_number": chapter_num,
                            "chapter_name": title,
                            "page": page,
                            "level": level
                        })
                
                # If we found chapter information, return it
                if chapters:
                    logger.info(f"Extracted {len(chapters)} chapters from PDF metadata")
                    return {"source": "pdf_metadata", "chapters": chapters}
            
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting TOC from PDF metadata: {str(e)}")
        
        # APPROACH 2: Try to find TOC/contents section in text and use LLM to analyze it
        try:
            # Extract text from PDF if we haven't already
            text = self.extract_text_from_pdf(filename)
            
            # Look for TOC/contents sections
            toc_patterns = [
                r'(?:TABLE\s+OF\s+CONTENTS|CONTENTS|Index|TABLE\s+OF\s+CONTENT).*?(?=CHAPTER|Chapter|INTRODUCTION|Introduction)',
                r'(?:Contents|Index).*?(?=Chapter|CHAPTER)'
            ]
            
            toc_section = None
            for pattern in toc_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    toc_section = match.group(0)
                    break
            
            if toc_section and self.llm:
                logger.info("Found TOC section in text, using LLM to extract structure")
                
                # Limit length to avoid payload errors
                toc_section = toc_section[:3000] if len(toc_section) > 3000 else toc_section
                
                system_message = {
                    "role": "system", 
                    "content": "You are an expert in analyzing the structure of books, especially religious texts."
                }
                
                user_message = {
                    "role": "user",
                    "content": f"""Analyze this table of contents/index section and extract the book structure.

    Text section:
    {toc_section}

    Extract all chapters with their numbers and titles. Return ONLY a JSON array of objects with:
    1. chapter_number: The chapter number (integer)
    2. chapter_name: The complete chapter title/name

    Return valid JSON that can be parsed."""
                }
                
                response = self.llm.call_llm([system_message, user_message])
                
                if response:
                    try:
                        # Parse the JSON response
                        try:
                            # Try direct parsing first
                            structure = json.loads(response)
                        except json.JSONDecodeError:
                            # Try to extract JSON from markdown code blocks
                            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                            if json_match:
                                structure = json.loads(json_match.group(1))
                            else:
                                logger.warning("Failed to parse LLM response as JSON")
                                structure = None
                        
                        if structure and isinstance(structure, list):
                            logger.info(f"Successfully extracted {len(structure)} chapters from TOC using LLM")
                            return {"source": "llm_toc_analysis", "chapters": structure}
                    except Exception as e:
                        logger.error(f"Error parsing book structure from LLM: {str(e)}")
        except Exception as e:
            logger.error(f"Error with LLM TOC analysis: {str(e)}")
        
        # APPROACH 3: Fall back to direct chapter marker detection in text (
        try:
            # Extract text if we haven't already
            if not 'text' in locals():
                text = self.extract_text_from_pdf(filename)
            
            logger.info("Falling back to regex-based chapter detection")
            
            # Try different chapter marker patterns
            chapter_patterns = [
                r'(?:CHAPTER|Chapter)\s+(\d+)[:\s]+([^\n]+)',  # Chapter with title
                r'(?:CHAPTER|Chapter)\s+(\d+)',  # Just chapter number
                r'(?:TEXT|Text)\s+1'  # First verse is often chapter marker
            ]
            
            for pattern in chapter_patterns:
                chapter_matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if chapter_matches:
                    logger.info(f"Found {len(chapter_matches)} chapter markers using pattern: {pattern}")
                    
                    chapters = []
                    for match in chapter_matches:
                        chapter_num = int(match.group(1)) if len(match.groups()) >= 1 else 1
                        chapter_name = match.group(2) if len(match.groups()) >= 2 else f"Chapter {chapter_num}"
                        
                        chapters.append({
                            "chapter_number": chapter_num,
                            "chapter_name": chapter_name,
                            "position": match.start()  # Store position for chapter boundaries
                        })
                    
                    if chapters:
                        # Sort by position in text
                        chapters.sort(key=lambda x: x.get("position", 0))
                        return {"source": "regex_chapter_markers", "chapters": chapters}
            
            # If nothing else worked, treat as a single chapter
            logger.warning("No chapter structure detected. Treating as a single chapter.")
            return {
                "source": "single_chapter_fallback",
                "chapters": [{
                    "chapter_number": 1,
                    "chapter_name": "Chapter 1",
                    "position": 0
                }]
            }
        
        except Exception as e:
            logger.error(f"Error with regex chapter detection: {str(e)}")
            # Last resort fallback
            return {
                "source": "error_fallback",
                "chapters": [{
                    "chapter_number": 1,
                    "chapter_name": "Chapter 1",
                    "position": 0
                }]
            }
    
    def identify_verse_chunks(self, chapter_text):
        """Identify text chunks that contain verses with multiple numbering formats."""
        logger.info("Identifying verse chunks")
        
        chunks = []
        
        # Strategy 1: Find verses by "TEXT N" markers
        text_markers = list(re.finditer(r'TEXT\s+(\d+)', chapter_text))
        if text_markers:
            for i, marker in enumerate(text_markers):
                verse_num = int(marker.group(1))
                start_pos = marker.start()
                
                # Determine end position (start of next verse or end of chapter)
                end_pos = text_markers[i+1].start() if i < len(text_markers)-1 else len(chapter_text)
                
                # Add context around the chunk
                start_with_context = max(0, start_pos - 50)
                chunk = chapter_text[start_with_context:end_pos]
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} verse chunks using TEXT markers")
            return chunks
        
        # Strategy 2: Find verses by Sanskrit verse numbers (॥N॥)
        sanskrit_markers = list(re.finditer(r'॥\s*(?:\d+|[\u0966-\u096F]+)\s*॥', chapter_text))
        if sanskrit_markers:
            # Process Sanskrit markers
            current_chunk_start = 0
            
            for marker in sanskrit_markers:
                marker_pos = marker.start()
                
                # Find an appropriate starting point before the marker
                # (either after previous verse or beginning of document)
                chunk_start = current_chunk_start
                
                # Find the end of this verse (including the marker)
                chunk_end = marker.end() + 300  # Include some context after the verse
                
                # Extract the chunk
                chunk = chapter_text[chunk_start:min(chunk_end, len(chapter_text))]
                chunks.append(chunk)
                
                # Update for next iteration
                current_chunk_start = marker.end()
            
            logger.info(f"Found {len(chunks)} verse chunks using Sanskrit verse numbers")
            return chunks
        
        # Strategy 3: Find verses by "Verse N" markers
        verse_markers = list(re.finditer(r'[Vv]erse\s+(\d+)', chapter_text))
        if verse_markers:
            for i, marker in enumerate(verse_markers):
                verse_num = int(marker.group(1))
                start_pos = marker.start()
                
                # Determine end position (start of next verse or end of chapter)
                end_pos = verse_markers[i+1].start() if i < len(verse_markers)-1 else len(chapter_text)
                
                # Add context around the chunk
                start_with_context = max(0, start_pos - 50)
                chunk = chapter_text[start_with_context:end_pos]
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} verse chunks using Verse markers")
            return chunks
        
        # Strategy 4: Find verses by TRANSLATION/PURPORT markers
        section_markers = list(re.finditer(r'(TRANSLATION|PURPORT)\s*\n', chapter_text))
        if section_markers:
            # Group markers by verse
            i = 0
            while i < len(section_markers) - 1:
                if section_markers[i].group(1) == "TRANSLATION" and section_markers[i+1].group(1) == "PURPORT":
                    # Found a TRANSLATION-PURPORT pair
                    translation_start = section_markers[i].start()
                    
                    # Find the beginning of the verse (go back to find Sanskrit)
                    verse_start = max(0, translation_start - 500)  # Look back up to 500 chars
                    
                    # End of verse is either next TRANSLATION or end of text
                    verse_end = section_markers[i+2].start() if i+2 < len(section_markers) else len(chapter_text)
                    
                    chunks.append(chapter_text[verse_start:verse_end])
                    i += 2  # Skip both markers we just processed
                else:
                    i += 1  # Move to next marker
            
            if chunks:
                logger.info(f"Found {len(chunks)} verse chunks using TRANSLATION/PURPORT markers")
                return chunks
        
        # Strategy 5: Fallback to sliding window if no verses found
        if not chunks:
            logger.warning("No verse markers found. Using sliding window approach.")
            chunk_size = 1500
            overlap = 700  # Large overlap to avoid splitting verses
            for i in range(0, len(chapter_text), chunk_size - overlap):
                chunk = chapter_text[i:i + chunk_size]
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks using sliding window")
        
        return chunks
    
    def validate_verse(self, verse_data):
        """Validate verse data with focus on Sanskrit quality."""
        # Ensure verse number is > 0
        if verse_data["verse"] <= 0:
            verse_data["verse"] = 1
        
        # Validate Sanskrit quality
        sanskrit = verse_data.get("sanskrit", "").strip()
        if sanskrit:
            # Remove all whitespace for character counting
            sanskrit_chars = re.sub(r'\s', '', sanskrit)
            # Sanskrit should have a reasonable number of characters
            if len(sanskrit_chars) < 10:
                logger.warning(f"Sanskrit verse too short ({len(sanskrit_chars)} chars): {sanskrit}")
                # Don't reject, but mark for potential review
                verse_data["review_sanskrit"] = True
            
            # Check for proper Sanskrit verse structure
            if not re.search(r'[।॥]', sanskrit) and len(sanskrit_chars) > 20:
                # If long Sanskrit without proper markers, try to fix
                verse_data["sanskrit"] = sanskrit + " ॥"
                logger.info("Added missing verse end marker to Sanskrit")
        
        # Check for minimum content
        has_content = any([
            verse_data.get("sanskrit", "").strip(),
            verse_data.get("transliteration", "").strip(),
            verse_data.get("translation", "").strip()
        ])
        
        if not has_content:
            logger.warning(f"Skipping verse with insufficient content: {verse_data}")
            return False
        
        return True
    
    def extract_verse_components(self, text_chunk):
        """Extract verse components with support for all verse number formats and positions."""
        logger.info("Using extract_verse_components method")
        
        result = {
            "verse": 1,  # Default
            "sanskrit": "",
            "transliteration": "",
            "translation": "",
            "purport": ""
        }
        
        # 1. Extract verse number using multiple patterns and positions
        verse_patterns = [
            # Western formats
            r'TEXT\s+(\d+)',
            r'[Vv]erse\s+(\d+)',
            # Sanskrit verse numbers at end of verse
            r'॥\s*(\d+)\s*॥',
            # Sanskrit numerals in Devanagari
            r'॥\s*([\u0966-\u096F]+)\s*॥'  # Unicode range for Devanagari digits
        ]
        
        for pattern in verse_patterns:
            verse_match = re.search(pattern, text_chunk)
            if verse_match:
                verse_str = verse_match.group(1)
                # Convert Sanskrit numerals if needed
                if re.match(r'[\u0966-\u096F]+', verse_str):
                    # Map Devanagari digits to Arabic numerals
                    digit_map = {
                        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
                        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
                    }
                    verse_num = ''.join(digit_map.get(c, c) for c in verse_str)
                    result["verse"] = int(verse_num)
                else:
                    result["verse"] = int(verse_str)
                break
        
        # 2. Extract Sanskrit text - use several strategies
        
        # Strategy A: Look for Sanskrit text with verse number at the end
        sanskrit_end_pattern = r'([\u0900-\u097F][\u0900-\u097F\s]+॥\s*(?:\d+|[\u0966-\u096F]+)\s*॥)'
        sanskrit_match = re.search(sanskrit_end_pattern, text_chunk)
        if sanskrit_match:
            result["sanskrit"] = sanskrit_match.group(1).strip()
        
        # Strategy B: Look for Sanskrit text after TEXT marker
        if not result["sanskrit"] and result["verse"] > 0:
            text_marker = re.search(fr'TEXT\s+{result["verse"]}', text_chunk)
            if text_marker:
                # Process text after marker line by line
                remaining_text = text_chunk[text_marker.end():]
                lines = remaining_text.split('\n')
                sanskrit_lines = []
                
                for line in lines:
                    # If line contains substantial Devanagari
                    if re.search(r'[\u0900-\u097F]{2,}', line):
                        sanskrit_lines.append(line.strip())
                    # Stop when we hit Latin text that looks like transliteration
                    elif sanskrit_lines and re.search(r'[a-z]{3,}.*[āīūṛṁṅñṭḍṇśṣḥ]', line, re.IGNORECASE):
                        break
                
                if sanskrit_lines:
                    result["sanskrit"] = ' '.join(sanskrit_lines)
        
        # Strategy C: Find any substantial Devanagari block
        if not result["sanskrit"]:
            # Look for any continuous Devanagari text
            devanagari_blocks = re.findall(r'((?:[\u0900-\u097F][\u0900-\u097F\s।॥]+){3,})', text_chunk)
            if devanagari_blocks:
                # Take the largest block that's likely to be the verse
                largest_block = max(devanagari_blocks, key=lambda x: len(re.sub(r'\s', '', x)))
                if len(re.sub(r'\s', '', largest_block)) > 5:  # At least 5 Devanagari characters
                    result["sanskrit"] = largest_block.strip()
        
        # 3. Extract transliteration
        # First try to find Latin text with diacritical marks after known Sanskrit text
        if result["sanskrit"]:
            # Find position after Sanskrit
            sanskrit_pos = text_chunk.find(result["sanskrit"]) + len(result["sanskrit"])
            # Find next section marker
            next_marker_pos = min(
                pos for pos in [
                    text_chunk.find("TRANSLATION", sanskrit_pos),
                    text_chunk.find("WORD-FOR-WORD", sanskrit_pos),
                    len(text_chunk)
                ] if pos > sanskrit_pos
            )
            
            if next_marker_pos > sanskrit_pos:
                # Extract text between Sanskrit and next marker
                between_text = text_chunk[sanskrit_pos:next_marker_pos].strip()
                
                # Look for transliteration (lines with diacritical marks)
                translit_lines = []
                for line in between_text.split('\n'):
                    line = line.strip()
                    # If line contains diacritical marks and Latin letters
                    if re.search(r'[a-z].*[āīūṛṁṅñṭḍṇśṣḥ]', line, re.IGNORECASE):
                        translit_lines.append(line)
                
                if translit_lines:
                    result["transliteration"] = ' '.join(translit_lines)
        
        # If transliteration not found, try other patterns
        if not result["transliteration"]:
            # Look for block between Sanskrit and TRANSLATION
            translation_pos = text_chunk.find("TRANSLATION")
            if translation_pos > 0:
                # Look before TRANSLATION for text with diacritical marks
                before_translation = text_chunk[:translation_pos]
                transliteration_blocks = re.findall(
                    r'([a-z][a-z\s\-—āīūṛṁṅñṭḍṇśṣḥ,.;:\'\"]+)', 
                    before_translation, 
                    re.IGNORECASE | re.DOTALL
                )
                
                # Find the block with most diacritical marks
                if transliteration_blocks:
                    best_block = max(
                        transliteration_blocks,
                        key=lambda x: len(re.findall(r'[āīūṛṁṅñṭḍṇśṣḥ]', x))
                    )
                    if re.search(r'[āīūṛṁṅñṭḍṇśṣḥ]', best_block):
                        result["transliteration"] = best_block.strip()
        
        # 4. Extract translation - between TRANSLATION and PURPORT markers
        translation_patterns = [
            r'TRANSLATION\s*\n+(.*?)(?=\s*\n+\s*PURPORT|TEXT\s+\d+|VERSE\s+\d+|$)',
            r'TRANSLATION[:\s-]+\n*(.*?)(?=\s*\n+\s*PURPORT|TEXT\s+\d+|VERSE\s+\d+|$)'
        ]
        
        for pattern in translation_patterns:
            translation_match = re.search(pattern, text_chunk, re.DOTALL)
            if translation_match:
                result["translation"] = translation_match.group(1).strip()
                break
        
        # 5. Extract purport - after PURPORT marker until next TEXT/VERSE or end
        purport_patterns = [
            r'PURPORT\s*\n+(.*?)(?=\s*\n+\s*TEXT\s+\d+|VERSE\s+\d+|CHAPTER|$)',
            r'PURPORT[:\s-]+\n*(.*?)(?=\s*\n+\s*TEXT\s+\d+|VERSE\s+\d+|CHAPTER|$)'
        ]
        
        for pattern in purport_patterns:
            purport_match = re.search(pattern, text_chunk, re.DOTALL)
            if purport_match:
                result["purport"] = purport_match.group(1).strip()
                break
        
        # Clean up extracted components
        for key in ["sanskrit", "transliteration", "translation", "purport"]:
            if isinstance(result[key], str):
                # Clean up whitespace
                result[key] = re.sub(r'\s+', ' ', result[key]).strip()
        
        return result
    
    def process_chapter(self, chapter_info):
        """Process a chapter to extract verses."""
        chapter_num = chapter_info["chapter_number"]
        chapter_name = chapter_info["chapter_name"]
        chapter_text = chapter_info["text"]
        
        logger.info(f"Processing Chapter {chapter_num}: {chapter_name}")
        
        # Identify verse chunks
        verse_chunks = self.identify_verse_chunks(chapter_text)
        
        # Process each verse chunk
        verses = []
        seen_signatures = set()  # Track verse signatures to avoid duplicates
        
        for i, chunk in enumerate(verse_chunks):
            if i > 0 and i % 10 == 0:
                logger.info(f"Processed {i}/{len(verse_chunks)} verse chunks in Chapter {chapter_num}")
            
            try:
                # Use only traditional method for extraction, no LLM
                verse_data = self.extract_verse_components(chunk)
                
                # Add chapter information
                verse_data["chapter"] = chapter_num
                verse_data["chapter_name"] = chapter_name
                
                # Validate the verse data
                if not self.validate_verse(verse_data):
                    continue
                
                # Check for duplicates
                verse_sig = f"{verse_data['chapter']}-{verse_data['verse']}"
                if verse_sig in seen_signatures:
                    continue
                
                seen_signatures.add(verse_sig)
                verses.append(verse_data)
                
            except Exception as e:
                logger.error(f"Error processing verse chunk: {str(e)}")
        
        # Sort verses by verse number
        verses.sort(key=lambda x: x["verse"])
        
        logger.info(f"Extracted {len(verses)} verses from Chapter {chapter_num}")
        return verses
    
    def validate_verse(self, verse_data):
        """Validate verse data to ensure it meets minimum requirements."""
        # Ensure verse number is > 0
        if verse_data["verse"] <= 0:
            verse_data["verse"] = 1
        
        # Check for minimum content
        has_content = any([
            verse_data.get("sanskrit", "").strip(),
            verse_data.get("transliteration", "").strip(),
            verse_data.get("translation", "").strip()
        ])
        
        if not has_content:
            logger.warning(f"Skipping verse with insufficient content: {verse_data}")
            return False
        
        return True
    
    def extract_toc(self, filename):
        """Extract table of contents from the text."""
        logger.info(f"Extracting TOC for {filename}")
        
        try:
            # First try to extract TOC from PDF metadata
            filepath = os.path.join(self.data_dir, filename)
            doc = fitz.open(filepath)
            toc = doc.get_toc()
            
            if toc:
                logger.info(f"Found TOC in PDF metadata with {len(toc)} items")
                formatted_toc = []
                
                for level, title, page in toc:
                    # Try to extract chapter information
                    chapter_match = re.search(r'(?:Chapter|CHAPTER)[:\s-]*(\d+)', title, re.IGNORECASE)
                    chapter_num = int(chapter_match.group(1)) if chapter_match else None
                    
                    formatted_toc.append({
                        "level": level,
                        "title": title,
                        "chapter": chapter_num,
                        "page": page
                    })
                
                doc.close()
                return formatted_toc
            
            doc.close()
            logger.info("No TOC found in PDF metadata, generating from processed chapters")
            
            # If no TOC found, we'll generate one from the processed chapters
            return None
            
        except Exception as e:
            logger.error(f"Error extracting TOC: {str(e)}")
            return None
    
    def create_indices(self, verses, filename):
        """Create index files for efficient retrieval."""
        base_filename = os.path.splitext(filename)[0]
        
        logger.info(f"Creating indices for {base_filename}")
        
        # 1. Create chapter-verse index
        chapter_verse_index = {}
        for verse in verses:
            chapter = verse["chapter"]
            verse_num = verse["verse"]
            
            if chapter not in chapter_verse_index:
                chapter_verse_index[chapter] = {}
            
            chapter_verse_index[chapter][str(verse_num)] = verse
        
        # Save chapter-verse index
        cv_index_path = os.path.join(self.output_dir, f"index_processed_{base_filename}.json")
        with open(cv_index_path, 'w', encoding='utf-8') as f:
            json.dump(chapter_verse_index, f, ensure_ascii=False, indent=2)
        
        # 2. Create keyword index
        keyword_index = {}
        for verse in verses:
            text = f"{verse.get('transliteration', '')} {verse.get('translation', '')}"
            words = re.findall(r'\b\w{3,}\b', text.lower())
            
            for word in words:
                if word not in keyword_index:
                    keyword_index[word] = []
                
                entry = [verse["chapter"], verse["verse"]]
                if entry not in keyword_index[word]:
                    keyword_index[word].append(entry)
        
        # Save keyword index
        kw_index_path = os.path.join(self.output_dir, f"keyword_index_processed_{base_filename}.json")
        with open(kw_index_path, 'w', encoding='utf-8') as f:
            json.dump(keyword_index, f, ensure_ascii=False, indent=2)
        
        # 3. Generate TOC from verses if none was extracted
        chapters = {}
        for verse in verses:
            chapter = verse["chapter"]
            chapter_name = verse.get("chapter_name", f"Chapter {chapter}")
            
            if chapter not in chapters:
                chapters[chapter] = {
                    "level": 1,
                    "title": chapter_name,
                    "chapter": chapter,
                    "page": 1,  # Default page
                    "verses": []
                }
            
            chapters[chapter]["verses"].append(verse["verse"])
        
        # Convert to sorted list
        toc = [chapters[num] for num in sorted(chapters.keys())]
        
        # Add verse counts
        for item in toc:
            item["verse_count"] = len(item["verses"])
            item["verse_range"] = f"1-{max(item.get('verses', [1]))}" if item.get("verses") else "1"
            item.pop("verses", None)  # Remove verse list from final TOC
        
        # Save TOC
        toc_path = os.path.join(self.output_dir, f"toc_{base_filename}.json")
        with open(toc_path, 'w', encoding='utf-8') as f:
            json.dump(toc, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created indices: chapter-verse, keyword, and TOC")
        
        return {
            "chapter_verse_index": cv_index_path,
            "keyword_index": kw_index_path,
            "toc": toc_path
        }
    
    def process_file(self, filename):
        """Process a religious text file from extraction to indexing."""
        try:
            logger.info(f"Processing file: {filename}")
            
            # 1. Extract text from PDF
            text = self.extract_text_from_pdf(filename)
            
            # 2. Extract book structure using our consolidated method
            book_structure = self.extract_book_structure(filename)
            logger.info(f"Book structure source: {book_structure.get('source', 'unknown')}")
            
            # 3. Use structure to create chapters with their text
            chapters = []
            chapter_list = book_structure.get("chapters", [])
            
            if chapter_list:
                # Sort by chapter number for consistency
                chapter_list = sorted(chapter_list, key=lambda x: x.get("chapter_number", 0))
                
                # For PDF metadata or LLM analysis that doesn't have positions, find them
                if book_structure.get("source") in ["pdf_metadata", "llm_toc_analysis"]:
                    # Find chapter markers in text
                    for chapter in chapter_list:
                        chapter_num = chapter.get("chapter_number", 0)
                        title = chapter.get("chapter_name", f"Chapter {chapter_num}")
                        
                        # Create search patterns for this chapter
                        patterns = [
                            fr'CHAPTER\s+{chapter_num}\b',
                            fr'Chapter\s+{chapter_num}\b',
                            fr'{re.escape(title)}'
                        ]
                        
                        # Find position in text
                        for pattern in patterns:
                            match = re.search(pattern, text)
                            if match:
                                chapter["position"] = match.start()
                                break
                        
                        # If no position found, try to estimate from page number
                        if "position" not in chapter and "page" in chapter:
                            # Rough estimate: assume pages are distributed evenly through text
                            page = chapter.get("page", 0)
                            chapter["position"] = min(int(len(text) * (page / 1000)), len(text) - 1)
                
                # Extract text for each chapter
                for i, chapter in enumerate(chapter_list):
                    chapter_num = chapter.get("chapter_number", 0)
                    chapter_name = chapter.get("chapter_name", f"Chapter {chapter_num}")
                    start_pos = chapter.get("position", 0)
                    
                    # Get end position (start of next chapter or end of text)
                    if i < len(chapter_list) - 1:
                        end_pos = chapter_list[i+1].get("position", len(text))
                    else:
                        end_pos = len(text)
                    
                    # Ensure we don't have invalid positions
                    if start_pos >= len(text):
                        start_pos = 0
                    if end_pos > len(text):
                        end_pos = len(text)
                    if end_pos <= start_pos:
                        end_pos = len(text)
                    
                    chapter_text = text[start_pos:end_pos]
                    
                    chapters.append({
                        "chapter_number": chapter_num,
                        "chapter_name": chapter_name,
                        "text": chapter_text
                    })
                    
                logger.info(f"Created {len(chapters)} chapters from book structure")
            else:
                # If no chapters found, treat entire text as one chapter
                logger.warning("No chapters found in book structure. Using entire text as Chapter 1")
                chapters = [{
                    "chapter_number": 1,
                    "chapter_name": "Chapter 1",
                    "text": text
                }]
            
            # 4. Process each chapter to extract verses
            all_verses = []
            for chapter in chapters:
                chapter_verses = self.process_chapter(chapter)
                all_verses.extend(chapter_verses)

            # If didn't extract enough verses, try chapter-by-chapter with a different approach
            if len(all_verses) < 10 and len(chapters) > 0:
                logger.warning(f"Only extracted {len(all_verses)} verses. Trying backup approach.")
                
                for chapter in chapters:
                    # Use a simpler verse splitting approach
                    chapter_text = chapter["text"]
                    simple_chunks = []
                    
                    # Look for verse markers
                    verse_markers = re.finditer(r'(?:TEXT|VERSE)\s+(\d+)', chapter_text)
                    positions = [(int(m.group(1)), m.start()) for m in verse_markers]
                    
                    # Sort by verse number
                    positions.sort(key=lambda x: x[0])
                    
                    if positions:
                        # Create chunks based on verse marker positions
                        for i, (verse_num, pos) in enumerate(positions):
                            start = max(0, pos - 50)
                            end = positions[i+1][1] - 50 if i < len(positions) - 1 else len(chapter_text)
                            chunk = chapter_text[start:end]
                            
                            # Create simple verse entry
                            verse_data = {
                                "verse": verse_num,
                                "chapter": chapter["chapter_number"],
                                "chapter_name": chapter["chapter_name"],
                                "sanskrit": "",
                                "transliteration": "",
                                "translation": "",
                                "purport": chunk  # Store whole chunk as purport for now
                            }
                            
                            all_verses.append(verse_data)
                
                # Re-sort verses
                all_verses.sort(key=lambda x: (x["chapter"], x["verse"]))
                logger.info(f"Backup extraction produced {len(all_verses)} verses")
            
            # 5. Sort verses by chapter and verse number
            all_verses.sort(key=lambda x: (x["chapter"], x["verse"]))
            
            # 6. Save processed verses
            base_filename = os.path.splitext(filename)[0]
            output_path = os.path.join(self.output_dir, f"processed_{base_filename}.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_verses, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(all_verses)} verses to {output_path}")
            
            # 7. Create indices
            indices = self.create_indices(all_verses, filename)
            
            # 8. Save original TOC if it was in PDF metadata
            if book_structure.get("source") == "pdf_metadata":
                toc_path = os.path.join(self.output_dir, f"toc_{base_filename}_from_pdf.json")
                with open(toc_path, 'w', encoding='utf-8') as f:
                    json.dump(book_structure["chapters"], f, ensure_ascii=False, indent=2)
            
            return {
                "verses": len(all_verses),
                "chapters": len(chapters),
                "output_file": output_path,
                "indices": indices
            }
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise