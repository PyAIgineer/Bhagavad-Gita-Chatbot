import os
import json
import re
import logging
from typing import Dict, Any, List, Optional

# Import the LLM Manager
from llm_manager import LLMManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMServices:
    """Provides all LLM-related functionality for the application."""
    
    def __init__(self, model_name="llama3-8b-8192", api_key=None):
        """Initialize with the LLM Manager singleton."""
        # Get LLM instance from manager
        self.llm_manager = LLMManager.get_instance(api_key, model_name)
        logger.info(f"LLMServices initialized with model: {model_name}")
    
    def extract_verse_info(self, query: str) -> Dict[str, Any]:
        """Extract chapter, verse, and section information from a query."""
        logger.info(f"Extracting verse info from query: {query}")
        
        system_message = "You extract chapter, verse, and section information from queries about religious texts."
        user_message = f"""
        Extract the chapter number, verse number, and requested section (sanskrit, transliteration, translation, purport) from this query: 
        "{query}" 
        
        If no chapter/verse is mentioned, say "NO VERSE FOUND". 
        If no section is specified, assume "all".
        
        Return your answer in JSON format: 
        {{
            "chapter": <number or null>, 
            "verse": <number or null>, 
            "section": "all" / "sanskrit" / "transliteration" / "translation" / "purport"
        }}
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        content = self.llm_manager.call_llm(messages)
        if not content:
            return {"chapter": None, "verse": None, "section": "all"}
        
        # Extract JSON from the response
        json_match = re.search(r'{.*}', content, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                logger.info(f"LLM parsing result: {parsed_json}")
                
                # Convert to expected format
                result = {
                    "chapter": parsed_json.get("chapter"),
                    "verse": parsed_json.get("verse"),
                    "section": parsed_json.get("section", "all")
                }
                
                return result
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
        
        return {"chapter": None, "verse": None, "section": "all"}
    
    def extract_book_structure(self, text_sample: str) -> Dict[str, Any]:
        """
        Use LLM to extract book structure information from text.
        This is used when table of contents or index information is found.
        """
        logger.info("Extracting book structure using LLM")
        
        system_message = "You are an expert in analyzing religious texts and extracting their structure."
        user_message = f"""
        Analyze this sample from a religious book and extract its structure information:
        
        ```
        {text_sample[:2000]}  # Use first 2000 chars as sample
        ```
        
        Please identify:
        1. The overall book name/title
        2. How chapters are organized
        3. How verses are numbered and organized
        4. Any other structural elements (sections, parts, etc.)
        
        Return your analysis in JSON format:
        {{
            "book_title": "title if identified",
            "chapter_pattern": "how chapters are marked",
            "verse_pattern": "how verses are marked",
            "structure_notes": "any other observations about the text structure"
        }}
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        content = self.llm_manager.call_llm(messages)
        if not content:
            return {}
        
        # Extract JSON from the response
        json_match = re.search(r'{.*}', content, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                logger.info(f"LLM book structure result: {parsed_json}")
                return parsed_json
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
        
        return {}