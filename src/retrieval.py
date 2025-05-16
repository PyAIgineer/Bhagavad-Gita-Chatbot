import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional
import logging

# Import LLM manager directly
from llm_manager import LLMManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexLoader:
    """Handles loading of index files"""
    def __init__(self, data_dir: str = "processed_data"):
        """Initialize with data directory"""
        self.data_dir = data_dir
    
    def load_index(self, filepath: str) -> Dict:
        """Load index from file."""
        full_path = os.path.join(self.data_dir, filepath)
        logger.info(f"Loading index from {full_path}")
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                index = json.load(file)
            logger.info(f"Successfully loaded index from {full_path}")
            return index
        except Exception as e:
            logger.error(f"Error loading index from {full_path}: {str(e)}")
            return {}

class QueryParser:
    """Handles parsing user queries for verse information"""
    def __init__(self):
        """Initialize with direct access to LLM manager"""
        # Get LLM instance directly
        self.llm = LLMManager.get_instance()
    
    def parse_with_regex(self, query: str) -> Dict[str, Any]:
        """
        Use regex to extract chapter, verse, and section from query.
        Fallback method when LLM is unavailable.
        """
        logger.info(f"Parsing query with regex: {query}")
        
        # Extract chapter and verse using various patterns
        chapter_match = re.search(r'chapter\s+(\d+)', query.lower())
        verse_match = re.search(r'verse\s+(\d+)', query.lower())
        
        # Alternative pattern for "Chapter.Verse" format or "X.Y" format
        combined_match = re.search(r'(\d+)[.:,]\s*(\d+)', query)
        
        # Extract requested section
        section_match = re.search(r'(sanskrit|transliteration|translation|purport)', query.lower())
        
        result = {"chapter": None, "verse": None, "section": "all"}
        
        if chapter_match:
            result["chapter"] = int(chapter_match.group(1))
        
        if verse_match:
            result["verse"] = int(verse_match.group(1))
        elif combined_match:
            # If we found a combined chapter.verse format
            # And we don't already have a chapter from an explicit "chapter X" match
            if result["chapter"] is None:
                result["chapter"] = int(combined_match.group(1))
                result["verse"] = int(combined_match.group(2))
        
        if section_match:
            result["section"] = section_match.group(1)
        
        logger.info(f"Regex parsing result: {result}")
        return result
    
    def extract_verse_info(self, query: str) -> Dict[str, Any]:
        """Extract chapter, verse, and section information from a query using LLM."""
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
        
        content = self.llm.call_llm(messages)
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
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a query to extract verse information.
        Uses LLM if available, otherwise falls back to regex.
        """
        # Try LLM parsing first
        result = self.extract_verse_info(query)
        if result and (result.get("chapter") is not None or result.get("verse") is not None):
            return result
        
        # Fall back to regex
        return self.parse_with_regex(query)

class VerseRetriever:
    """Handles retrieving verses from indices"""
    def __init__(self, index_loader: IndexLoader):
        """Initialize with index loader"""
        self.index_loader = index_loader
        self.chapter_verse_index = None
        self.keyword_index = None
    
    def load_indices(self, chapter_verse_index_file: str, keyword_index_file: str):
        """Load the necessary indices"""
        self.chapter_verse_index = self.index_loader.load_index(chapter_verse_index_file)
        self.keyword_index = self.index_loader.load_index(keyword_index_file)
    
    def retrieve_verse(self, chapter: int, verse: int, section: str = "all") -> Dict[str, Any]:
        """Retrieve a specific verse by chapter and verse number."""
        logger.info(f"Retrieving verse: Chapter {chapter}, Verse {verse}, Section {section}")
        
        try:
            if not self.chapter_verse_index:
                logger.error("Chapter-verse index not loaded")
                return {"error": "Index not loaded"}
                
            # Get chapter dict
            chapter_dict = self.chapter_verse_index.get(str(chapter))
            if not chapter_dict:
                logger.warning(f"Chapter {chapter} not found")
                return {"error": f"Chapter {chapter} not found"}
            
            # Get verse dict
            verse_dict = chapter_dict.get(str(verse))
            if not verse_dict:
                logger.warning(f"Verse {verse} not found in Chapter {chapter}")
                return {"error": f"Verse {verse} not found in Chapter {chapter}"}
            
            # Return specific section or all
            if section == "all":
                logger.info(f"Retrieved complete verse {chapter}.{verse}")
                return verse_dict
            elif section in verse_dict:
                logger.info(f"Retrieved {section} for verse {chapter}.{verse}")
                # Include chapter and verse in the response
                return {
                    "chapter": chapter,
                    "verse": verse, 
                    section: verse_dict[section]
                }
            else:
                logger.warning(f"Section {section} not found for verse {chapter}.{verse}")
                return {"error": f"Section {section} not found for verse {chapter}.{verse}"}
        
        except Exception as e:
            logger.error(f"Error retrieving verse: {str(e)}")
            return {"error": str(e)}
    
    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search verses by keyword."""
        logger.info(f"Searching for keyword: {keyword}")
        
        if not self.keyword_index:
            logger.error("Keyword index not loaded")
            return []
            
        keyword = keyword.lower()
        
        results = []
        for word, chapter_verse_pairs in self.keyword_index.items():
            if keyword in word:
                for chapter, verse in chapter_verse_pairs[:limit]:
                    verse_data = self.retrieve_verse(chapter, verse)
                    if "error" not in verse_data:
                        results.append(verse_data)
                    
                    if len(results) >= limit:
                        break
                        
            if len(results) >= limit:
                break
        
        logger.info(f"Found {len(results)} results for keyword '{keyword}'")
        return results

class QueryProcessor:
    """Main processor for handling user queries"""
    def __init__(self, 
                 data_dir: str = "processed_data", 
                 index_file: str = "index_processed_verses.json",
                 keyword_index_file: str = "keyword_index_processed_verses.json"):
        """Initialize the query processor with all necessary components"""
        
        # Initialize other components
        self.index_loader = IndexLoader(data_dir)
        self.verse_retriever = VerseRetriever(self.index_loader)
        self.query_parser = QueryParser()  # Add this line to initialize QueryParser
        
        # Load indices
        self.verse_retriever.load_indices(index_file, keyword_index_file)
        
        logger.info(f"QueryProcessor initialized with index files: {index_file}, {keyword_index_file}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return the appropriate content.
        """
        logger.info(f"Processing query: {query}")
        
        # Check if it's a keyword search query
        if "search" in query.lower() or "find" in query.lower():
            # Extract keyword
            keyword_match = re.search(r'(search|find|about)\s+(\w+)', query.lower())
            if keyword_match:
                keyword = keyword_match.group(2)
                return {"type": "keyword_search", "results": self.verse_retriever.search_by_keyword(keyword)}
        
        # Try to parse as chapter-verse query
        try:
            parsed = self.query_parser.parse_query(query)
            
            chapter = parsed.get("chapter")
            verse = parsed.get("verse")
            section = parsed.get("section", "all")
            
            # If we have both chapter and verse, retrieve the verse
            if chapter is not None and verse is not None:
                result = self.verse_retriever.retrieve_verse(chapter, verse, section)
                return {"type": "verse_lookup", "result": result}
            else:
                logger.warning(f"Could not extract chapter and verse from query: {query}")
                return {"type": "error", "message": "Could not identify chapter and verse. Please try a different query format."}
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"type": "error", "message": str(e)}