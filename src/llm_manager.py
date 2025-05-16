import os
import time  # Add this import
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMManager:
    """Singleton class to manage LLM instance that can be used across different files."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, api_key=None, model_name="llama3-8b-8192"):
        """Get or create the LLM singleton instance."""
        if cls._instance is None:
            cls._instance = cls(api_key, model_name)
        return cls._instance
    
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        """Initialize the LLM with API key and model name."""
        # Only allow initialization through get_instance
        if LLMManager._instance is not None:
            raise RuntimeError("Use LLMManager.get_instance() to get an instance")
        
        self.model_name = model_name  # Fix: Use model_name instead of model
        self.api_keys = []  # Store keys directly
        self.api_clients = []
        self.current_client_index = 0
        
        # Track API call timestamps for rate limiting
        self.call_timestamps = {}
        self.max_rpm = 25  # Max requests per minute
        
        # Collect API keys from environment variables
        if api_key:
            self.api_keys.append(api_key)
        if "GROQ_API_KEY" in os.environ:
            self.api_keys.append(os.environ["GROQ_API_KEY"])
        if "GROQ_API_KEY2" in os.environ:
            self.api_keys.append(os.environ["GROQ_API_KEY2"])
        if "GROQ_API_KEY3" in os.environ:
            self.api_keys.append(os.environ["GROQ_API_KEY3"])
        
        # Initialize call timestamps
        for key in self.api_keys:
            self.call_timestamps[key] = []
            
        # Import Groq
        try:
            from groq import Groq
            self.Groq = Groq
            
            # Initialize a client for each API key
            for key in self.api_keys:
                try:
                    client = Groq(api_key=key, max_retries=0)
                    self.api_clients.append(client)
                except Exception as e:
                    logger.error(f"Error initializing Groq client: {str(e)}")
        except ImportError:
            logger.warning("Groq library not found. Please install with 'pip install groq'")
            self.Groq = None
        
        if not self.api_clients:
            logger.warning("No API keys available or all client initializations failed")
        else:
            logger.info(f"LLMManager initialized with {len(self.api_clients)} API clients and model: {model_name}")
    
    def get_key(self):
        """Get the next available API key respecting rate limits."""
        if not self.api_keys:
            logger.warning("No API keys available")
            return None
            
        # Check if any key is available under rate limits
        now = time.time()
        for key in self.api_keys:
            # Remove timestamps older than 1 minute
            self.call_timestamps[key] = [
                ts for ts in self.call_timestamps[key] 
                if now - ts < 60
            ]
            
            # If key has capacity, return it
            if len(self.call_timestamps[key]) < self.max_rpm:
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
            wait_time = 60 - (now - min_time) + 0.5  # Add a small buffer
            logger.info(f"All API keys at capacity. Waiting {wait_time:.2f}s for {min_key}")
            time.sleep(wait_time)
            self.call_timestamps[min_key] = [
                ts for ts in self.call_timestamps[min_key] 
                if now + wait_time - ts < 60
            ]
            return min_key
        
        return None
    
    def record_usage(self, key):
        """Record usage of a key to track rate limits."""
        if key in self.call_timestamps:
            self.call_timestamps[key].append(time.time())
            
    def call_llm(self, messages, temperature=0, retries=2):
        """Call LLM with automatic key rotation and rate limiting."""
        if not hasattr(self, 'Groq') or self.Groq is None:
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
        keys_tried = 0
        total_attempts = 0
        max_attempts = retries * len(self.api_keys) + 1
        
        while keys_tried < len(self.api_keys) and total_attempts < max_attempts:
            try:
                api_key = self.get_key()
                if not api_key:
                    logger.error("No API key available")
                    return None
                
                keys_tried += 1
                total_attempts += 1
                
                # Create a fresh client with max_retries=0
                client = self.Groq(api_key=api_key, max_retries=0)
                
                logger.info(f"Trying API key {keys_tried}/{len(self.api_keys)} - Attempt {total_attempts}/{max_attempts}")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                
                self.record_usage(api_key)
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"LLM call failed: {error_str}")
                
                # If hit rate limit (429), try next key immediately
                if "rate limit" in error_str.lower() or "429" in error_str:
                    logger.warning(f"Rate limit hit. Trying next API key.")
                    continue
                elif "timeout" in error_str.lower() or "connection" in error_str.lower():
                    # Network errors - add exponential backoff
                    wait_time = min(2 ** (total_attempts - 1), 30)  # Max 30 seconds
                    logger.info(f"Network error. Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    # For other errors, stop trying
                    logger.error(f"Unexpected error: {error_str}")
                    return None
        
        logger.warning(f"All API keys exhausted after {total_attempts} attempts. Aborting LLM call.")
        return None