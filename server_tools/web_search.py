import logging
import time
import random
from duckduckgo_search import DDGS
from typing import List, Dict, Any
import json
import os
from diskcache import Cache

# Import environment variables for rate limiting
try:
    from environment import WEB_SEARCH_MIN_DELAY, WEB_SEARCH_MAX_DELAY, WEB_SEARCH_CACHE_TTL
except ImportError:
    # Fallback defaults if environment.py is not available
    WEB_SEARCH_MIN_DELAY = 2.0
    WEB_SEARCH_MAX_DELAY = 30.0
    WEB_SEARCH_CACHE_TTL = 3600

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self, max_results=5, cache_dir=None):
        self.ddgs = DDGS()
        self.max_results = int(max_results)  # Ensure it's always an integer
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "web_search_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self.cache = Cache(cache_dir)
            logger.info(f"WebSearcher cache initialized at: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}. Searches will not be cached.")
            self.cache = None
        
        # Rate limiting parameters
        self.min_delay = WEB_SEARCH_MIN_DELAY  # Minimum delay between requests
        self.max_delay = WEB_SEARCH_MAX_DELAY  # Maximum delay for exponential backoff
        self.last_request_time = 0
        self.consecutive_failures = 0
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting with exponential backoff"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Calculate delay based on consecutive failures
        if self.consecutive_failures > 0:
            # Exponential backoff: 2^failures * base_delay + random jitter
            delay = min(self.max_delay, (2 ** self.consecutive_failures) * self.min_delay)
            delay += random.uniform(0, delay * 0.1)  # Add 10% jitter
            logger.info(f"Rate limit backoff: waiting {delay:.1f}s (failure #{self.consecutive_failures})")
        else:
            delay = self.min_delay
        
        # Ensure minimum delay since last request
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key for search query"""
        return f"websearch:{query.lower().strip()}:{max_results}"
    
    def search(self, query: str, max_results=None) -> List[Dict[str, Any]]:
        """Perform a web search using DuckDuckGo with rate limiting and caching"""
        try:
            logger.info(f"Performing web search for: {query}")
            
            # Use provided max_results or fall back to instance default
            limit = int(max_results) if max_results is not None else self.max_results
            
            # Check cache first
            cache_key = self._get_cache_key(query, limit)
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info(f"Web search cache hit for query: {query[:50]}...")
                    return cached_result
            
            # Rate limiting
            self._wait_for_rate_limit()
            
            # Perform the search with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Search attempt {attempt + 1}/{max_retries}")
                    
                    # Perform the search
                    results = list(self.ddgs.text(
                        query,
                        region="wt-wt",
                        safesearch="Off",
                        timelimit="m",  # Last month
                    ))
                    
                    # Limit results
                    results = results[:limit]
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "title": result.get("title", ""),
                            "snippet": result.get("body", ""),
                            "link": result.get("href", ""),
                            "source": "web_search"
                        })
                    
                    # Success - reset failure counter
                    self.consecutive_failures = 0
                    
                    # Cache the result
                    if self.cache:
                        # Cache using configured TTL
                        self.cache.set(cache_key, formatted_results, expire=WEB_SEARCH_CACHE_TTL)
                        logger.debug(f"Cached web search results for: {query[:50]}...")
                    
                    logger.info(f"Found {len(formatted_results)} web search results")
                    return formatted_results
                    
                except Exception as search_error:
                    error_msg = str(search_error).lower()
                    
                    if "ratelimit" in error_msg or "202" in error_msg:
                        self.consecutive_failures += 1
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                                     f"Consecutive failures: {self.consecutive_failures}")
                        
                        if attempt < max_retries - 1:
                            # Exponential backoff before retry
                            backoff_delay = min(30, (2 ** attempt) * 5)
                            logger.info(f"Waiting {backoff_delay}s before retry...")
                            time.sleep(backoff_delay)
                            continue
                        else:
                            logger.error("Max retries reached for rate limited search")
                            break
                    else:
                        # Non-rate-limit error
                        logger.error(f"Search error (attempt {attempt + 1}): {search_error}")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Brief delay before retry
                            continue
                        else:
                            break
            
            # All retries failed
            logger.error(f"Web search failed after {max_retries} attempts")
            return []
            
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Error performing web search: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear the search cache"""
        if self.cache:
            try:
                self.cache.clear()
                logger.info("Web search cache cleared")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        if self.cache:
            try:
                return {
                    "cache_size": len(self.cache),
                    "cache_directory": self.cache.directory
                }
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                return {"error": str(e)}
        else:
            return {"cache": "disabled"}
    
    def __del__(self):
        """Cleanup cache on destruction"""
        if hasattr(self, 'cache') and self.cache:
            try:
                self.cache.close()
            except:
                pass 