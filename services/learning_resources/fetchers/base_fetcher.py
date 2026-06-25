# services/learning_resources/fetchers/base_fetcher.py
# Abstract base class for resource fetchers

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from utils.logger import Logger

logger = Logger(__name__)


class BaseFetcher(ABC):
    """
    Abstract base class for learning resource fetchers.
    
    All fetchers from different sources (YouTube, GeeksforGeeks, Coursera, etc.)
    inherit from this class and implement the search() method.
    
    The fetcher returns raw dictionaries which are then normalized
    by ResourceNormalizer to a unified schema.
    """
    
    def __init__(self, source: str):
        """
        Initialize fetcher.
        
        Args:
            source: Source identifier (e.g., "youtube", "geeksforgeeks")
        """
        self.source = source
        logger.info(f"{self.__class__.__name__} initialized for source: {source}")
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for resources matching the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            **kwargs: Additional source-specific parameters
        
        Returns:
            List of raw resource dictionaries (source-specific format)
            Each dict will be normalized by ResourceNormalizer.normalize_*()
        
        Raises:
            Exception: On network or API errors
        """
        pass
    
    def validate_search_query(self, query: str) -> bool:
        """Validate search query before processing."""
        if not query:
            logger.warning("Empty search query")
            return False
        
        if len(query) > 500:
            logger.warning(f"Query too long: {len(query)} chars")
            return False
        
        return True
