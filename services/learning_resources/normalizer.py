# services/learning_resources/normalizer.py
# Unified resource normalization across multiple sources

from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib
from utils.logger import Logger

logger = Logger(__name__)


@dataclass
class LearningResource:
    """
    Unified learning resource schema.
    
    All learning resources from different sources are normalized to this structure
    for consistent handling, ranking, and storage.
    """
    # Core Fields
    id: str  # Format: "{source}_{hash(url)}" - unique resource identifier
    title: str  # Resource title
    description: str  # Summary or description
    source: str  # youtube | geeksforgeeks | coursera | edx | etc.
    url: str  # Direct link to resource
    
    # Content Metadata
    duration_seconds: int  # Total duration in seconds (0 if unknown)
    author: str = ""  # Creator/instructor name
    difficulty: str = "intermediate"  # beginner | intermediate | advanced
    language: str = "en"  # ISO language code
    cost: str = "free"  # free | paid | freemium
    
    # Discovery Metadata
    published_date: Optional[str] = None  # ISO format: "2024-01-15"
    last_updated: Optional[str] = None  # ISO format
    tags: list = field(default_factory=list)  # [topic1, topic2, ...]
    
    # Quality Metrics
    popularity_score: float = 0.5  # 0.0-1.0 (view count normalized)
    rating: float = 0.0  # 0.0-5.0 (user rating if available)
    rating_count: int = 0  # Number of ratings
    
    # Relevance (computed during ranking)
    relevance_score: float = 0.0  # 0.0-1.0 (set during semantic ranking)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_metadata: Dict[str, Any] = field(default_factory=dict)  # Raw source-specific data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_embedding_text(self) -> str:
        """
        Generate text for embedding generation.
        Combines title, description, and tags for semantic search.
        """
        parts = [
            self.title,
            self.description,
            " ".join(self.tags) if self.tags else "",
            self.author,
        ]
        return " | ".join(p.strip() for p in parts if p.strip())


class ResourceNormalizer:
    """
    Converts resources from different sources to unified LearningResource format.
    
    Handles normalization, validation, and ID generation.
    """
    
    # Source authority scores (used in ranking)
    SOURCE_AUTHORITY = {
        "youtube": 0.80,
        "geeksforgeeks": 0.85,
        "coursera": 0.95,
        "edx": 0.95,
        "udemy": 0.70,
        "linkedin_learning": 0.80,
        "microsoft_learn": 0.92,
        "aws_skill_builder": 0.90,
        "google_cloud_skills": 0.90,
    }
    
    @staticmethod
    def _generate_id(source: str, url: str) -> str:
        """Generate unique ID from source and URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{source}_{url_hash}"
    
    @staticmethod
    def _parse_duration(duration_input: Any) -> int:
        """
        Parse duration from various formats to seconds.
        
        Handles:
        - Integer (seconds)
        - String "10:30" (minutes:seconds)
        - String "1h 30m" (human readable)
        - None (returns 0)
        """
        if duration_input is None:
            return 0
        
        if isinstance(duration_input, int):
            return max(0, duration_input)
        
        if isinstance(duration_input, str):
            duration_input = duration_input.strip().lower()
            
            # Handle "MM:SS" format
            if ":" in duration_input and not ("h" in duration_input or "m" in duration_input):
                try:
                    parts = duration_input.split(":")
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        return minutes * 60 + seconds
                except Exception:
                    pass
            
            # Handle "Xh Ym" or "Xm" format
            total_seconds = 0
            try:
                if "h" in duration_input:
                    hours = int(duration_input.split("h")[0].strip())
                    total_seconds += hours * 3600
                    duration_input = duration_input.split("h")[1] if "h" in duration_input else ""
                
                if "m" in duration_input:
                    minutes = int(duration_input.split("m")[0].strip())
                    total_seconds += minutes * 60
                
                if total_seconds > 0:
                    return total_seconds
            except Exception:
                pass
        
        return 0
    
    @staticmethod
    def _normalize_difficulty(difficulty: Any) -> str:
        """Normalize difficulty to standard format."""
        if not difficulty:
            return "intermediate"
        
        diff_str = str(difficulty).strip().lower()
        
        # Map variants to standard
        if any(x in diff_str for x in ["beginner", "introductory", "basic", "level 1", "junior"]):
            return "beginner"
        elif any(x in diff_str for x in ["advanced", "expert", "level 3", "senior"]):
            return "advanced"
        else:
            return "intermediate"
    
    @staticmethod
    def _normalize_cost(cost: Any) -> str:
        """Normalize cost to standard format."""
        if not cost:
            return "free"
        
        cost_str = str(cost).strip().lower()
        
        if any(x in cost_str for x in ["paid", "premium", "subscription", "$", "₹"]):
            return "paid"
        elif any(x in cost_str for x in ["free"]):
            return "free"
        else:
            return "freemium"
    
    @staticmethod
    def _normalize_tags(tags: Any) -> list:
        """Normalize tags to list of strings."""
        if not tags:
            return []
        
        if isinstance(tags, str):
            # Split by comma or space
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        elif isinstance(tags, (list, tuple)):
            tags = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tags = []
        
        # Remove duplicates, preserve order
        seen = set()
        result = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                result.append(tag)
        
        return result[:20]  # Max 20 tags
    
    @staticmethod
    def normalize_youtube(raw: Dict[str, Any]) -> LearningResource:
        """Normalize YouTube video to LearningResource."""
        title = str(raw.get("title", "Untitled")).strip()
        url = str(raw.get("url") or raw.get("video_url") or "").strip()
        
        if not url:
            logger.warning("YouTube resource missing URL")
            url = ""
        
        description = str(raw.get("description", "")).strip()[:500]
        author = str(raw.get("channel") or raw.get("author", "")).strip()
        duration = ResourceNormalizer._parse_duration(raw.get("duration"))
        
        # YouTube metrics
        view_count = raw.get("view_count", 0)
        popularity = min(1.0, (view_count or 0) / 1000000)  # Normalize by 1M views
        
        tags = raw.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags = ResourceNormalizer._normalize_tags(tags) or ["video", "tutorial"]
        
        resource_id = ResourceNormalizer._generate_id("youtube", url)
        
        return LearningResource(
            id=resource_id,
            title=title,
            description=description,
            source="youtube",
            url=url,
            duration_seconds=duration,
            author=author,
            difficulty=ResourceNormalizer._normalize_difficulty(raw.get("difficulty")),
            cost="free",
            popularity_score=popularity,
            rating=float(raw.get("rating", 0.0)),
            published_date=raw.get("published_date"),
            tags=tags,
            source_metadata={
                "video_id": raw.get("video_id"),
                "channel_id": raw.get("channel_id"),
                "view_count": view_count,
            }
        )
    
    @staticmethod
    def normalize_geeksforgeeks(raw: Dict[str, Any]) -> LearningResource:
        """Normalize GeeksforGeeks article to LearningResource."""
        title = str(raw.get("title", "Untitled")).strip()
        url = str(raw.get("url") or raw.get("link", "")).strip()
        
        if not url:
            logger.warning("GeeksForGeeks resource missing URL")
            url = ""
        
        description = str(raw.get("description", "")).strip()[:500]
        author = str(raw.get("author", "GeeksforGeeks")).strip()
        
        # Estimate duration from word count (assume 200 words per minute)
        content = str(raw.get("content", "")).strip()
        word_count = len(content.split())
        duration = max(300, (word_count // 200) * 60)  # Min 5 minutes
        
        tags = raw.get("tags", []) or ["programming", "tutorial"]
        tags = ResourceNormalizer._normalize_tags(tags)
        
        resource_id = ResourceNormalizer._generate_id("geeksforgeeks", url)
        
        return LearningResource(
            id=resource_id,
            title=title,
            description=description,
            source="geeksforgeeks",
            url=url,
            duration_seconds=duration,
            author=author,
            difficulty=ResourceNormalizer._normalize_difficulty(raw.get("difficulty")),
            cost="free",
            popularity_score=float(raw.get("popularity", 0.7)),
            published_date=raw.get("published_date"),
            tags=tags,
            source_metadata={
                "word_count": word_count,
                "practice_problems": raw.get("practice_problems", []),
            }
        )
    
    @staticmethod
    def normalize_generic(
        title: str,
        url: str,
        source: str,
        description: str = "",
        author: str = "",
        duration: int = 0,
        difficulty: str = "intermediate",
        cost: str = "free",
        tags: list = None,
        **kwargs
    ) -> LearningResource:
        """
        Generic normalization for any source.
        
        Useful for new sources that don't have specialized normalizers yet.
        """
        if not url:
            logger.warning(f"{source} resource missing URL")
            url = ""
        
        resource_id = ResourceNormalizer._generate_id(source, url)
        
        return LearningResource(
            id=resource_id,
            title=title.strip(),
            description=str(description).strip()[:500],
            source=source,
            url=url,
            duration_seconds=ResourceNormalizer._parse_duration(duration),
            author=str(author).strip(),
            difficulty=ResourceNormalizer._normalize_difficulty(difficulty),
            cost=ResourceNormalizer._normalize_cost(cost),
            tags=ResourceNormalizer._normalize_tags(tags),
            popularity_score=float(kwargs.get("popularity_score", 0.5)),
            rating=float(kwargs.get("rating", 0.0)),
            published_date=kwargs.get("published_date"),
            source_metadata=kwargs.get("source_metadata", {}),
        )
    
    @staticmethod
    def normalize_batch(
        resources: list,
        source: str,
        normalizer_func=None
    ) -> list:
        """
        Normalize a batch of resources.
        
        Args:
            resources: List of raw resource dicts
            source: Source type (youtube, geeksforgeeks, etc.)
            normalizer_func: Optional custom normalizer function
        
        Returns:
            List of LearningResource objects
        """
        if normalizer_func is None:
            if source == "youtube":
                normalizer_func = ResourceNormalizer.normalize_youtube
            elif source == "geeksforgeeks":
                normalizer_func = ResourceNormalizer.normalize_geeksforgeeks
            else:
                normalizer_func = ResourceNormalizer.normalize_generic
        
        normalized = []
        for raw in resources:
            try:
                resource = normalizer_func(raw) if source != "generic" else normalizer_func(source=source, **raw)
                if resource and resource.title and resource.url:
                    normalized.append(resource)
            except Exception as e:
                logger.warning(f"Failed to normalize {source} resource: {str(e)}")
                continue
        
        return normalized
