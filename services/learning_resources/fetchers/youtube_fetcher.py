# services/learning_resources/fetchers/youtube_fetcher.py
# YouTube resource fetcher — real retrieval only, no mock data

import asyncio
import re
import json
from typing import List, Dict, Any, Optional
from .base_fetcher import BaseFetcher
from utils.logger import Logger

logger = Logger(__name__)


class YouTubeFetcher(BaseFetcher):
    """
    Fetch learning resources from YouTube.

    Priority:
      1. YouTube Data API v3 (if YOUTUBE_API_KEY is set) — most reliable
      2. httpx-based innertube scraping — no API key needed, best-effort

    Never returns mock/dummy data. Returns empty list on failure.
    """

    INNERTUBE_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
    INNERTUBE_URL = "https://www.youtube.com/youtubei/v1/search"
    YT_API_SEARCH = "https://www.googleapis.com/youtube/v3/search"
    YT_API_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(self, api_key: str = None):
        super().__init__("youtube")
        self.api_key = api_key
        self._httpx_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 5,
        duration: str = "any",
        order: str = "relevance",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not self.validate_search_query(query):
            return []

        limit = min(limit, 20)

        try:
            if self.api_key:
                results = await self._search_via_api(query, limit, duration, order)
            else:
                results = await self._search_via_scraping(query, limit)

            logger.info(f"YouTube: {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"YouTube search failed for '{query}': {e}")
            return []

    # ------------------------------------------------------------------
    # Strategy 1: Official YouTube Data API v3
    # ------------------------------------------------------------------

    async def _search_via_api(
        self,
        query: str,
        limit: int,
        duration: str,
        order: str,
    ) -> List[Dict[str, Any]]:
        try:
            import googleapiclient.discovery as discovery
        except ImportError:
            logger.warning("google-api-python-client not installed; falling back to scraping")
            return await self._search_via_scraping(query, limit)

        try:
            client = discovery.build("youtube", "v3", developerKey=self.api_key)

            duration_map = {
                "short": "short",
                "medium": "medium",
                "long": "long",
            }

            request = client.search().list(
                q=query,
                part="snippet",
                maxResults=limit,
                type="video",
                relevanceLanguage="en",
                order=order,
                videoDuration=duration_map.get(duration, "any"),
            )
            response = await asyncio.to_thread(request.execute)
            results = []

            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                details = await self._get_video_details_api(client, video_id)

                results.append({
                    "title": snippet["title"],
                    "description": snippet.get("description", "")[:500],
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "video_id": video_id,
                    "channel": snippet["channelTitle"],
                    "published_date": snippet.get("publishedAt", "")[:10],
                    "duration": details.get("duration", 0),
                    "view_count": details.get("view_count", 0),
                    "tags": snippet.get("tags", []),
                    "difficulty": "intermediate",
                })

            return results

        except Exception as e:
            logger.error(f"YouTube API error: {e}; falling back to scraping")
            return await self._search_via_scraping(query, limit)

    async def _get_video_details_api(self, client, video_id: str) -> Dict[str, Any]:
        try:
            req = client.videos().list(id=video_id, part="contentDetails,statistics")
            resp = await asyncio.to_thread(req.execute)
            if not resp.get("items"):
                return {}
            video = resp["items"][0]
            duration_str = video["contentDetails"]["duration"]
            duration_sec = self._parse_iso8601_duration(duration_str)
            view_count = int(video["statistics"].get("viewCount", 0))
            return {"duration": duration_sec, "view_count": view_count}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Strategy 2: httpx-based scraping of YouTube's internal API
    # ------------------------------------------------------------------

    async def _search_via_scraping(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Call YouTube's internal web API (no API key needed).
        Uses the same endpoint the browser uses for search suggestions.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; cannot scrape YouTube")
            return []

        try:
            # YouTube's internal search endpoint (innertube API)
            payload = {
                "context": {
                    "client": {
                        "clientName": "WEB",
                        "clientVersion": "2.20240101.00.00",
                        "hl": "en",
                        "gl": "US",
                    }
                },
                "query": query,
                "params": "EgIQAQ==",  # filter: videos only
            }
            headers = {
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "X-YouTube-Client-Name": "1",
                "X-YouTube-Client-Version": "2.20240101.00.00",
                "Origin": "https://www.youtube.com",
                "Referer": "https://www.youtube.com/",
            }

            url = f"{self.INNERTUBE_URL}?key={self.INNERTUBE_KEY}&prettyPrint=false"

            async with httpx.AsyncClient(verify=False, timeout=15) as client:
                resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code != 200:
                logger.warning(f"YouTube innertube returned {resp.status_code}")
                return []

            data = resp.json()
            return self._parse_innertube_response(data, limit)

        except Exception as e:
            logger.error(f"YouTube scraping failed: {e}")
            return []

    def _parse_innertube_response(
        self, data: Dict, limit: int
    ) -> List[Dict[str, Any]]:
        """Parse the deeply-nested innertube API response into flat video dicts."""
        results = []
        try:
            sections = (
                data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
            )
            for section in sections:
                items = (
                    section.get("itemSectionRenderer", {}).get("contents", [])
                )
                for item in items:
                    if len(results) >= limit:
                        break
                    video = item.get("videoRenderer")
                    if not video:
                        continue

                    video_id = video.get("videoId", "")
                    if not video_id:
                        continue

                    title = self._extract_text(video.get("title"))
                    description = self._extract_text(
                        video.get("descriptionSnippet")
                    )
                    channel = self._extract_text(
                        video.get("ownerText") or video.get("longBylineText")
                    )
                    duration_str = self._extract_text(
                        video.get("lengthText")
                    )
                    duration_sec = self._parse_duration_text(duration_str)
                    view_str = self._extract_text(video.get("viewCountText"))
                    view_count = self._parse_view_count(view_str)
                    published = self._extract_text(video.get("publishedTimeText"))

                    results.append({
                        "title": title,
                        "description": description[:500],
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "video_id": video_id,
                        "channel": channel,
                        "published_date": self._normalize_published_date(published),
                        "duration": duration_sec,
                        "view_count": view_count,
                        "tags": [],
                        "difficulty": "intermediate",
                    })

        except Exception as e:
            logger.warning(f"innertube parse error: {e}")

        return results

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(node) -> str:
        """Extract plain text from YouTube's runs/simpleText node."""
        if not node:
            return ""
        if isinstance(node, str):
            return node
        if "simpleText" in node:
            return node["simpleText"]
        if "runs" in node:
            return "".join(run.get("text", "") for run in node["runs"])
        return ""

    @staticmethod
    def _parse_duration_text(text: str) -> int:
        """Convert '14:32' or '1:04:22' to seconds."""
        if not text:
            return 0
        parts = text.strip().split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            pass
        return 0

    @staticmethod
    def _parse_view_count(text: str) -> int:
        """Parse '1.2M views', '45K views', '1,234 views' → int."""
        if not text:
            return 0
        text = text.lower().replace(",", "").replace(" views", "").replace(" view", "")
        try:
            if "m" in text:
                return int(float(text.replace("m", "")) * 1_000_000)
            elif "k" in text:
                return int(float(text.replace("k", "")) * 1_000)
            else:
                digits = re.sub(r"[^\d]", "", text)
                return int(digits) if digits else 0
        except ValueError:
            return 0

    @staticmethod
    def _normalize_published_date(text: str) -> Optional[str]:
        """
        Convert relative dates like '3 years ago' to approximate ISO date.
        Returns None if unparseable.
        """
        if not text:
            return None
        from datetime import datetime, timedelta
        text = text.lower().strip()
        now = datetime.utcnow()
        try:
            match = re.match(r"(\d+)\s+(\w+)\s+ago", text)
            if match:
                n, unit = int(match.group(1)), match.group(2).rstrip("s")
                delta = {
                    "second": timedelta(seconds=n),
                    "minute": timedelta(minutes=n),
                    "hour": timedelta(hours=n),
                    "day": timedelta(days=n),
                    "week": timedelta(weeks=n),
                    "month": timedelta(days=n * 30),
                    "year": timedelta(days=n * 365),
                }.get(unit, timedelta())
                return (now - delta).strftime("%Y-%m-%d")
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_iso8601_duration(duration_str: str) -> int:
        """Convert ISO 8601 duration string to seconds."""
        try:
            match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
            if match:
                h = int(match.group(1) or 0)
                m = int(match.group(2) or 0)
                s = int(match.group(3) or 0)
                return h * 3600 + m * 60 + s
        except Exception:
            pass
        return 0
