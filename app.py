"""
NATA AI Dataset Pipeline — Production Ready
==========================================

Oromo/Amharic news scraper, AI processor, and dataset builder.
Runs 24/7 with robust error handling, retries, and observability.

Usage:
    export GEMINI_API_KEY="..."
    export FIRESTORE_PROJECT_ID="..."
    export TELEGRAM_BOT_TOKEN="..."
    export TELEGRAM_CHAT_ID="..."
    export FIRESTORE_API_TOKEN="..."  # Optional but recommended
    python pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Set, TypeVar

import requests
import google.generativeai as genai

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nata_pipeline")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Immutable configuration loaded from environment variables."""
    
    channels: List[str] = field(default_factory=lambda: [
        "bbcafaanoromoo",
        "fana_afaan_oromoo", 
        "obn_afaan_oromoo",
        "voaafaanoromoo",
    ])
    batch_size: int = 20
    loop_interval_seconds: int = 3600
    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.2
    min_text_length: int = 70
    
    # API Keys
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    firestore_project_id: Optional[str] = field(default_factory=lambda: os.getenv("FIRESTORE_PROJECT_ID"))
    firestore_api_token: Optional[str] = field(default_factory=lambda: os.getenv("FIRESTORE_API_TOKEN"))
    telegram_bot_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    request_timeout: int = 15
    
    def validate(self) -> None:
        """Fail fast if required credentials are missing."""
        if not self.gemini_api_key:
            raise ValueError("Environment variable GEMINI_API_KEY is required.")
        if not self.firestore_project_id:
            raise ValueError("Environment variable FIRESTORE_PROJECT_ID is required.")
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials missing — reports will be disabled.")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RawPost:
    """Post scraped directly from Telegram web."""
    channel: str
    text: str
    hash: str
    
    @classmethod
    def from_text(cls, channel: str, text: str) -> "RawPost":
        """Create a RawPost with auto-generated hash."""
        clean = text.strip()
        post_hash = hashlib.md5(clean.encode("utf-8")).hexdigest()
        return cls(channel=channel, text=clean, hash=post_hash)


@dataclass
class ProcessedItem:
    """Cleaned and translated item ready for storage."""
    text_oromo: str
    text_amharic: str
    topic: str
    quality_score: float
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["ProcessedItem"]:
        """Safely create from Gemini JSON output. Returns None if invalid."""
        try:
            oromo = str(data.get("text_oromo", "")).strip()
            amharic = str(data.get("text_amharic", "")).strip()
            topic = str(data.get("topic", "news")).strip().lower()
            score = float(data.get("quality_score", 8.5))
            
            # Validation: both texts must exist
            if not oromo or not amharic:
                return None
                
            # Normalize topic
            valid_topics = {"news", "politics", "sports", "business"}
            if topic not in valid_topics:
                topic = "news"
                
            # Clamp score
            score = max(0.0, min(10.0, score))
            
            return cls(
                text_oromo=oromo,
                text_amharic=amharic,
                topic=topic,
                quality_score=score,
            )
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to parse processed item: {e}")
            return None


@dataclass
class RunStats:
    """Statistics for a single pipeline run."""
    scanned: int = 0
    unique: int = 0
    saved: int = 0
    duration_seconds: int = 0
    errors: int = 0


@dataclass
class PipelineStats:
    """Cumulative all-time statistics."""
    total_scanned: int = 0
    total_saved: int = 0
    total_runs: int = 0
    total_errors: int = 0


# =============================================================================
# RETRY DECORATOR
# =============================================================================

T = TypeVar("T")

def with_retry(
    max_retries: int = 3,
    delay: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """Decorator that retries a function with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = delay * (2 ** (attempt - 1))  # Exponential backoff
                        if on_retry:
                            on_retry(attempt, e)
                        else:
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt}/{max_retries}): {e}. "
                                f"Retrying in {wait:.1f}s..."
                            )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
            
            # If we get here, all retries failed
            raise last_exception or RuntimeError("Unexpected retry failure")
        return wrapper
    return decorator


# =============================================================================
# COMPONENTS
# =============================================================================

class TelegramScraper:
    """Handles scraping of Telegram web channels."""
    
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    TEXT_PATTERN = re.compile(
        r'<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>',
        re.DOTALL,
    )
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        })
    
    @with_retry(max_retries=2, delay=1.0, exceptions=(requests.RequestException,))
    def scrape_channel(self, channel: str) -> List[RawPost]:
        """Scrape a single Telegram channel."""
        url = f"https://t.me/s/{channel}"
        logger.info(f"Scraping channel: {channel}")
        
        response = self.session.get(url, timeout=self.config.request_timeout)
        response.raise_for_status()
        
        html = response.text
        matches = self.TEXT_PATTERN.findall(html)
        
        posts: List[RawPost] = []
        for match in matches:
            # Clean HTML tags
            clean = self.HTML_TAG_PATTERN.sub("", match).strip()
            clean = self.WHITESPACE_PATTERN.sub(" ", clean)
            
            # Filter: only keep substantial text
            if len(clean) > self.config.min_text_length:
                posts.append(RawPost.from_text(channel, clean))
        
        logger.info(f"Channel {channel}: extracted {len(posts)} posts")
        return posts
    
    def scrape_all(self) -> List[RawPost]:
        """Scrape all configured channels with polite delays."""
        all_posts: List[RawPost] = []
        
        for channel in self.config.channels:
            try:
                posts = self.scrape_channel(channel)
                all_posts.extend(posts)
                time.sleep(1)  # Polite delay between channels
            except Exception as e:
                logger.error(f"Failed to scrape {channel}: {e}")
        
        return all_posts
    
    @staticmethod
    def deduplicate(posts: List[RawPost]) -> List[str]:
        """Remove duplicate posts based on hash, return just the texts."""
        seen_hashes: Set[str] = set()
        unique_texts: List[str] = []
        
        for post in posts:
            if post.hash not in seen_hashes:
                seen_hashes.add(post.hash)
                unique_texts.append(post.text)
        
        logger.info(f"Deduplication: {len(posts)} → {len(unique_texts)} unique posts")
        return unique_texts


class GeminiProcessor:
    """Processes text batches using Google's Gemini AI."""
    
    SYSTEM_PROMPT = """You are an expert NLP Dataset Editor for Oromo and Amharic text.
Task:
1. Filter out spam, repetitive headers, and non-informative text.
2. Clean and humanize the Afaan Oromo text ("text_oromo").
3. Translate or generate accurate Amharic text ("text_amharic").
4. Classify topic ("news", "politics", "sports", "business").
5. Rate quality score (0.0 to 10.0).

STRICT INSTRUCTION:
Return ONLY a strictly formatted valid JSON Array of objects. No markdown, no prose.
Schema:
[
  {
    "text_oromo": "...",
    "text_amharic": "...",
    "topic": "news",
    "quality_score": 9.0
  }
]

Input batch:
"""
    
    def __init__(self, config: Config):
        self.config = config
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON array from text, handling various Gemini output formats."""
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
        text = text.strip()
        
        if not text:
            return None
        
        # If already starts with [, good
        if text.startswith("["):
            return text
        
        # Try to find JSON array embedded in text
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None
    
    @with_retry(
        max_retries=3,
        delay=2.0,
        exceptions=(Exception,),
        on_retry=lambda attempt, err: logger.warning(f"Gemini attempt {attempt} failed: {err}"),
    )
    def process_batch(self, texts: List[str]) -> List[ProcessedItem]:
        """Process a batch of texts through Gemini."""
        if not texts:
            return []
        
        prompt = self.SYSTEM_PROMPT + json.dumps(texts, ensure_ascii=False)
        
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.config.gemini_temperature},
        )
        
        raw_text = response.text
        json_str = self._extract_json(raw_text)
        
        if json_str is None:
            logger.warning("Gemini response did not contain valid JSON array.")
            logger.debug(f"Raw response: {raw_text[:500]}")
            return []
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed: {e}")
            return []
        
        if not isinstance(parsed, list):
            logger.warning(f"Expected list, got {type(parsed).__name__}")
            return []
        
        # Convert and validate each item
        items: List[ProcessedItem] = []
        for entry in parsed:
            if isinstance(entry, dict):
                item = ProcessedItem.from_dict(entry)
                if item:
                    items.append(item)
                else:
                    logger.debug(f"Skipping invalid Gemini output item: {entry}")
            else:
                logger.debug(f"Skipping non-dict item in Gemini output")
        
        logger.info(f"Gemini batch: {len(texts)} in → {len(items)} valid out")
        return items
    
    def process_all(self, texts: List[str]) -> List[ProcessedItem]:
        """Process all texts in configurable batch sizes."""
        if not texts:
            return []
        
        all_items: List[ProcessedItem] = []
        chunks = [
            texts[i : i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]
        
        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"Processing batch {idx}/{len(chunks)} (size: {len(chunk)})")
            try:
                items = self.process_batch(chunk)
                all_items.extend(items)
                time.sleep(2)  # Rate limit protection
            except Exception as e:
                logger.error(f"Batch {idx} failed permanently: {e}")
        
        return all_items


class FirestoreStorage:
    """Persists processed items to Firestore via REST API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = (
            f"https://firestore.googleapis.com/v1/projects/{config.firestore_project_id}"
            f"/databases/(default)/documents/cleaned_dataset"
        )
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        if config.firestore_api_token:
            self.session.headers["Authorization"] = f"Bearer {config.firestore_api_token}"
            logger.info("Firestore: using Bearer token authentication")
        else:
            logger.warning("Firestore: no API token — requests may fail with 403")
    
    def _build_document(self, item: ProcessedItem) -> dict[str, Any]:
        """Build Firestore REST API document body."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        return {
            "fields": {
                "text_oromo": {"stringValue": item.text_oromo},
                "text_amharic": {"stringValue": item.text_amharic},
                "topic": {"stringValue": item.topic},
                "quality_score": {"doubleValue": item.quality_score},
                "created_at": {"timestampValue": timestamp},
            }
        }
    
    @with_retry(max_retries=2, delay=1.5, exceptions=(requests.RequestException,))
    def save(self, item: ProcessedItem) -> bool:
        """Save a single item to Firestore."""
        body = self._build_document(item)
        
        response = self.session.post(self.base_url, json=body, timeout=10)
        
        if response.status_code == 200:
            return True
        
        logger.error(
            f"Firestore save failed: HTTP {response.status_code} — {response.text[:200]}"
        )
        return False
    
    def save_all(self, items: List[ProcessedItem]) -> int:
        """Save multiple items, counting successes. One failure doesn't stop others."""
        saved_count = 0
        
        for item in items:
            try:
                if self.save(item):
                    saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save item after retries: {e}")
        
        logger.info(f"Firestore: saved {saved_count}/{len(items)} items")
        return saved_count


class TelegramReporter:
    """Sends run summary reports to Telegram."""
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config.telegram_bot_token and config.telegram_chat_id)
        
        if self.enabled:
            self.base_url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
        else:
            logger.info("Telegram reporter disabled (missing credentials)")
    
    def send_report(self, run_stats: RunStats, cumulative: PipelineStats) -> None:
        """Send formatted report to Telegram."""
        if not self.enabled:
            return
        
        message = (
            f"📊 *NATA AI Real Dataset Pipeline Report*\n\n"
            f"⏱ *Current Run Summary ({run_stats.duration_seconds}s):*\n"
            f"• Total Web Scanned: *{run_stats.scanned}*\n"
            f"• Unique Filtered: *{run_stats.unique}*\n"
            f"• Saved to Firestore: *{run_stats.saved}*\n"
            f"• Errors: *{run_stats.errors}*\n\n"
            f"📈 *Cumulative Totals (All-Time):*\n"
            f"• Total Scanned: *{cumulative.total_scanned}*\n"
            f"• Total Saved JSON: *{cumulative.total_saved}*\n"
            f"• Total Runs Completed: *{cumulative.total_runs}*\n"
            f"• Total Errors: *{cumulative.total_errors}*\n\n"
            f"🚀 *Status: 24/7 Active & Running on HuggingFace*"
        )
        
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram report sent successfully")
        except Exception as e:
            logger.error(f"Telegram report failed: {e}")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class Pipeline:
    """
    Main orchestrator that wires all components together.
    Handles a single run and updates cumulative statistics.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.scraper = TelegramScraper(config)
        self.processor = GeminiProcessor(config)
        self.storage = FirestoreStorage(config)
        self.reporter = TelegramReporter(config)
        self.cumulative = PipelineStats()
        self._shutdown_requested = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self._shutdown_requested = True
    
    def run_once(self) -> RunStats:
        """Execute one full pipeline cycle."""
        start_time = time.time()
        logger.info("=" * 50)
        logger.info("Starting Pipeline Run")
        logger.info("=" * 50)
        
        stats = RunStats()
        
        # ------------------------------------
                # -----------------------------------------------------------------
        # PIPELINE EXECUTION STEPS
        # -----------------------------------------------------------------
        try:
            # 1. Scrape raw posts from Telegram web
            raw_posts = self.scraper.scrape_all()
            stats.scanned = len(raw_posts)
            
            # 2. Deduplicate texts
            unique_texts = self.scraper.deduplicate(raw_posts)
            stats.unique = len(unique_texts)
            
            # 3. Process batches through Gemini AI
            processed_items = self.processor.process_all(unique_texts)
            
            # 4. Save cleaned JSON items to Firestore
            stats.saved = self.storage.save_all(processed_items)
            
        except Exception as e:
            logger.critical(f"Unhandled pipeline failure during execution: {e}")
            stats.errors += 1
        
        # Calculate duration and update cumulative statistics
        stats.duration_seconds = int(time.time() - start_time)
        
        self.cumulative.total_scanned += stats.scanned
        self.cumulative.total_saved += stats.saved
        self.cumulative.total_runs += 1
        self.cumulative.total_errors += stats.errors
        
        # 5. Send report to Telegram
        self.reporter.send_report(stats, self.cumulative)
        
        logger.info(
            f"Run completed in {stats.duration_seconds}s | "
            f"Scanned: {stats.scanned} | Unique: {stats.unique} | Saved: {stats.saved}"
        )
        return stats

    def run_forever(self) -> None:
        """Run the pipeline continuously in an infinite loop with interval delays."""
        logger.info("Starting continuous 24/7 NATA AI Pipeline execution loop...")
        
        while not self._shutdown_requested:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Unexpected error in pipeline run loop: {e}")
            
            if self._shutdown_requested:
                break
                
            logger.info(f"Sleeping for {self.config.loop_interval_seconds} seconds until next run...")
            
            # Sleep in small increments to respond quickly to shutdown signals
            sleep_time = self.config.loop_interval_seconds
            while sleep_time > 0 and not self._shutdown_requested:
                time.sleep(min(5, sleep_time))
                sleep_time -= 5
                
        logger.info("Pipeline gracefully stopped.")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    try:
        config = Config()
        config.validate()
        
        pipeline = Pipeline(config)
        
        # Single-run mode for GitHub Actions / Cron, or Continuous mode for VPS / HuggingFace
        if os.getenv("SINGLE_RUN", "false").lower() in ("true", "1", "yes"):
            pipeline.run_once()
        else:
            pipeline.run_forever()
            
    except Exception as e:
        logger.critical(f"Pipeline initialization failed: {e}")
        sys.exit(1)
