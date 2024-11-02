import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

# Rate limiting configuration for Groq
GROQ_REQUESTS_PER_MINUTE = 29
GROQ_TOKENS_PER_MINUTE = 14000


class RateLimit:
    def __init__(self, limit, window):
        self.limit = limit  # Maximum allowed in the window
        self.window = window  # Time window in seconds
        self.entries = deque()  # Stores (timestamp, count)

    def wait(self, count=1):
        current_time = time.time()
        # Remove entries outside the window
        while self.entries and current_time - self.entries[0][0] >= self.window:
            self.entries.popleft()

        total_count = sum(c for _, c in self.entries)
        if total_count + count > self.limit:
            oldest_time = self.entries[0][0]
            sleep_time = self.window - (current_time - oldest_time)
            if sleep_time > 0:
                logger.info(
                    f"Rate limit of {self.limit} per {self.window} seconds reached. "
                    f"Sleeping for {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)
                current_time = time.time()
                # Clean up again after sleeping
                while self.entries and current_time - self.entries[0][0] >= self.window:
                    self.entries.popleft()

        self.entries.append((current_time, count))


class RateLimiter:
    def __init__(self, requests_per_minute, tokens_per_minute):
        window = 60  # 1 minute window in seconds
        self.request_limit = RateLimit(requests_per_minute, window)
        self.token_limit = RateLimit(tokens_per_minute, window)

    def wait(self, tokens):
        self.request_limit.wait()  # Each request counts as 1
        self.token_limit.wait(tokens)  # Tokens consumed in the request
