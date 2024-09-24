import logging
import time

logger = logging.getLogger(__name__)

# Rate limiting configuration for Groq
GROQ_REQUESTS_PER_MINUTE = 29
GROQ_TOKENS_PER_MINUTE = 14000


class RateLimiter:
    def __init__(self, requests_per_minute, tokens_per_minute):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps = []
        self.token_usage = []

    def wait(self, tokens):
        current_time = time.time()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            t for t in self.request_timestamps if current_time - t < 60
        ]
        self.token_usage = [t for t in self.token_usage if current_time - t[0] < 60]

        # Check if we've exceeded the request limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)

        # Check if we've exceeded the token limit
        total_tokens = sum(t[1] for t in self.token_usage)
        if total_tokens + tokens > self.tokens_per_minute:
            sleep_time = 60 - (current_time - self.token_usage[0][0])
            if sleep_time > 0:
                logger.info(
                    f"Token limit reached. Sleeping for {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)

        # Update timestamps and token usage
        self.request_timestamps.append(time.time())
        self.token_usage.append((time.time(), tokens))
