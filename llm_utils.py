import json
import logging
import re
import time
from typing import List
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser
from cost_analyzer import CostAnalyzer
from prompt_templates import QA_GENERATION_TEMPLATE

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


def extract_json_from_response(response: str) -> List[dict]:
    json_matches = re.findall(r"\[[\s\S]*\]", response, re.DOTALL)
    results = []
    for json_str in json_matches:
        try:
            json_obj = json.loads(json_str)
            if isinstance(json_obj, list):
                results.extend(json_obj)
            elif isinstance(json_obj, dict):
                results.append(json_obj)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {json_str}")
    return results


def generate_questions_answers(
    text_chunk: str, llm, cost_analyzer: CostAnalyzer, rate_limiter: RateLimiter = None
) -> List[dict]:
    chain = QA_GENERATION_TEMPLATE | llm | StrOutputParser()
    prompt_text = QA_GENERATION_TEMPLATE.format(text=text_chunk)
    input_tokens = cost_analyzer.count_tokens(prompt_text)
    logger.debug(f"Sending chunk of {len(text_chunk)} characters to LLM")

    if rate_limiter:
        rate_limiter.wait(input_tokens)

    try:
        response = chain.invoke({"text": text_chunk})
        logger.debug(f"Received response of {len(response)} characters from LLM")
        logger.debug(f"Raw LLM response: {response}")
    except Exception as e:
        logger.error(f"Error generating Q&A pairs: {str(e)}")
        return []

    output_tokens = cost_analyzer.count_tokens(response)
    input_cost, output_cost = cost_analyzer.add_usage(input_tokens, output_tokens)
    total_cost = input_cost + output_cost
    logger.info(
        f"Chunk processing cost: ${total_cost:.6f} (Input: ${input_cost:.6f}, Output: ${output_cost:.6f})"
    )

    result = extract_json_from_response(response)
    if result:
        logger.info(f"Generated {len(result)} Q&A pairs from this chunk")
        return result
    else:
        logger.error("No valid JSON objects found in the response")
        return []


def process_text(
    text: str,
    llm,
    cost_analyzer: CostAnalyzer,
    rate_limiter: RateLimiter = None,
) -> List[dict]:
    logger.debug(f"Processing text of length {len(text)}")
    responses = generate_questions_answers(text, llm, cost_analyzer, rate_limiter)
    logger.info(f"Generated {len(responses)} Q&A pairs")
    return responses


def format_alpaca_dataset(qa_pairs: List[dict]) -> List[dict]:
    formatted_data = []
    for i, qa in enumerate(qa_pairs):
        try:
            formatted_data.append(
                {
                    "instruction": str(qa.get("instruction", "")),
                    "input": str(qa.get("input", "")),
                    "output": str(qa.get("output", "")),
                }
            )
        except Exception as e:
            logger.error(f"Error formatting Q&A pair {i}: {str(e)}")
            logger.debug(f"Problematic Q&A pair: {qa}")
    return formatted_data
