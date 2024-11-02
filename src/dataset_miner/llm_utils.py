import json
import logging
import os
import re
from typing import List, Optional
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from pydantic import SecretStr

from dataset_miner.cost_analyzer import CostAnalyzer
from dataset_miner.project_types import ChatModel, CliArgs
from dataset_miner.prompt_templates import QA_GENERATION_TEMPLATE
from dataset_miner.rate_limiter import (
    GROQ_REQUESTS_PER_MINUTE,
    GROQ_TOKENS_PER_MINUTE,
    RateLimiter,
)

logger = logging.getLogger(__name__)


def initialize_llm(args: CliArgs):
    llm: Optional[ChatModel] = None
    if args.use_groq:
        groq_api_key_value = os.getenv("GROQ_API_KEY")
        if groq_api_key_value is not None:
            groq_api_key = SecretStr(groq_api_key_value)
        else:
            logger.error(
                "❌ GROQ_API_KEY not found in environment variables. Please set it in your .env file."
            )
            return None, None
        llm = ChatGroq(model=args.model, api_key=groq_api_key, stop_sequences=None)
        rate_limiter = RateLimiter(GROQ_REQUESTS_PER_MINUTE, GROQ_TOKENS_PER_MINUTE)
        logger.info("🚀 Using Groq with rate limiting")
    else:
        llm = OllamaLLM(model=args.model)
        rate_limiter = None
        logger.info("🚀 Using Ollama")
    return llm, rate_limiter


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
    text_chunk: str,
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter: Optional[RateLimiter] = None,
) -> List[dict]:
    # Input Token Count
    prompt_text_for_count = QA_GENERATION_TEMPLATE.format(text=text_chunk)
    input_tokens = cost_analyzer.count_tokens(prompt_text_for_count)

    if rate_limiter:
        rate_limiter.wait(input_tokens)

    try:
        logger.debug(f"Sending chunk of {len(text_chunk)} characters to Ollama")
        chain = QA_GENERATION_TEMPLATE | llm | StrOutputParser()
        response = chain.invoke({"text": text_chunk})
        logger.debug(f"Received response of {len(response)} characters from Ollama")
        logger.debug(f"Raw LLM response: {response}")
    except Exception as e:
        logger.error(f"Error generating Q&A pairs: {str(e)}")
        return []

    # Output Token Count
    output_tokens = cost_analyzer.count_tokens(response)
    # Token Cost Analysis
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
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter: Optional[RateLimiter] = None,
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
                    "instruction": qa.get("instruction", ""),
                    "input": qa.get("input", ""),
                    "output": qa.get("output", ""),
                }
            )
        except Exception as e:
            logger.error(f"Error formatting Q&A pair {i}: {str(e)}")
            logger.debug(f"Problematic Q&A pair: {qa}")
    return formatted_data
