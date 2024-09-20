import json
import logging
import re
from typing import List
from langchain_community.llms import Ollama
from langchain.schema import StrOutputParser
from cost_analyzer import CostAnalyzer
from prompt_templates import QA_GENERATION_TEMPLATE

logger = logging.getLogger(__name__)


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
    text_chunk: str, llm: Ollama, cost_analyzer: CostAnalyzer
) -> List[dict]:
    chain = QA_GENERATION_TEMPLATE | llm | StrOutputParser()
    prompt_text = QA_GENERATION_TEMPLATE.format(text=text_chunk)
    input_tokens = cost_analyzer.count_tokens(prompt_text)
    logger.debug(f"Sending chunk of {len(text_chunk)} characters to Ollama")

    try:
        response = chain.invoke({"text": text_chunk})
        logger.debug(f"Received response of {len(response)} characters from Ollama")
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
    text: str, llm: Ollama, cost_analyzer: CostAnalyzer, chunk_size: int = 2000
) -> List[dict]:
    logger.debug(f"Processing text of length {len(text)}")
    responses = generate_questions_answers(text, llm, cost_analyzer)
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
