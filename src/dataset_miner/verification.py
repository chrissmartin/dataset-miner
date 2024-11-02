import logging
from typing import List, Dict
from langchain.schema import StrOutputParser
from dataset_miner.cost_analyzer import CostAnalyzer
from dataset_miner.project_types import ChatModel
from dataset_miner.prompt_templates import VERIFICATION_TEMPLATE

logger = logging.getLogger(__name__)


def verify_qa_pair(
    context: str,
    qa_pair: Dict[str, str],
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter=None,
) -> Dict[str, str]:
    question = qa_pair["instruction"]
    answer = qa_pair["output"]

    # Input Token Count
    prompt_text_for_count = VERIFICATION_TEMPLATE.format(
        context=context, question=question, answer=answer
    )
    input_tokens = cost_analyzer.count_tokens(prompt_text_for_count)

    if rate_limiter:
        rate_limiter.wait(input_tokens)

    try:
        logger.debug(f"Verifying Q&A pair: {qa_pair}")
        chain = VERIFICATION_TEMPLATE | llm | StrOutputParser()
        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "answer": answer,
            }
        )
        logger.debug(f"Verification response: {response}")
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return {"status": "ERROR", "explanation": str(e)}

    # Output Token Count
    output_tokens = cost_analyzer.count_tokens(response)
    # Token Cost Analysis
    verification_cost = cost_analyzer.add_verification_usage(
        input_tokens, output_tokens
    )
    logger.info(f"💰 Verification cost: ${verification_cost:.6f}")

    if response.strip().upper().startswith("CORRECT"):
        return {"status": "CORRECT", "explanation": response[7:].strip()}
    elif response.strip().upper().startswith("INCORRECT"):
        return {"status": "INCORRECT", "explanation": response[9:].strip()}
    else:
        return {"status": "UNKNOWN", "explanation": "Unexpected verification response"}


def verify_dataset(
    context: str,
    qa_pairs: List[Dict[str, str]],
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter=None,
) -> List[Dict[str, str]]:
    verified_dataset: List[Dict[str, str]] = []
    for qa_pair in qa_pairs:
        verification_result: Dict[str, str] = verify_qa_pair(
            context, qa_pair, llm, cost_analyzer, rate_limiter
        )
        verified_pair: Dict[str, str] = {
            **qa_pair,
            "verification": str(verification_result),
        }
        verified_dataset.append(verified_pair)
    return verified_dataset
