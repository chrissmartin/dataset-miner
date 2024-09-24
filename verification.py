import logging
from typing import List, Dict
from langchain.schema import StrOutputParser
from cost_analyzer import CostAnalyzer
from prompt_templates import VERIFICATION_TEMPLATE

logger = logging.getLogger(__name__)


def verify_qa_pair(
    context: str,
    qa_pair: Dict[str, str],
    llm,
    cost_analyzer: CostAnalyzer,
    rate_limiter=None,
) -> Dict[str, str]:
    chain = VERIFICATION_TEMPLATE | llm | StrOutputParser()

    prompt_text = VERIFICATION_TEMPLATE.format(
        context=context, question=qa_pair["instruction"], answer=qa_pair["output"]
    )

    input_tokens = cost_analyzer.count_tokens(prompt_text)

    if rate_limiter:
        rate_limiter.wait(input_tokens)

    try:
        response = chain.invoke(
            {
                "context": context,
                "question": qa_pair["instruction"],
                "answer": qa_pair["output"],
            }
        )
        logger.debug(f"Verification response: {response}")
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return {"status": "ERROR", "explanation": str(e)}

    output_tokens = cost_analyzer.count_tokens(response)
    input_cost, output_cost = cost_analyzer.add_usage(input_tokens, output_tokens)
    total_cost = input_cost + output_cost
    logger.info(
        f"💰 Verification cost: ${total_cost:.6f} (Input: ${input_cost:.6f}, Output: ${output_cost:.6f})"
    )

    if response.startswith("CORRECT"):
        return {"status": "CORRECT", "explanation": response[7:].strip()}
    elif response.startswith("INCORRECT"):
        return {"status": "INCORRECT", "explanation": response[9:].strip()}
    else:
        return {"status": "UNKNOWN", "explanation": "Unexpected verification response"}


def verify_dataset(
    context: str,
    qa_pairs: List[Dict[str, str]],
    llm,
    cost_analyzer: CostAnalyzer,
    rate_limiter=None,
) -> List[Dict[str, str]]:
    verified_dataset = []
    for qa_pair in qa_pairs:
        verification_result = verify_qa_pair(
            context, qa_pair, llm, cost_analyzer, rate_limiter
        )
        verified_pair = {**qa_pair, "verification": verification_result}
        verified_dataset.append(verified_pair)
    return verified_dataset
