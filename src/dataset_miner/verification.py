import logging
from typing import List, TypedDict
from langchain.schema import StrOutputParser
from dataset_miner.cost_analyzer import CostAnalyzer
from dataset_miner.project_types import ChatModel
from dataset_miner.prompt_templates import VERIFICATION_TEMPLATE

logger = logging.getLogger(__name__)


class VerificationResult(TypedDict):
    status: str
    explanation: str


class QAPair(TypedDict):
    instruction: str
    input: str
    output: str


class VerifiedQAPair(QAPair):
    verification: VerificationResult


def verify_qa_pair(
    context: str,
    qa_pair: QAPair,
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter=None,
) -> VerificationResult:
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
        return VerificationResult(status="ERROR", explanation=str(e))

    # Output Token Count
    output_tokens = cost_analyzer.count_tokens(response)
    # Token Cost Analysis
    verification_cost = cost_analyzer.add_verification_usage(
        input_tokens, output_tokens
    )
    logger.info(f"💰 Verification cost: ${verification_cost:.6f}")

    if response.strip().upper().startswith("CORRECT"):
        return VerificationResult(status="CORRECT", explanation=response[7:].strip())
    elif response.strip().upper().startswith("INCORRECT"):
        return VerificationResult(status="INCORRECT", explanation=response[9:].strip())
    else:
        return VerificationResult(
            status="UNKNOWN", explanation="Unexpected verification response"
        )


def verify_dataset(
    context: str,
    qa_pairs: List[QAPair],
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter=None,
) -> List[VerifiedQAPair]:
    verified_dataset: List[VerifiedQAPair] = []
    for qa_pair in qa_pairs:
        verification_result: VerificationResult = verify_qa_pair(
            context, qa_pair, llm, cost_analyzer, rate_limiter
        )
        verified_pair: VerifiedQAPair = {
            **qa_pair,
            "verification": verification_result,
        }
        verified_dataset.append(verified_pair)
    return verified_dataset
