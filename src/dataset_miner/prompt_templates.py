from langchain.prompts import PromptTemplate

QA_GENERATION_TEMPLATE_TEXT = """
Generate any number of questions and their answers based on the following text.
The number of questions generated can be based on the context itself.
Respond with a JSON array containing objects with 'instruction', 'input', and 'output' keys.
The 'instruction' should be the question, 'input' can be any sample inputs, and if there is no input,
it should be an empty string, and 'output' should be the answer.
Focus on general questions relevant to the given text.
There's no need to generate questions related to location, time, or other specific details of the documents.
Text: {text}
"""

QA_GENERATION_TEMPLATE = PromptTemplate(
    input_variables=["text"],
    template=QA_GENERATION_TEMPLATE_TEXT.strip(),
)

VERIFICATION_TEMPLATE_TEXT = """
You are a helpful assistant that verifies the relevance of a question and the correctness of an answer based on the provided context.

Context: {context}
Question: {question}
Answer: {answer}

Determine if the question is relevant to the context and if the answer is correct and relevant to the context.

Respond with either 'CORRECT' or 'INCORRECT', followed by a brief explanation.

Provide your response **exactly** in the following JSON format:

{{
    "verdict": "CORRECT" or "INCORRECT",
    "explanation": "Your brief explanation here."
}}

**Example response:**

{{
    "verdict": "CORRECT",
    "explanation": "The answer accurately reflects the information in the context."
}}
"""


VERIFICATION_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template=VERIFICATION_TEMPLATE_TEXT.strip(),
)
