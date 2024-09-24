from langchain.prompts import PromptTemplate

QA_GENERATION_TEMPLATE = PromptTemplate(
    input_variables=["text"],
    template="""Generate a set of meaningful and contextually relevant questions and answers based on the following text. The number of questions should be appropriate to the content provided. Ensure that the questions focus on general concepts, processes, or important insights present in the text. Avoid asking questions based on specific values, dates, locations, or other particular details that are merely examples.

Respond in the form of a JSON array where each object contains:
- 'instruction': The generated question.
- 'input': Any relevant contextual input; leave as an empty string if not needed.
- 'output': The corresponding answer.

Your generated questions should be informative, insightful, and highlight key ideas or concepts from the text, without focusing on trivial or overly specific details.

Text: {text}
""",
)
