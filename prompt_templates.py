from langchain.prompts import PromptTemplate

QA_GENERATION_TEMPLATE = PromptTemplate(
    input_variables=["text"],
    template="""Generate any number of questions and their answers based on the following text. The number of questions generated can be based on the context itself. Respond with a JSON array containing objects with 'instruction', 'input', and 'output' keys. The 'instruction' should be the question, 'input' can be any sample inputs, and if not input it should be empty string, and 'output' should be the answer. Focus on general questions relevant to the given text. There's no need to generate question related to location, time, or other specific details of documents. Text: {text}""",
)

# Add more prompt templates here as needed
