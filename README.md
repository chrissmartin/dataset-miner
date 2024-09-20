# Dataset Miner

Dataset Miner is a powerful tool designed to generate question-answer (Q&A) pairs from various file types using AI models. It processes documents such as PDFs, text files, Word documents, JSON, CSV, and Excel files to create a dataset suitable for fine-tuning language models or other NLP tasks.

## Features

- Supports multiple file formats: PDF, TXT, DOCX, JSON, CSV, XLSX, XLS
- Uses Ollama AI models for Q&A pair generation
- Processes files in chunks to handle large documents efficiently
- Provides cost analysis for token usage
- Outputs data in Alpaca dataset format
- Includes logging for easy debugging and progress tracking

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/your-username/dataset-miner.git
   cd dataset-miner
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with the following command:

```
python main.py -source <input_directory> -model <ollama_model_id> [--output <output_file>] [--debug]
```

Arguments:

- `-source`: Directory containing files to mine (required)
- `-model`: Ollama AI model ID/slug to use for mining (required)
- `--output`: Output JSON file for the mined dataset (default: mined_dataset.json)
- `--debug`: Enable debug logging

Example:

```
python main.py -source ./documents -model gpt-4o-mini --output mined_data.json
```

## Project Structure

- `main.py`: Entry point of the application
- `data_extractor.py`: Contains functions to extract text from various file formats
- `llm_utils.py`: Utility functions for interacting with the AI model and processing text
- `cost_analyzer.py`: Tracks token usage and estimates costs
- `prompt_templates.py`: Defines prompt templates for the AI model
- `requirements.txt`: Lists all required Python packages

## Output

The tool generates a JSON file containing Q&A pairs in the Alpaca dataset format:

```json
[
  {
    "instruction": "Question goes here",
    "input": "Any relevant input (if applicable)",
    "output": "Answer goes here"
  },
  ...
]
```

## Cost Analysis

The tool provides a summary of token usage and estimated costs based on the specified model's pricing. This information is logged at the end of the mining process.

## Logging

Detailed logs are written to the console, including progress updates, token usage, and any errors encountered during the mining process. Use the `--debug` flag for more verbose logging.

## Contributing

Contributions to the Dataset Miner project are welcome! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

[Specify your chosen license here]

## Disclaimer

This tool uses AI models to generate content. While it strives for accuracy, the generated Q&A pairs should be reviewed for quality and appropriateness before use in production systems or datasets.
