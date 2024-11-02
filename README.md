# Dataset Miner

Dataset Miner is a powerful Python tool designed to generate high-quality question-answer (Q&A) pairs from various document formats using AI models. It processes documents such as PDFs, text files, Word documents, programming code files, JSON, CSV, and Excel files to create datasets suitable for fine-tuning language models or other NLP tasks.

## 🌟 Features

- **Multiple Format Support**:
  - Documents: PDF, TXT, DOCX
  - Data Files: JSON, CSV, XLSX, XLS
  - Code Files: Python, Java, JavaScript, TypeScript, C++, and many more
- **Advanced Processing**:
  - Smart text chunking with support for code-aware splitting
  - Intelligent header detection for markdown and HTML files
  - Robust encoding handling for various file formats
  - Support for table extraction from Word documents
- **AI Integration**:
  - Compatible with both Ollama and Groq AI models
  - Customizable prompt templates for Q&A generation
  - Optional verification of generated Q&A pairs
- **Performance & Control**:
  - Rate limiting for API calls
  - Token usage tracking and cost analysis
  - Progress tracking with detailed logging
  - Support for processing large documents efficiently

## 📋 Requirements

- Python 3.8 or higher
- Dependencies (automatically installed):
  - langchain & langchain-community
  - langchain-groq (for Groq integration)
  - langchain-ollama (for Ollama integration)
  - PyPDF2 (PDF processing)
  - python-docx (Word document processing)
  - pandas & openpyxl (Excel/CSV processing)
  - tiktoken (token counting)
  - Additional utilities: tqdm, colorama, python-dotenv

## 💻 Installation

1. Install from PyPI:

   ```bash
   pip install dataset-miner
   ```

2. Or install from source:
   ```bash
   git clone https://github.com/chrissmartin/dataset-miner.git
   cd dataset-miner
   pip install -e .
   ```

## 🚀 Usage

### Command Line Interface

```bash
dataset-miner -source <input_directory> -model <ai_model_name> [options]
```

Required arguments:

- `-source`: Directory containing files to process
- `-model`: AI model identifier (e.g., "gpt-4o-mini" for Ollama)

Optional arguments:

- `--output`: Output JSON file path (default: mined_dataset.json)
- `--use-groq`: Use Groq instead of Ollama
- `--verify`: Enable verification of generated Q&A pairs
- `--debug`: Enable debug logging
- `--remove-empty-columns`: Remove empty columns from CSV/Excel files

### Python API

```python
from dataset_miner import mine_documents

# Basic usage
mined_data, output_path = mine_documents(
    source_dir="./documents",
    model="gpt-4o-mini",
    output_file="dataset.json"
)

# Advanced usage with all options
mined_data, output_path = mine_documents(
    source_dir="./documents",
    model="gpt-4o-mini",
    output_file="dataset.json",
    use_groq=True,
    remove_empty_columns=True,
    verify=True,
    debug=True
)
```

## 📤 Output Format

The tool generates a JSON file containing Q&A pairs in the Alpaca dataset format:

```json
[
  {
    "instruction": "Question text here",
    "input": "Additional context (if any)",
    "output": "Answer text here"
  }
]
```

When verification is enabled, each entry includes additional verification metadata:

```json
[
  {
    "instruction": "Question text here",
    "input": "Additional context (if any)",
    "output": "Answer text here",
    "verification": {
      "status": "CORRECT",
      "explanation": "Verification details"
    }
  }
]
```

## 📊 Cost Analysis

The tool provides detailed cost analysis and usage statistics:

- Token usage tracking (input/output)
- Cost breakdown by operation type
- Verification costs (if enabled)
- Average cost per Q&A pair
- Total cost summary

## 🔍 Logging

The tool provides comprehensive logging with different verbosity levels:

- Basic progress updates
- Token usage and cost tracking
- Error reporting and debugging information
- Color-coded console output for better visibility

Enable debug logging with the `--debug` flag for more detailed information.

## 🔐 Environment Variables

- `GROQ_API_KEY`: Required when using Groq integration (set in .env file)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature enhancements
- Documentation improvements
- Test coverage expansion

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

While Dataset Miner strives for accuracy in Q&A pair generation, the output should be reviewed for quality and appropriateness before use in production systems or datasets. The generated content depends on the AI model used and may require manual verification for critical applications.
