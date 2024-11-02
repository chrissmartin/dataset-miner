"""Module containing file type definitions and mappings."""

import os
import logging
from typing import Dict, List, Set
from langchain_text_splitters import Language

logger = logging.getLogger(__name__)

# Document types supported by the data extractor
DOCUMENT_EXTENSIONS: Set[str] = {
    ".pdf",  # PDF documents
    ".txt",  # Plain text files
    ".docx",  # Word documents
    ".json",  # JSON files
    ".csv",  # CSV files
    ".xlsx",  # Excel files
    ".xls",  # Legacy Excel files
}

# Mapping of file extensions to supported language types
EXTENSION_TO_LANGUAGE: Dict[str, Language] = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".php": Language.PHP,
    ".proto": Language.PROTO,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".tex": Language.LATEX,
    ".html": Language.HTML,
    ".sol": Language.SOL,
    ".cs": Language.CSHARP,
    ".cbl": Language.COBOL,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,
    ".ex": Language.ELIXIR,
    ".ps1": Language.POWERSHELL,
    ".rst": Language.RST,
    ".go": Language.GO,
}


def get_file_extension(filename: str) -> str:
    # Check if the input is just an extension (e.g., ".py")
    if filename.startswith(".") and filename.count(".") == 1:
        logger.info(f"Direct extension provided: {filename}")
        return filename

    extension = os.path.splitext(filename.lower())[1]
    logger.info(f"Extension Extracted: {extension} from filename: {filename}")
    return extension


def get_all_supported_extensions() -> Set[str]:
    """
    Get all supported file extensions, combining both document types
    and programming language file types.

    Returns:
        Set of supported file extensions including both document and code files
    """
    return DOCUMENT_EXTENSIONS.union(set(EXTENSION_TO_LANGUAGE.keys()))


def is_supported_file(filename: str) -> bool:
    return is_supported_extension(get_file_extension(filename))


def is_supported_code_file(filename: str) -> bool:
    extension = get_file_extension(filename)
    return extension in EXTENSION_TO_LANGUAGE


def is_supported_extension(extension: str) -> bool:
    """
    Check if a given file extension is supported.

    Args:
        extension: File extension to check (with or without leading dot)

    Returns:
        True if the extension is supported, False otherwise
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension.lower() in get_all_supported_extensions()


def get_file_type(filename: str) -> str:
    """
    Get the type category of a file based on its extension.

    Args:
        extension: File extension to categorize (with or without leading dot)

    Returns:
        String indicating the file type category: 'document', 'code', or 'unknown'
    """

    extension: str = get_file_extension(filename)

    if not extension.startswith("."):
        extension = f".{extension}"
    extension = extension.lower()

    if extension in DOCUMENT_EXTENSIONS:
        return "document"
    elif extension in EXTENSION_TO_LANGUAGE:
        return "code"
    else:
        return "unknown"


def get_language(filename: str) -> Language:
    """
    Get the Language enum value for a given file extension.

    Args:
        filename: File Name

    Returns:
        Language enum value

    Raises:
        ValueError: If the extension is not a supported programming language
    """

    extension: str = get_file_extension(filename)

    if not extension.startswith("."):
        extension = f".{extension}"
    extension = extension.lower()

    if extension not in EXTENSION_TO_LANGUAGE:
        raise ValueError(f"Unsupported programming language extension: {extension}")

    return EXTENSION_TO_LANGUAGE[extension]


def get_file_list(source_dir: str) -> List[str]:
    supported_extensions = get_all_supported_extensions()

    # List all files and filter by supported extensions
    return [
        f
        for f in os.listdir(source_dir)
        if os.path.splitext(f.lower())[1] in supported_extensions
    ]
