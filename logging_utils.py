import logging
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)


class ColorfulFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.CYAN
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Style.RESET_ALL,
        logging.INFO: Fore.GREEN
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Style.RESET_ALL,
        logging.ERROR: Fore.RED
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Style.RESET_ALL,
        logging.CRITICAL: Back.RED
        + Fore.WHITE
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Style.RESET_ALL,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(debug=False, log_file=None):
    log_level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColorfulFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
    logging.basicConfig(level=log_level, handlers=handlers)
    logging.getLogger("langchain").setLevel(logging.WARNING)
