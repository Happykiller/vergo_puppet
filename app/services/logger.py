#app\services\logger.py
import logging
import inspect

# ANSI color codes for different log levels, used to enhance log readability in the console
LOG_COLORS = {
    'DEBUG': '\033[94m',   # Blue for debugging information
    'INFO': '\033[92m',    # Green for general information
    'WARNING': '\033[93m', # Yellow for warnings
    'ERROR': '\033[91m',   # Red for error messages
    'CRITICAL': '\033[95m' # Magenta for critical issues
}

# Code to reset console color after each log entry
RESET_COLOR = '\033[0m'

# Custom formatter class to apply color coding based on log level
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Determine color for the log level of the current record
        log_color = LOG_COLORS.get(record.levelname, RESET_COLOR)
        # Format the log message and apply color, then reset color at the end
        message = super().format(record)
        return f"{log_color}{message}{RESET_COLOR}"

# Dynamically retrieve the name of the calling module to label logs accurately
caller_frame = inspect.stack()[1]
module = inspect.getmodule(caller_frame[0])
logger_name = module.__name__ if module else '__main__'  # Fallback to '__main__' if module is not found

# Create a logger with the derived module name for global use
logger = logging.getLogger(logger_name)

# Only configure the logger if it hasn't been configured already (avoid duplicate handlers)
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    
    # Define a log format to include timestamp, function name, and log level
    formatter = CustomFormatter('[%(asctime)s][%(funcName)s][%(levelname)s] %(message)s')
    
    # Set up a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Attach the console handler to the logger
    logger.addHandler(console_handler)

# Prevent the logger from propagating messages to the root logger
logger.propagate = False
