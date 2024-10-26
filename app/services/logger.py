import logging
import inspect

# Codes de couleur ANSI pour les différents niveaux de log
LOG_COLORS = {
    'DEBUG': '\033[94m',   # Bleu
    'INFO': '\033[92m',    # Vert
    'WARNING': '\033[93m', # Jaune
    'ERROR': '\033[91m',   # Rouge
    'CRITICAL': '\033[95m' # Magenta
}

# Code pour réinitialiser la couleur
RESET_COLOR = '\033[0m'

# Formatter personnalisé pour ajouter des couleurs en fonction du niveau de log
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, RESET_COLOR)
        message = super().format(record)
        return f"{log_color}{message}{RESET_COLOR}"

# Récupérer dynamiquement le nom du module appelant
caller_frame = inspect.stack()[1]
module = inspect.getmodule(caller_frame[0])
logger_name = module.__name__ if module else '__main__'

# Créer un logger globalement utilisable
logger = logging.getLogger(logger_name)

# Configurer le logger s'il n'est pas déjà configuré
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    
    # Créer un formatteur
    formatter = CustomFormatter('[%(asctime)s][%(funcName)s][%(levelname)s]%(message)s')
    
    # Créer un handler de console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Ajouter le handler au logger
    logger.addHandler(console_handler)

# S'assurer que le logger n'est pas propagé vers le logger racine
logger.propagate = False