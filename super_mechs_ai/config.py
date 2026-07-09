# config.py
import os

# ========== PANTALLA ==========
SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 768

# Región del juego (ajusta según tu pantalla)
GAME_REGION = {
    'x': 0,
    'y': 0,
    'width': 1366,
    'height': 768
}

# ========== RUTAS ==========
MODEL_DIR = "./models"
LOG_DIR = "./logs"
SCREENSHOTS_DIR = "./screenshots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ========== CONFIGURACIÓN DE IA ==========
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
TOTAL_TIMESTEPS = 100000

# ========== DETECCIÓN ==========
MIN_CONFIDENCE = 0.5
GAME_CHECK_INTERVAL = 1  # segundos

# ========== DEBUG ==========
DEBUG_MODE = True
SHOW_SCREEN = True
SAVE_SCREENSHOTS = False
