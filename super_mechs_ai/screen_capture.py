# screen_capture.py
import pyautogui
import cv2
import numpy as np
import time
from PIL import Image
import config

class ScreenCapture:
    def __init__(self):
        self.game_region = config.GAME_REGION
        self.last_frame = None
        
    def get_screen(self):
        """Captura la pantalla de Google Play Games en tiempo real"""
        try:
            # Capturar región de pantalla
            x = self.game_region['x']
            y = self.game_region['y']
            width = self.game_region['width']
            height = self.game_region['height']
            
            # Capturar screenshot
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            
            # Convertir a numpy array
            screen = np.array(screenshot)
            
            # Convertir RGB a BGR para OpenCV
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            
            self.last_frame = screen
            return screen
            
        except Exception as e:
            print(f"❌ Error capturando pantalla: {e}")
            return None
    
    def click(self, x, y, duration=0.1):
        """Hacer clic en coordenadas específicas"""
        try:
            pyautogui.click(x, y, duration=duration)
            time.sleep(0.05)
        except Exception as e:
            print(f"❌ Error al hacer clic: {e}")
    
    def double_click(self, x, y):
        """Doble clic"""
        try:
            pyautogui.doubleClick(x, y)
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error en doble clic: {e}")
    
    def swipe(self, x1, y1, x2, y2, duration=0.5):
        """Deslizar de un punto a otro (simula drag)"""
        try:
            pyautogui.moveTo(x1, y1)
            time.sleep(0.1)
            pyautogui.drag(x2 - x1, y2 - y1, duration=duration)
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error en deslizamiento: {e}")
    
    def hold_button(self, x, y, duration=1):
        """Mantener presionado un botón"""
        try:
            pyautogui.mouseDown(x, y)
            time.sleep(duration)
            pyautogui.mouseUp(x, y)
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error al mantener botón: {e}")
    
    def get_pixel_color(self, x, y):
        """Obtener color de un píxel específico"""
        try:
            if self.last_frame is not None:
                color = self.last_frame[y, x]
                return color
            return None
        except Exception as e:
            print(f"❌ Error obteniendo color: {e}")
            return None
    
    def save_frame(self, filename="frame.png"):
        """Guardar frame actual para debug"""
        try:
            if self.last_frame is not None:
                cv2.imwrite(f"screenshots/{filename}", self.last_frame)
                print(f"📸 Frame guardado: {filename}")
        except Exception as e:
            print(f"❌ Error guardando frame: {e}")
