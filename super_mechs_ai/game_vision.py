# game_vision.py
import cv2
import numpy as np
from PIL import Image
import pytesseract
import config

class GameVision:
    """Reconocimiento visual del juego Super Mechs"""
    
    def __init__(self):
        self.game_elements = {
            'buttons': [],
            'enemies': [],
            'hp_bars': [],
            'cooldowns': [],
            'text': {}
        }
    
    def analyze_screen(self, screen):
        """Analiza la pantalla completa y extrae información del juego"""
        if screen is None:
            return None
        
        analysis = {
            'raw_screen': screen,
            'buttons': self.detect_buttons(screen),
            'enemies': self.detect_enemies(screen),
            'hp_info': self.detect_hp_bars(screen),
            'cooldowns': self.detect_cooldowns(screen),
            'game_state': self.detect_game_state(screen),
            'text': self.extract_text(screen)
        }
        
        return analysis
    
    def detect_buttons(self, screen):
        """Detecta botones interactivos del juego"""
        buttons = []
        
        try:
            # Convertir a HSV para mejor detección
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Rango de colores para botones (típicamente rojo, azul, verde)
            # Rojo
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            
            # Azul
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Combinar máscaras
            mask = cv2.bitwise_or(mask_red, mask_blue)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filtrar por tamaño mínimo
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Verificar que sea rectangulado
                    if w > 30 and h > 30 and w < 400 and h < 200:
                        buttons.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'center_x': x + w // 2,
                            'center_y': y + h // 2,
                            'area': area
                        })
            
            # Eliminar duplicados (botones muy cercanos)
            buttons = self._remove_duplicates(buttons, distance=50)
            
        except Exception as e:
            print(f"❌ Error detectando botones: {e}")
        
        return buttons
    
    def detect_enemies(self, screen):
        """Detecta enemigos/mechas en pantalla"""
        enemies = []
        
        try:
            # Convertir a escala de grises y aplicar umbral
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Buscar contornos grandes (enemigos)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por tamaño (enemigos suelen ser grandes)
                if area > 2000 and area < 100000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    enemies.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2,
                        'area': area
                    })
            
            enemies = self._remove_duplicates(enemies, distance=100)
            
        except Exception as e:
            print(f"❌ Error detectando enemigos: {e}")
        
        return enemies
    
    def detect_hp_bars(self, screen):
        """Detecta barras de vida"""
        hp_info = {
            'player_hp': None,
            'player_hp_percent': None,
            'enemy_hp': None,
            'enemy_hp_percent': None,
            'hp_bars': []
        }
        
        try:
            # Buscar barras verdes (HP completo) y rojas (daño)
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Verde para HP
            lower_green = np.array([35, 100, 100])
            upper_green = np.array([85, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Rojo para daño
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            
            # Encontrar barras
            contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # HP bars suelen ser rectangulos horizontales
                if w > 50 and h < 30 and w > h:
                    hp_info['hp_bars'].append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'position': 'top' if y < screen.shape[0] // 2 else 'bottom'
                    })
            
        except Exception as e:
            print(f"❌ Error detectando barras de HP: {e}")
        
        return hp_info
    
    def detect_cooldowns(self, screen):
        """Detecta cooldowns de habilidades"""
        cooldowns = []
        
        try:
            # Buscar áreas semi-transparentes o con efecto de countdown
            # Típicamente en esquinas o abajo de la pantalla
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # Buscar áreas oscuras (overlay de cooldown)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Cooldowns suelen estar en áreas específicas
                if 30 < w < 150 and 30 < h < 150:
                    if y > screen.shape[0] - 200:  # Parte inferior
                        cooldowns.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h
                        })
            
        except Exception as e:
            print(f"❌ Error detectando cooldowns: {e}")
        
        return cooldowns
    
    def detect_game_state(self, screen):
        """Detecta el estado actual del juego"""
        state = {
            'in_battle': False,
            'battle_won': False,
            'battle_lost': False,
            'menu_open': False,
            'is_loading': False
        }
        
        try:
            # Detectar pantalla de victoria (suele tener colores verdes)
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Verde = victoria
            lower_green = np.array([35, 100, 100])
            upper_green = np.array([85, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            if cv2.countNonZero(mask_green) > (screen.shape[0] * screen.shape[1] * 0.1):
                state['battle_won'] = True
            
            # Rojo = derrota
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            
            if cv2.countNonZero(mask_red) > (screen.shape[0] * screen.shape[1] * 0.1):
                state['battle_lost'] = True
            
            # En batalla si hay enemigos detectados
            enemies = self.detect_enemies(screen)
            state['in_battle'] = len(enemies) > 0
            
        except Exception as e:
            print(f"❌ Error detectando estado del juego: {e}")
        
        return state
    
    def extract_text(self, screen):
        """Extrae texto de la pantalla con OCR"""
        text_data = {}
        
        try:
            # Aplicar preprocesamiento
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # OCR
            text = pytesseract.image_to_string(enhanced)
            text_data['all_text'] = text
            
            # Buscar palabras clave
            keywords = ['HP', 'Damage', 'Attack', 'Defense', 'Win', 'Lose', 'Level']
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    text_data[keyword.lower()] = True
            
        except Exception as e:
            print(f"⚠️ OCR no disponible (instala tesseract): {e}")
        
        return text_data
    
    def _remove_duplicates(self, elements, distance=50):
        """Elimina elementos duplicados/muy cercanos"""
        if not elements:
            return []
        
        filtered = []
        for elem in elements:
            is_duplicate = False
            for existing in filtered:
                dist = np.sqrt(
                    (elem['center_x'] - existing['center_x']) ** 2 +
                    (elem['center_y'] - existing['center_y']) ** 2
                )
                if dist < distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(elem)
        
        return filtered
    
    def draw_analysis(self, screen, analysis):
        """Dibuja los elementos detectados en la pantalla (para debug)"""
        if screen is None or analysis is None:
            return screen
        
        debug_screen = screen.copy()
        
        # Dibujar botones
        for button in analysis.get('buttons', []):
            cv2.rectangle(
                debug_screen,
                (button['x'], button['y']),
                (button['x'] + button['width'], button['y'] + button['height']),
                (0, 255, 0), 2
            )
            cv2.putText(debug_screen, "BTN", (button['x'], button['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dibujar enemigos
        for enemy in analysis.get('enemies', []):
            cv2.rectangle(
                debug_screen,
                (enemy['x'], enemy['y']),
                (enemy['x'] + enemy['width'], enemy['y'] + enemy['height']),
                (0, 0, 255), 2
            )
            cv2.putText(debug_screen, "ENEMY", (enemy['x'], enemy['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Dibujar HP bars
        for hp_bar in analysis.get('hp_info', {}).get('hp_bars', []):
            cv2.rectangle(
                debug_screen,
                (hp_bar['x'], hp_bar['y']),
                (hp_bar['x'] + hp_bar['width'], hp_bar['y'] + hp_bar['height']),
                (255, 0, 0), 2
            )
        
        # Dibujar cooldowns
        for cooldown in analysis.get('cooldowns', []):
            cv2.rectangle(
                debug_screen,
                (cooldown['x'], cooldown['y']),
                (cooldown['x'] + cooldown['width'], cooldown['y'] + cooldown['height']),
                (255, 255, 0), 2
            )
        
        return debug_screen
