# game_state.py
import numpy as np
import cv2
import config

class GameState:
    """Mantiene y actualiza el estado actual del juego"""
    
    def __init__(self):
        self.current_screen = None
        self.analysis = None
        self.battle_active = False
        self.last_action = None
        self.action_history = []
        self.game_started = False
        self.battle_count = 0
        self.wins = 0
        self.losses = 0
        
    def update(self, screen, analysis):
        """Actualiza el estado del juego con nueva pantalla"""
        self.current_screen = screen
        self.analysis = analysis
        
        if analysis and 'game_state' in analysis:
            game_state = analysis['game_state']
            
            # Detectar cambios en batalla
            if game_state.get('in_battle') and not self.battle_active:
                self.battle_active = True
                self.battle_count += 1
                print(f"🎮 Batalla #{self.battle_count} iniciada")
            
            elif not game_state.get('in_battle') and self.battle_active:
                self.battle_active = False
                
                if game_state.get('battle_won'):
                    self.wins += 1
                    print(f"✅ ¡VICTORIA! Total: {self.wins} victorias")
                elif game_state.get('battle_lost'):
                    self.losses += 1
                    print(f"❌ DERROTA. Total: {self.losses} derrotas")
    
    def get_state_vector(self, screen):
        """Convierte la pantalla en un vector para la RL"""
        if screen is None:
            return np.zeros(7056)  # 84x84 aplanado
        
        # Redimensionar y normalizar
        resized = cv2.resize(screen, (84, 84))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0
        
        # Aplanar
        vector = normalized.flatten()
        
        return vector
    
    def get_available_actions(self):
        """Retorna las acciones disponibles según el estado actual"""
        actions = []
        
        if not self.analysis:
            return [0]  # Acción por defecto
        
        # Si hay botones, podemos hacer clic
        buttons = self.analysis.get('buttons', [])
        for i, button in enumerate(buttons):
            actions.append({
                'type': 'click_button',
                'index': i,
                'x': button['center_x'],
                'y': button['center_y'],
                'button_data': button
            })
        
        # Si hay enemigos, podemos atacar (clic en ellos)
        enemies = self.analysis.get('enemies', [])
        for i, enemy in enumerate(enemies):
            actions.append({
                'type': 'attack_enemy',
                'index': i,
                'x': enemy['center_x'],
                'y': enemy['center_y'],
                'enemy_data': enemy
            })
        
        # Acción de no hacer nada (esperar)
        actions.append({
            'type': 'wait',
            'x': None,
            'y': None
        })
        
        return actions if actions else [{'type': 'wait', 'x': None, 'y': None}]
    
    def record_action(self, action_index, action_data):
        """Registra una acción ejecutada"""
        self.last_action = {
            'index': action_index,
            'data': action_data,
            'timestamp': len(self.action_history)
        }
        self.action_history.append(self.last_action)
    
    def get_reward(self):
        """Calcula la recompensa actual"""
        reward = 0
        
        if not self.analysis or not self.analysis.get('game_state'):
            return reward
        
        game_state = self.analysis['game_state']
        
        # Recompensa por victoria
        if game_state.get('battle_won'):
            reward = 10.0
        
        # Penalización por derrota
        elif game_state.get('battle_lost'):
            reward = -5.0
        
        # Pequeña recompensa por estar en batalla
        elif game_state.get('in_battle'):
            reward = 0.1
        
        return reward
    
    def is_done(self):
        """Retorna si la batalla ha terminado"""
        if not self.analysis or not self.analysis.get('game_state'):
            return False
        
        game_state = self.analysis['game_state']
        return game_state.get('battle_won') or game_state.get('battle_lost')
    
    def get_stats(self):
        """Retorna estadísticas del juego"""
        total_battles = self.wins + self.losses
        win_rate = (self.wins / total_battles * 100) if total_battles > 0 else 0
        
        return {
            'total_battles': total_battles,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': f"{win_rate:.1f}%",
            'current_battle': self.battle_count,
            'battle_active': self.battle_active
        }
    
    def reset(self):
        """Reinicia el estado para una nueva batalla"""
        self.action_history = []
        self.last_action = None
        self.battle_active = False
