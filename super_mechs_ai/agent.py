# agent.py
import numpy as np
from gym import Env
import gym.spaces as spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import config

class SuperMechsEnv(Env):
    """Entorno de Gym para Super Mechs"""
    
    def __init__(self, screen_capture, game_vision, game_state):
        super(SuperMechsEnv, self).__init__()
        
        self.screen_capture = screen_capture
        self.game_vision = game_vision
        self.game_state = game_state
        
        # Espacio de acciones: número de acciones posibles
        self.action_space = spaces.Discrete(20)  # Máximo 20 acciones
        
        # Espacio de observación: imagen de 84x84 en escala de grises
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84),
            dtype=np.uint8
        )
        
        self.steps = 0
        self.max_steps = 300
        self.last_reward = 0
        
    def step(self, action):
        """Ejecuta una acción en el juego"""
        
        # Obtener acciones disponibles
        available_actions = self.game_state.get_available_actions()
        
        # Validar que la acción esté dentro del rango disponible
        if action >= len(available_actions):
            action = 0
        
        action_data = available_actions[action]
        
        # Ejecutar acción
        if action_data['type'] == 'click_button':
            self.screen_capture.click(action_data['x'], action_data['y'])
            
        elif action_data['type'] == 'attack_enemy':
            self.screen_capture.click(action_data['x'], action_data['y'])
            
        elif action_data['type'] == 'wait':
            pass  # Solo esperar
        
        # Registrar acción
        self.game_state.record_action(action, action_data)
        
        # Esperar un poco para que el juego responda
        import time
        time.sleep(0.5)
        
        # Capturar nueva pantalla
        screen = self.screen_capture.get_screen()
        
        # Analizar pantalla
        analysis = self.game_vision.analyze_screen(screen)
        
        # Actualizar estado
        self.game_state.update(screen, analysis)
        
        # Obtener recompensa
        reward = self.game_state.get_reward()
        self.last_reward = reward
        
        # Obtener vector de estado
        obs = self._get_observation(screen)
        
        # Verificar si terminó
        done = self.game_state.is_done() or self.steps >= self.max_steps
        
        self.steps += 1
        
        return obs, reward, done, {}
    
    def reset(self):
        """Reinicia el entorno"""
        self.steps = 0
        self.game_state.reset()
        
        import time
        time.sleep(1)
        
        # Capturar pantalla inicial
        screen = self.screen_capture.get_screen()
        
        # Analizar
        analysis = self.game_vision.analyze_screen(screen)
        
        # Actualizar estado
        self.game_state.update(screen, analysis)
        
        # Obtener observación
        obs = self._get_observation(screen)
        
        return obs
    
    def render(self, mode='human'):
        """Muestra el estado actual del juego"""
        if config.SHOW_SCREEN and self.game_state.current_screen is not None:
            import cv2
            
            # Dibujar análisis
            debug_screen = self.game_vision.draw_analysis(
                self.game_state.current_screen,
                self.game_state.analysis
            )
            
            # Añadir información de estado
            stats = self.game_state.get_stats()
            
            info_text = f"Battle: {stats['current_battle']} | " \
                       f"W: {stats['wins']} L: {stats['losses']} | " \
                       f"WR: {stats['win_rate']}"
            
            cv2.putText(debug_screen, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Super Mechs AI - Real Time Vision", debug_screen)
            cv2.waitKey(1)
    
    def _get_observation(self, screen):
        """Convierte la pantalla en observación para la RL"""
        if screen is None:
            return np.zeros((84, 84), dtype=np.uint8)
        
        # Convertir a escala de grises
        if len(screen.shape) == 3:
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        else:
            gray = screen
        
        # Redimensionar a 84x84
        resized = cv2.resize(gray, (84, 84))
        
        return resized.astype(np.uint8)


class SuperMechsAgent:
    """Agente RL para jugar Super Mechs"""
    
    def __init__(self, env):
        self.env = env
        self.model = None
        self.total_timesteps = 0
        
    def train(self, total_timesteps=100000):
        """Entrena el agente"""
        print(f"\n🧠 Creando modelo PPO...")
        
        # Crear callback para guardar checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=config.MODEL_DIR,
            name_prefix="super_mechs_model"
        )
        
        # Crear modelo
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=config.LEARNING_RATE,
            n_steps=2048,
            batch_size=config.BATCH_SIZE,
            tensorboard_log=config.LOG_DIR
        )
        
        print(f"⚙️ Iniciando entrenamiento por {total_timesteps} pasos...")
        
        # Entrenar
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        self.total_timesteps = total_timesteps
        
        print(f"\n✅ Entrenamiento completado!")
        
        return self.model
    
    def save(self, path=None):
        """Guarda el modelo"""
        if self.model is None:
            print("⚠️ No hay modelo para guardar")
            return
        
        path = path or f"{config.MODEL_DIR}/super_mechs_final"
        self.model.save(path)
        print(f"💾 Modelo guardado en: {path}")
    
    def load(self, path):
        """Carga un modelo entrenado"""
        self.model = PPO.load(path)
        print(f"📂 Modelo cargado desde: {path}")
    
    def play(self, episodes=10):
        """Juega episodios con el modelo entrenado"""
        if self.model is None:
            print("❌ No hay modelo cargado")
            return
        
        print(f"\n🎮 Jugando {episodes} episodios...")
        
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # Predecir acción
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Ejecutar acción
                obs, reward, done, _ = self.env.step(action)
                
                total_reward += reward
                steps += 1
                
                # Renderizar
                self.env.render()
            
            stats = self.env.get_stats() if hasattr(self.env, 'get_stats') else {}
            print(f"Episodio {episode + 1}: Reward Total={total_reward:.2f}, Pasos={steps}")
        
        print(f"\n📊 Juego completado!")
        stats = self.env.game_state.get_stats()
        print(f"Estadísticas finales: {stats}")


import cv2
