#!/usr/bin/env python3
# main.py
"""
🤖 Super Mechs AI - Agente autónomo de aprendizaje
Ver la pantalla en tiempo real y jugar automáticamente
"""

import time
import sys
import cv2
import numpy as np
from screen_capture import ScreenCapture
from game_vision import GameVision
from game_state import GameState
from agent import SuperMechsEnv, SuperMechsAgent
import config

def print_banner():
    """Imprime el banner inicial"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║           🤖 SUPER MECHS AI - Auto Player 🤖          ║
    ║                                                       ║
    ║    La IA verá tu pantalla y jugará automáticamente   ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)

def check_setup():
    """Verifica que todo esté configurado correctamente"""
    print("\n🔍 Verificando configuración...")
    
    # Verificar resolución
    print(f"📺 Resolución esperada: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
    
    # Verificar que Google Play Games esté visible
    print("⚠️ Asegúrate de que Google Play Games esté abierto con Super Mechs")
    print("⚠️ La ventana debe estar en primer plano")
    
    input("Presiona ENTER cuando esté listo...")

def test_capture():
    """Prueba la captura de pantalla"""
    print("\n📸 Probando captura de pantalla...")
    
    sc = ScreenCapture()
    
    print("⏳ Capturando en 3 segundos...")
    time.sleep(3)
    
    screen = sc.get_screen()
    
    if screen is None:
        print("❌ Error: No se pudo capturar pantalla")
        return False
    
    print(f"✅ Captura exitosa: {screen.shape}")
    
    # Mostrar pantalla capturada
    if config.SHOW_SCREEN:
        print("📹 Mostrando pantalla capturada...")
        cv2.imshow("Captura de Pantalla", screen)
        print("Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True

def test_vision(screen):
    """Prueba la visión del juego"""
    print("\n👁️ Probando visión del juego...")
    
    gv = GameVision()
    analysis = gv.analyze_screen(screen)
    
    if analysis:
        print(f"✅ Análisis completado")
        print(f"  - Botones detectados: {len(analysis.get('buttons', []))}")
        print(f"  - Enemigos detectados: {len(analysis.get('enemies', []))}")
        print(f"  - Barras HP: {len(analysis.get('hp_info', {}).get('hp_bars', []))}")
        print(f"  - Cooldowns: {len(analysis.get('cooldowns', []))}")
        
        game_state = analysis.get('game_state', {})
        print(f"  - En batalla: {game_state.get('in_battle')}")
        print(f"  - Victoria: {game_state.get('battle_won')}")
        print(f"  - Derrota: {game_state.get('battle_lost')}")
        
        # Mostrar pantalla con análisis
        if config.SHOW_SCREEN:
            debug_screen = gv.draw_analysis(screen, analysis)
            cv2.imshow("Análisis Visual", debug_screen)
            print("Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
    
    print("❌ Error en análisis visual")
    return False

def main():
    """Función principal"""
    print_banner()
    
    # Verificar setup
    check_setup()
    
    # Prueba de captura
    if not test_capture():
        print("❌ Abortando...")
        return
    
    # Obtener pantalla para prueba visual
    print("\n📸 Capturando pantalla para análisis...")
    sc = ScreenCapture()
    time.sleep(2)
    test_screen = sc.get_screen()
    
    if test_screen is None:
        print("❌ No se pudo capturar pantalla")
        return
    
    # Prueba de visión
    if not test_vision(test_screen):
        print("❌ Problemas con visión del juego")
        return
    
    # Crear componentes
    print("\n🔧 Inicializando componentes...")
    
    screen_capture = ScreenCapture()
    game_vision = GameVision()
    game_state = GameState()
    
    print("✅ Componentes listos")
    
    # Crear entorno y agente
    print("\n🎮 Creando entorno de IA...")
    env = SuperMechsEnv(screen_capture, game_vision, game_state)
    agent = SuperMechsAgent(env)
    
    print("✅ Entorno listo")
    
    # Menú principal
    while True:
        print("\n" + "="*50)
        print("¿Qué deseas hacer?")
        print("="*50)
        print("1. Entrenar la IA (recomendado: 50000+ pasos)")
        print("2. Jugar con modelo entrenado")
        print("3. Cargar modelo existente")
        print("4. Ver estadísticas")
        print("5. Salir")
        print("="*50)
        
        choice = input("Elige opción (1-5): ").strip()
        
        if choice == "1":
            timesteps = input("¿Cuántos pasos de entrenamiento? (default: 100000): ").strip()
            try:
                timesteps = int(timesteps) if timesteps else 100000
            except:
                timesteps = 100000
            
            print(f"\n⚠️ IMPORTANTE:")
            print(f"- Mantén Google Play Games visible")
            print(f"- NO interrumpas el juego")
            print(f"- Entrena al menos 50000 pasos para buenos resultados")
            print(f"- Esto puede tomar 30+ minutos\n")
            
            confirm = input("¿Confirmas entrenar? (s/n): ").strip().lower()
            
            if confirm == 's':
                print("\n🚀 INICIANDO ENTRENAMIENTO...")
                print("La IA estará viendo y jugando automáticamente...\n")
                
                try:
                    agent.train(total_timesteps=timesteps)
                    
                    # Guardar modelo
                    save_choice = input("\n¿Guardar modelo? (s/n): ").strip().lower()
                    if save_choice == 's':
                        agent.save()
                    
                    print("✅ Entrenamiento completado")
                    
                except KeyboardInterrupt:
                    print("\n⛔ Entrenamiento interrumpido por el usuario")
                    
                    save_choice = input("¿Guardar progreso? (s/n): ").strip().lower()
                    if save_choice == 's':
                        agent.save()
                
                except Exception as e:
                    print(f"❌ Error durante entrenamiento: {e}")
        
        elif choice == "2":
            if agent.model is None:
                print("❌ No hay modelo entrenado. Primero entrena o carga uno.")
                continue
            
            episodes = input("¿Cuántos episodios? (default: 5): ").strip()
            try:
                episodes = int(episodes) if episodes else 5
            except:
                episodes = 5
            
            print(f"\n🎮 Jugando {episodes} episodios...")
            print("La IA jugar de manera automática...\n")
            
            try:
                agent.play(episodes=episodes)
            except KeyboardInterrupt:
                print("\n⛔ Juego interrumpido")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "3":
            model_path = input("Ruta del modelo (default: models/super_mechs_final): ").strip()
            model_path = model_path or f"{config.MODEL_DIR}/super_mechs_final"
            
            try:
                agent.load(model_path)
                print("✅ Modelo cargado")
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")
        
        elif choice == "4":
            stats = game_state.get_stats()
            print("\n📊 ESTADÍSTICAS:")
            print("="*50)
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            print("="*50)
        
        elif choice == "5":
            print("\n👋 ¡Adiós!")
            env.close()
            break
        
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
