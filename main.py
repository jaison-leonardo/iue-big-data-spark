import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(message)s')

def print_banner():
    print("\n" + "=" * 65)
    print(" CLASIFICACIÓN MÉDICA CNN (Pulmón y Colon) ")
    print("=" * 65)

def mostrar_menu():
    print_banner()
    print("Selecciona una opción:")
    print(" 1. Extraer y Preprocesar Datos (Generar base .npz)")
    print(" 2. Construir, Entrenar y Evaluar Modelo CNN (models/keras)")
    print(" 3. Abrir Interfaz Web (Gradio + Envíos SMS Twilio)")
    print(" 4. Salir")
    print("-" * 65)
    
    opcion = input("-> Ingresa un número: ")
    return opcion.strip()

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)
    
    load_dotenv()
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    if not twilio_sid or "tu_account" in twilio_sid:
         logging.warning("(WARNING) Advertencia: Credenciales Twilio pendientes en el archivo .env")
    else:
         logging.info("(OK) Entorno de variables y motor base configurados correctamente.")
         
    while True:
        try:
            opc = mostrar_menu()
            
            if opc == '1':
                logging.info("\n>>> EXTRACCIÓN Y PREPROCESAMIENTO <<<")
                from src import preprocess
                preprocess.prepare_data(use_cache=False)
                input("\nPresiona ENTER para volver al menú...")
                
            elif opc == '2':
                logging.info("\n>>> COMPILACIÓN Y ENTRENAMIENTO <<<")
                from src import train
                train.main()
                input("\nPresiona ENTER para volver al menú...")
                
            elif opc == '3':
                logging.info("\n>>> WEB APP FRONT-END <<<")
                from app import app_gradio
                app_gradio.main()
                break # Servidor bloqueará el terminal, así que apagamos loop menu
                
            elif opc == '0':
                 print("Cerrando servidor central. ¡Buen día!")
                 break
                 
            else:
                 print("\n(ERROR) Opción no reconocida. Selecciona números del 0 al 4.")
                 
        except KeyboardInterrupt:
             print("\nInterrupción detectada (Ctrl+C). Abortando proceso actual.")
        except Exception as e:
             logging.error(f"Fallo durante la invocación: {str(e)}")
             input("Presiona ENTER para continuar...")

if __name__ == "__main__":
    main()
