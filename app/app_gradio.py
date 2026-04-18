"""
Módulo Front-End interactivo para inferencias con Gradio + Funcionalidad Notificaciones
"""
import sys
import os
import logging
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import gradio as gr
from src.infer import predict_image
# Integración de la dependencia del Servidor SMS
from src.twilio_notify import send_sms

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def gradio_predict_wrapper(image_array, country_str: str, input_phone: str):
    """
    Recibe la imagen, el menú desplegable del país y un string del teléfono desnudo. Orquesta la conexión dual
    solicitando inferencia e invocando el módulo de SMS si existe input adicional.
    """
    if image_array is None:
        return "Petición Inválida...", None, "SMS omitido a falta de imagen."

    try:
        # 1. Llamado a Inteligencia Artificial local
        prediction_result = predict_image(image_array)
        
        predicted_class = prediction_result.get("class", "Clase Desconocida")
        confidence = prediction_result.get("confidence", 0.0)
        all_probs_dict = prediction_result.get("all_probabilities", {})
        
        resumen_txt = f"Predicción resultante catalogada como: {predicted_class.upper()} (Precisión al {confidence*100:.2f}%)"
        
        # 2. Llamado a Invocación Twilio en base a los metadatos predichos
        estado_sms = "No insertaste teléfono. No se envió mensajería externa."
        
        # Validar si el cliente efectivamente rellenó el textbox numérico
        if input_phone and input_phone.strip() != "":
            # Extraer sólo el indicativo inicial ignorando paréntesis. (ej: de "+57 (Colombia) extrae "+57")
            prefijo_pais = country_str.split(" ")[0] if country_str else ""
            clean_phone = input_phone.replace(" ", "").replace("-", "").strip()
            
            # Por si el usuario igual insertó el '+' manualmente a pesar del desplegable, lo respetamos
            full_number = clean_phone if clean_phone.startswith("+") else f"{prefijo_pais}{clean_phone}"
            
            sms_aprobado, msj_consola = send_sms(full_number, prediction_result)
            if sms_aprobado:
                estado_sms = f"{msj_consola}"
            else:
                estado_sms = f"{msj_consola}"

        return resumen_txt, all_probs_dict, estado_sms

    except FileNotFoundError as e:
        error_msg = f"Faltan componentes estructurales u modelo: {str(e)} "
        logging.error(error_msg)
        return error_msg, None, "SMS bloqueado por caida parcial."
    except Exception as e:
        error_msg = f"Error Interno:\n{str(e)}"
        logging.error("Traceback Completo del Motor:")
        traceback.print_exc()
        return error_msg, None, "SMS bloqueado por error 500."


def main():
    
    with gr.Blocks(title="Diagnósticos Médicos Automáticos", theme=gr.themes.Base()) as interface:
        
        gr.Markdown("<h1 style='text-align: center'>Evaluador de Imágenes de Diagnóstico Médico</h1>")
        
        with gr.Row():
            # CAJA IZQUIERDA: Recepción de variables de usuario
            with gr.Column(scale=1):
                image_input = gr.Image(label="Imagen radiológica / patológica (.jpg)", type="numpy")
                
                gr.Markdown("### Notificación SMS")
                # División interna nativa de Gradio para colocar Controles lado a lado
                with gr.Row():
                    country_input = gr.Dropdown(
                        choices=[
                            "+57 (Colombia)", 
                            "+1 (USA/Canadá)", 
                            "+52 (México)", 
                            "+34 (España)", 
                            "+54 (Argentina)", 
                            "+56 (Chile)", 
                            "+51 (Perú)", 
                            "+55 (Brasil)"
                        ],
                        value="+57 (Colombia)",
                        label="País (Buscador)",
                        filterable=True, # Activa un buscador automático de teclado al cliquear
                        scale=1
                    )
                    phone_input = gr.Textbox(
                        label="Digita tu Número de Celular", 
                        placeholder="Ej: 3001233524...",
                        scale=2
                    )
                
                predict_btn = gr.Button("Realizar Predición", variant="primary")
            
            # CAJA DERECHA: Visualización y status
            with gr.Column(scale=1):
                text_output = gr.Textbox(label="Resumen", interactive=False, lines=2)
                
                # Componente Transparente: Log e info sobre mensajeria Twilio
                sms_output = gr.Textbox(label="Monitor de envio Twilio", interactive=False, lines=1)
                
                label_output = gr.Label(label="Detalle", num_top_classes=4)

        # Emparejar Click a inputs múltiples y salidas múltiples
        predict_btn.click(
            fn=gradio_predict_wrapper,
            inputs=[image_input, country_input, phone_input],
            outputs=[text_output, label_output, sms_output]
        )
        
    logging.info("Exponiendo API Gradio Frontend...")
    interface.launch(server_name="127.0.0.1", show_error=True)

if __name__ == "__main__":
    main()
