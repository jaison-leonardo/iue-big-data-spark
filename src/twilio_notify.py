import os
import re
import logging
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

load_dotenv()

def validar_formato_numero(numero: str) -> bool:
    """
    Valida un formato internacional de teléfono. Se asume requerido el símbolo '+'
    seguido de código de país e inmediatamente los números (Ej: +57310...).
    """
    numero = numero.strip()
    # Verifica el (+) inicial y entre 10 a 14 digitos numéricos después
    patron_internacional = r'^\+\d{10,14}$'
    if re.match(patron_internacional, numero):
        return True
    return False

def send_sms(to_number: str, prediction: dict) -> tuple:
    """
    Envía el mensaje SMS de Alerta usando Twilio API Client.
    Valida credenciales y previene el crasheo retornando una tupla estándar.
    Retorna: (Booleano Exito, Mensaje String)
    """
    if not to_number or to_number.strip() == "":
        return False, "Sin número ingresado para enviar SMS."
        
    to_number = to_number.strip()
    
    # 1. Validación de Expresión Regular para evitar consumos api erróneos
    if not validar_formato_numero(to_number):
        msg = f"El número '{to_number}' no respeta el estándar. Requiere indicativo (ej: +573001234567)."
        logging.warning(msg)
        return False, msg

    # 2. Carga segura de variables (No Hardcode)
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_PHONE_NUMBER')

    if not account_sid or not auth_token or not from_number or 'tu_account_sid_aqui' in account_sid:
        msg = "Fallo de Autenticación: Credenciales vacías o sin configurar en el archivo local .env."
        logging.error(msg)
        return False, msg

    # 3. Formatear y construir texto final de notificación diagnosticada
    predicted_class = prediction.get("class", "Desconocido")
    confidence = prediction.get("confidence", 0.0)
    
    message_body = f"Resultado diagnóstico: {predicted_class.upper()} (confianza: {confidence*100:.2f}%)"

    try:
        logging.info("Lanzando Petición API a servidores de Twilio...")
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )
        
        msg_exito = f"El mensaje fue enviado correctamente hacia Twilio correctamente SID:[{message.sid}]"
        logging.info(msg_exito)
        return True, msg_exito

    except TwilioRestException as e:
        msg = f"Twilio rest error: Credenciales inválidas, límite excedido, o número no verificado. INFO: {e.msg}"
        logging.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Fallo General y desconocido de conexión de red mandando Twilio SMS: {str(e)}"
        logging.error(msg)
        return False, msg
