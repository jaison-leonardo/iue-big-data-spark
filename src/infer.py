"""
Módulo para la inferencia del modelo (predicción con Deep Learning).
Provee las dependencias lógicas para integrarse fácilmente con interfaces externas como Gradio.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

_MODEL = None
_CLASS_NAMES = None

def load_trained_model(model_path='models/model.keras', classes_path='models/classes.json'):
    """
    Carga el Keras Model y el Diccionario de Categorías, y lo guarda globalmente (Singleton style cache)
    """
    global _MODEL, _CLASS_NAMES

    # Si ya se encuentran ambos precargados (Útil cuando un servidor esté vivo), omitimos el re-cálculo
    if _MODEL is not None and _CLASS_NAMES is not None:
        return _MODEL, _CLASS_NAMES

    # Prevención de errores lógicos
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error Crítico: El modelo no existe en {model_path}. Por favor, entrena la red localmente ejecuta src/train.py.")
    
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Error Crítico: Diccionario de clases no hallado en {classes_path}")

    logging.info(f"Levantando el peso estructural del modelo Keras guardado en {model_path}...")
    _MODEL = tf.keras.models.load_model(model_path)
    
    # Reconstruyendo mapping
    with open(classes_path, 'r') as f:
        _CLASS_NAMES = json.load(f)
        
    logging.info(f"Modelo cargado y en standby. Clases para infererir: {_CLASS_NAMES}")
    return _MODEL, _CLASS_NAMES

def preprocess_image(image_input, expected_size=(128, 128)):
    if isinstance(image_input, str): 
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"La ruta local de imagen es inválida o el archivo faltante: {image_input}")
        
        ext = os.path.splitext(image_input)[-1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            raise ValueError("Invalidez por tipo de extensión, provee un plano JPEG/PNG")

        try:
            img = load_img(image_input, target_size=expected_size)
            img_array = img_to_array(img)
        except Exception as e:
            raise ValueError(f"Fallo durante la apertura tensorial con Keras: {str(e)}")       
    # Gradio (Arrays)
    elif isinstance(image_input, np.ndarray):
        img_array = image_input
        if img_array.shape[:2] != expected_size:
             img_tensor = tf.image.resize(img_array, expected_size)
             img_array = img_tensor.numpy()
    else:
        raise TypeError(f"Aporte invalido: Entregaste {type(image_input)}. Se requiere str_path o arreglo numpy.")

    if img_array.shape[-1] == 4:
         # Si contiene capa superficial extra (Canal Alfa), se borra (RGBA -> RGB)
         img_array = img_array[..., :3]

    # (Normalizar estricto, 0-1)
    img_array = img_array / 255.0
    
    # Amplificación al lote Batch que la red asimila siempre al procesar (Batch=1)
    img_batch = np.expand_dims(img_array, axis=0) # Pasa de (224, 224, 3) -> (1, 224, 224, 3)
    return img_batch

def predict_image(image_input):
    model, class_names = load_trained_model()
    img_batch = preprocess_image(image_input)
    logging.info("Ejecutando model.predict() sobre imagen particular ...")
    
    probabilities = model.predict(img_batch, verbose=0)[0] 
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    predicted_class = class_names[predicted_idx]
    
    all_prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    response = {
        "class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_prob_dict
    }
    logging.info(f"##  Veredicto Diagnóstico: '{predicted_class.upper()}'  (Confianza Acertada {confidence*100:.2f}%)  ##")
    return response
