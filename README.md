# Sistema de Clasificación de Imágenes Médicas (CNN)

## Descripción del Proyecto
Este ecosistema resuelve un problema completo de Big Data e Inteligencia Artificial orientado a predecir enfermedades (ej. Cáncer de Pulmón y Colon) a través de imágenes histopatológicas. 

Consta de una Red Neuronal Convolucional (CNN) programada en Keras/TensorFlow, un pipeline de Data Augmentation y compresión de arreglos NumPy en memoria RAM, una interfaz web visual construida con **Gradio**, y un sistema de notificaciones vía SMS propulsado por **Twilio**.

---

## Estructura de Carpetas

```text
/
├── app/                  # Interfaz Front-End (Gradio)
│   └── app_gradio.py
├── data/                 # Directorio de datos
│   ├── processed/        # Arreglos comprimidos generados (.npz)
│   └── raw/              # Aquí adentro van las carpetas con las clases e imágenes
├── models/               # Archivos model.keras, historial .png y classes.json
├── src/                  # Dominio Backend Lógico
│   ├── preprocess.py     # Carga de la BD, escalado, división y caché
│   ├── train.py          # Construcción Keras, EarlyStopping, validación Test
│   ├── infer.py          # Motor de predicción
│   └── twilio_notify.py  # SDK de mensajería SMS Twilio
├── .env                  # Ubicación de Secretos y API Keys (Twilio)
├── requirements.txt      # Manifiesto de librerías en versiones estables 
├── main.py               # Orquestador General Interactivo (Entry Point Central)
└── README.md             # Documento guía y manual de uso
```

---

## Requisitos Previos e Instalación

Para asegurar compatibilidad plena con Tensorflow y evitar carencía de compilados, se recomienda usar comandos nativos de **Python 3.12**.

1. **Crear y activar un entorno virtual separado:**
   * En Windows:
     ```powershell
     py -3.12 -m venv venv
     venv\Scripts\activate
     ```

2. **Instalar dependencias necesarias:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configurar Credenciales (.env):**
   Abre el archivo `.env` encriptado proporcionado y sustituye por las keys privadas sacadas del panel desarrollador de [Twilio](https://www.twilio.com/):
   ```text
   TWILIO_ACCOUNT_SID=AC_tu_sid_real...
   TWILIO_AUTH_TOKEN=tu_token_real...
   TWILIO_PHONE_NUMBER=+123456789...
   ```

---

## Flujo de Ejecución (Orquestador Centralizado)

### Paso 1: Configurar los Datos Crudos (Dataset Kaggle)
Las colecciones de imágenes deben estar en la carpeta `data/raw/` separadas por clase. 
*(El nombre explícito que tengan estas subcarpetas será el que adopte la máquina y el Frontend como su diagnóstico visual)*. Ejemplo:
```text
data/raw/
   ├── Cancer_de_Colon_Adenocarcinoma/
   │    ├── img1.jpg
   │    └── img2.jpg
   └── Tejido_Normal_de_Pulmon/
        ├── img1.jpg
        └── img2.jpg
```

### Paso 2: Invocar el CLI Maestro
Para ejecutar:
```powershell
python main.py
```

Se mostrara un menu con las siguientes opciones:

> **Opciones Lógicas del Menú:**
> 1. **`1. Extraer y Preprocesar Datos Crudos`**: Limpia memorias en caché, procesa todas las imágenes ubicadas en `data/raw/`, las nivela y extrae el archivo comprimido guardándola a en el caché optimizado `.npz`.
> 2. **`2. Entrenar y Evaluar Modelo CNN Keras`**: Requiere haber concluido el paso 1 una sola vez. Conecta un aumento tensorial a los pixeles (Aumentation), entrena los metadatos e incrusta el compilado final como `model.keras` en `models/`.
> 3. **`3. Ejecuta el servidor WEB Gradio`**: Ejecuta un servidor Gráfico sobre la Web Local `127.0.0.1`. Permite seleccionar imagenes e inyecta la API de *Twilio* para el envío del notificaciones vía SMS.
