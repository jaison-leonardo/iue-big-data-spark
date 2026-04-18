# Sistema de Clasificación de Imágenes Médicas (CNN)

## Descripción del Proyecto
Este ecosistema resuelve un problema completo de Big Data e Inteligencia Artificial orientado a predecir enfermedades (ej. Cáncer de Pulmón y Colon) a través de imágenes histopatológicas. 

Consta de una Red Neuronal Convolucional (CNN) programada en Keras/TensorFlow, un pipeline de Data Augmentation y compresión de arreglos NumPy para un manejo muy eficiente de la memoria RAM, una interfaz web visual generada maravillosamente con **Gradio**, y un sistema de notificaciones de dictamen médico en tiempo real vía SMS propulsado por **Twilio**.

---

## Estructura de Carpetas

```text
/
├── app/                  # Interfaz Front-End (Gradio)
│   └── app_gradio.py
├── data/                 # Depósitos de datos
│   ├── processed/        # Arreglos comprimidos generados (.npz)
│   └── raw/              # Aquí adentro van las carpetas con tus clases e imágenes
├── models/               # Archivos model.keras, historial .png y classes.json
├── src/                  # Dominio Backend Lógico
│   ├── preprocess.py     # Carga de la BD, escalado, división al vuelo y caché
│   ├── train.py          # Construcción Keras, EarlyStopping, validación Test
│   ├── infer.py          # Motor de predicción independiente
│   └── twilio_notify.py  # SDK de mensajería SMS Twilio
├── .env                  # Ubicación de Secretos y API Keys (Twilio)
├── requirements.txt      # Manifiesto de librerías en versiones estables 
├── main.py               # Orquestador General Interactivo (Entry Point Central)
└── README.md             # Documento guía y manual de uso
```

---

## Requisitos Previos e Instalación

Para asegurar compatibilidad plena con Tensorflow y evitar carencía de compilados, se recomienda fuertemente usar comandos estables nativos de **Python 3.12**.

1. **Crear y activar un entorno virtual separado:**
   * En Windows:
     ```powershell
     py -3.12 -m venv venv
     venv\Scripts\activate
     ```
   * En Linux/Mac:
     ```bash
     python3.12 -m venv venv
     source venv/bin/activate
     ```

2. **Instalar dependencias necesarias:**
   Asegúrate de estar en el directorio raíz ejecutando:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configurar Credenciales (.env):**
   Abre el archivo `.env` encriptado proporcionado y sustituye por tus keys privadas sacadas del panel desarrollador de [Twilio](https://www.twilio.com/):
   ```text
   TWILIO_ACCOUNT_SID=AC_tu_sid_real...
   TWILIO_AUTH_TOKEN=tu_token_real...
   TWILIO_PHONE_NUMBER=+123456789...
   ```

---

## Flujo de Ejecución (Orquestador Centralizado)

A diferencia de sistemas estáticos, **ya no necesitas ejecutar archivos secundarios uno por uno**. El ecosistema entero es gobernado de forma amistosa por el Orquestador nativo en la raíz del proyecto.

### Paso 1: Configurar los Datos Crudos (Dataset Kaggle)
Asegúrate de incorporar tus colecciones de imágenes en tu carpeta `data/raw/` separándolas por clase. 
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
Escribe este único comando mágico en la línea de la terminal:
```powershell
python main.py
```

Al hacerlo, serás recibido por un hermoso **Menú Interactivo Consolidado**. Simplemente tipea el número del módulo que necesites avanzar paso a paso:

> **Opciones Lógicas del Menú:**
> 1. **`[1] Extraer y Preprocesar Datos Crudos`**: Limpia inteligentemente memorias en caché muertas, procesa todas las imágenes ubicadas en `data/raw/`, las nivela y extrae matemática comprimida guardándola a salvo en el caché optimizado `.npz`.
> 2. **`[2] Entrenar y Evaluar Modelo CNN Keras`**: Requiere hacer concluido el paso 1 una sola vez. Conecta un aumento tensorial a los pixeles (Aumentation), entrena de forma robusta tus metadatos e incrusta el compilado final como `model.keras` en `models/`.
> 3. **`[3] Inferencia Aislada por Consola`** *(Depuración)*: Prueba puramente manual. Le insertas rutas estáticas de fotos como String y te devuelve el JSON de probabilidad de forma plana.
> 4. **`[4] Lanzar Plataforma Médica Operativa Final`**: Módulo culiminante. Levanta nativamente un servidor Gráfico sobre tu Web Local `127.0.0.1`. Permite arrastrar biopsias a la web e inyecta la API de *Twilio* por si el médico en turno requiere disparar el envío del dictámen vía SMS inmediatamente al celular del paciente.
