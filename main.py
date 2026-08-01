from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from groq import Groq
import json
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Ruta raíz para que muestre tu index.html si existe
@app.get("/", response_class=HTMLResponse)
def home():
    archivo_html = "index.html"
    if os.path.exists(archivo_html):
        with open(archivo_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Bienvenido a Guardian AI Backend 🚀</h1><p>La API está funcionando correctamente.</p>"

# ... el resto de tus rutas de análisis (/api/analizar-resultados, etc.) ...
# 1. Configurar el cliente de Groq con tu key gsk_...
API_KEY = os.environ.get("GROQ_API_KEY", "gsk_AQUI_PEGAS_TU_KEY")
client = Groq(api_key=API_KEY)

# 2. Inicializar FastAPI
app = FastAPI(title="TriageLab AI - Groq Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analizar-resultados")
async def analizar_resultados(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")

    try:
        # 3. Extraer texto del PDF
        pdf_reader = PyPDF2.PdfReader(file.file)
        texto_extraido = ""
        for page in pdf_reader.pages:
            texto_extraido += page.extract_text() + "\n"

        if not texto_extraido.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")

        # 4. Prompt y reglas
        system_prompt = """
        Actúa como 'Guardian AI', un asistente especializado en traducir resultados de laboratorio clínico a un lenguaje humano, empático y sencillo para pacientes ansiosos.
        
        REGLAS CRÍTICAS (GUARDRAILS):
        - TIENES PROHIBIDO EMITIR DIAGNÓSTICOS.
        - Si hay un valor que ponga en riesgo inminente la vida, el status debe ser "Crítico" y debes sugerir consultar al médico de inmediato.
        - Explica de forma calmada qué significa cada valor.
        
        Extrae los 2 o 3 valores más relevantes del siguiente texto de análisis clínico y devuélvelos ESTRICTAMENTE en formato JSON plano (un array de objetos con las propiedades: test_name, status, explanation).
        """

        user_prompt = f"TEXTO DEL ANÁLISIS:\n{texto_extraido}"

        # 5. Llamada a Groq (usando Llama 3 o modelos rápidos disponibles)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # 6. Parsear respuesta
        respuesta_ia = completion.choices[0].message.content
        
        # Groq con json_object devuelve un objeto {"resultados": [...]} o directamente el array
        # Aseguramos el formato adaptándolo a lo que espera el frontend
        datos_parseados = json.loads(respuesta_ia)
        
        # Si la IA devolvió un objeto contenedor en vez de lista directa
        if isinstance(datos_parseados, dict):
            resultados_json = datos_parseados.get("resultados", list(datos_parseados.values())[0])
        else:
            resultados_json = datos_parseados

        return {
            "mensaje": "Análisis completado con éxito",
            "resultados": resultados_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")