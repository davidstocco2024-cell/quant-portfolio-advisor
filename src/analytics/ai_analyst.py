import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def generate_market_context(ticker: str, asset_metrics: dict) -> dict:
    """
    Uses Gemini to generate a professional financial context for a specific asset.
    Returns a structured JSON with Macro Context, Key Risks, and a Recommendation.
    """
    if not client:
        return {"error": "GEMINI_API_KEY not found in .env file."}

    # Safely format the metrics (handles both floats and strings)
    vol = asset_metrics.get('Annualized Volatility', 'N/A')
    mdd = asset_metrics.get('Max Drawdown', 'N/A')
    sharpe = asset_metrics.get('Sharpe Ratio', 'N/A')

    # Format numbers if they are numeric, otherwise keep as is
    vol_str = f"{float(vol):.2f}" if isinstance(vol, (int, float)) else str(vol)
    mdd_str = f"{float(mdd):.2f}" if isinstance(mdd, (int, float)) else str(mdd)
    sharpe_str = f"{float(sharpe):.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)

    prompt = f"""
    Actúa como un Analista de Inversiones Senior. Estás escribiendo un breve informe para un cliente sobre el activo: {ticker}.
    
    Aquí están las métricas históricas de riesgo de este activo:
    - Volatilidad Anualizada: {vol_str}
    - Max Drawdown: {mdd_str}
    - Sharpe Ratio: {sharpe_str}
    
    Basándote en estas métricas y en tu conocimiento general del mercado actual (año 2024 en adelante), genera un análisis conciso.
    """
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "macro_context": types.Schema(
                    type=types.Type.STRING, 
                    description="Un párrafo corto explicando el contexto macroeconómico actual que afecta al activo."
                ),
                "risk_assessment": types.Schema(
                    type=types.Type.STRING, 
                    description="Evaluación del riesgo basada en las métricas proporcionadas. Menciona el Drawdown."
                ),
                "recommendation": types.Schema(
                    type=types.Type.STRING, 
                    description="Sugerencia de acción: 'Sobreponderar', 'Mantener' o 'Infraponderar'."
                )
            },
            required=["macro_context", "risk_assessment", "recommendation"]
        ),
        temperature=0.2 # Low temperature for more factual and consistent financial analysis
    )

    try:
        # Intento 1: El modelo más nuevo
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=config
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Primary model failed ({e}). Trying fallback model...")
        try:
            # Intento 2: Modelo alternativo (más estable si el primero se satura)
            response = client.models.generate_content(
                model='gemini-3.6-flash', 
                contents=prompt,
                config=config
            )
            return json.loads(response.text)
        except Exception as fallback_e:
            return {"error": f"Both AI models failed. Primary: {str(e)} | Fallback: {str(fallback_e)}"}