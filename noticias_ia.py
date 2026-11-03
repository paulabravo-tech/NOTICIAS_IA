"""
Agente diario de noticias de IA -> Telegram.

Uso:
    python noticias_ia.py --hola     # Etapa 1: solo envia un mensaje de prueba a Telegram
    python noticias_ia.py --seco     # Etapa 2: lee noticias y Gemini elige 5, pero lo IMPRIME (no envia)
    python noticias_ia.py            # Etapa 3: flujo completo, envia a Telegram

Variables de entorno necesarias:
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID      (siempre)
    GEMINI_API_KEY                        (etapas 2 y 3)
    GEMINI_MODEL                          (opcional, por defecto gemini-2.5-flash)
"""

import calendar
import html
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

# Fuentes iniciales. Si alguna falla o cambia de direccion, el programa la omite.
# Para agregar o quitar fuentes, edita esta lista.
FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Google AI": "https://blog.google/technology/ai/rss/",
    "TechCrunch IA": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge IA": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "MIT Technology Review": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    # VentureBeat desactivado: su feed no respondio en la prueba (puede bloquear lectores automaticos).
    # "VentureBeat IA": "https://venturebeat.com/category/ai/feed/",
    "Ars Technica": "https://arstechnica.com/ai/feed/",
}

HORAS_ATRAS = 24          # solo noticias de las ultimas 24 horas
MAX_CANDIDATAS = 60       # maximo de noticias que se le pasan a Gemini
CUANTAS = 5               # cuantas noticias quieres en el reporte
MODELO = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
USER_AGENT = "Mozilla/5.0 (compatible; NoticiasIA/1.0)"


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def enviar_telegram(texto: str) -> None:
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": texto[:4000],  # limite de Telegram: 4096 caracteres
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Telegram respondio {resp.status_code}: {resp.text}")


# ---------------------------------------------------------------------------
# Recoleccion y limpieza
# ---------------------------------------------------------------------------

def limpiar_html(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto or "")
    texto = html.unescape(texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_titulo(titulo: str) -> str:
    return re.sub(r"[^a-z0-9]", "", titulo.lower())


def recolectar() -> list[dict]:
    limite = datetime.now(timezone.utc) - timedelta(hours=HORAS_ATRAS)
    noticias = []

    for fuente, url in FEEDS.items():
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] {fuente}: error al leer el feed ({e})")
            continue

        if not feed.entries:
            print(f"[aviso] {fuente}: sin entradas (feed caido o direccion cambiada)")
            continue

        encontradas = 0
        for e in feed.entries:
            t = e.get("published_parsed") or e.get("updated_parsed")
            if not t:
                continue
            fecha = datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
            if fecha < limite:
                continue
            titulo = limpiar_html(e.get("title", ""))
            enlace = e.get("link", "")
            if not titulo or not enlace:
                continue
            noticias.append(
                {
                    "fuente": fuente,
                    "titulo": titulo,
                    "enlace": enlace,
                    "resumen": limpiar_html(e.get("summary", ""))[:400],
                    "fecha": fecha,
                }
            )
            encontradas += 1
        print(f"[ok] {fuente}: {encontradas} noticias recientes")

    # Quitar duplicados por enlace y por titulo parecido
    vistos_enlace, vistos_titulo, unicas = set(), set(), []
    for n in sorted(noticias, key=lambda x: x["fecha"], reverse=True):
        clave_t = normalizar_titulo(n["titulo"])
        if n["enlace"] in vistos_enlace or clave_t in vistos_titulo:
            continue
        vistos_enlace.add(n["enlace"])
        vistos_titulo.add(clave_t)
        unicas.append(n)

    return unicas[:MAX_CANDIDATAS]


# ---------------------------------------------------------------------------
# Seleccion con Gemini
# ---------------------------------------------------------------------------

def elegir_con_gemini(candidatas: list[dict]) -> list[dict]:
    from google import genai
    from google.genai import types

    cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    lista = "\n".join(
        f"[{i}] ({n['fuente']}) {n['titulo']} -- {n['resumen']}"
        for i, n in enumerate(candidatas)
    )

    prompt = f"""Eres un editor de noticias de inteligencia artificial.
De la siguiente lista de noticias de las ultimas 24 horas, elige las {CUANTAS} MAS RELEVANTES
para una persona que sigue la IA por trabajo.
Busca variedad: como maximo 2 de las noticias elegidas pueden tratar el mismo tema general.
Si un articulo parece contenido patrocinado o publicitario, descartalo.

Criterios de relevancia, en orden:
1. Lanzamientos de modelos, productos o herramientas importantes.
2. Cambios de regulacion o politica con impacto real.
3. Investigacion o avances tecnicos significativos.
4. Movimientos de negocio relevantes (inversiones, alianzas, adquisiciones).
Evita repetir el mismo tema: si varias noticias hablan de lo mismo, elige solo la mejor.
Ignora rumores, opinion sin hechos y publicidad.

Para cada una devuelve:
- "indice": el numero entre corchetes de la lista.
- "titulo": titulo en espanol, claro y breve.
- "resumen": 2 a 3 lineas en espanol que expliquen que paso y por que importa.
  No repitas la misma frase de cierre en varios resumenes ni fuerces un enfoque comun.
Basate SOLO en la informacion de la lista; no inventes datos.

Responde unicamente con un arreglo JSON de {CUANTAS} objetos.

NOTICIAS:
{lista}
"""

    respuesta = cliente.models.generate_content(
        model=MODELO,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    elegidas = json.loads(respuesta.text)
    resultado = []
    for item in elegidas[:CUANTAS]:
        i = int(item["indice"])
        if not 0 <= i < len(candidatas):
            continue  # indice invalido: se descarta en vez de inventar
        base = candidatas[i]
        resultado.append(
            {
                "titulo": item["titulo"],
                "resumen": item["resumen"],
                "enlace": base["enlace"],  # el enlace sale de nuestros datos, no de Gemini
                "fuente": base["fuente"],
            }
        )
    return resultado


# ---------------------------------------------------------------------------
# Mensaje
# ---------------------------------------------------------------------------

def armar_mensaje(items: list[dict]) -> str:
    hoy = datetime.now().strftime("%d/%m/%Y")
    partes = [f"<b>Las {len(items)} noticias de IA de hoy</b> ({hoy})\n"]
    for n, item in enumerate(items, 1):
        partes.append(
            f"<b>{n}. {html.escape(item['titulo'], quote=False)}</b>\n"
            f"{html.escape(item['resumen'], quote=False)}\n"
            f"<a href=\"{html.escape(item['enlace'], quote=True)}\">Leer en {html.escape(item['fuente'], quote=False)}</a>\n"
        )
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> None:
    args = set(sys.argv[1:])

    if "--hola" in args:
        enviar_telegram("Hola. Si lees esto, tu bot de Telegram funciona.")
        print("Mensaje de prueba enviado.")
        return

    candidatas = recolectar()
    print(f"\nCandidatas tras limpiar duplicados: {len(candidatas)}")

    if not candidatas:
        mensaje = "Hoy no encontre noticias recientes en las fuentes configuradas."
    else:
        items = elegir_con_gemini(candidatas)
        mensaje = armar_mensaje(items) if items else "Gemini no devolvio noticias validas hoy."

    if "--seco" in args:
        print("\n--- VISTA PREVIA (no se envia) ---\n")
        print(mensaje)
        return

    enviar_telegram(mensaje)
    print("Reporte enviado a Telegram.")


if __name__ == "__main__":
    main()
