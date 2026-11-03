"""
Agente diario de noticias de IA -> Telegram.

Uso:
    python noticias_ia.py --hola     # Etapa 1: solo envia un mensaje de prueba a Telegram
    python noticias_ia.py --seco     # Etapa 2: lee noticias y Gemini elige 5, pero lo IMPRIME (no envia)
    python noticias_ia.py            # Etapa 3: flujo completo, envia a Telegram

Variables de entorno necesarias:
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID      (siempre)
    GEMINI_API_KEY                        (etapas 2 y 3)
    GEMINI_MODEL                          (opcional, por defecto gemini-3.5-flash-lite)
"""

import calendar
import html
import json
import os
import re
import sys
import time
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
# Los modelos 2.5 fueron retirados por Google (404 / 503 persistente). Se usan los de la familia 3.x.
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Si el modelo principal esta saturado o ya no existe, se prueba con estos (en orden).
MODELOS_RESPALDO = ["gemini-3.6-flash"]
ESPERAS_REINTENTO = [10, 30, 60]  # segundos de espera entre reintentos del mismo modelo
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

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )

    # Reintentos: si el modelo esta saturado (503) o con limite (429), espera y reintenta;
    # si sigue fallando, pasa al siguiente modelo de la lista.
    modelos = [MODELO] + [m for m in MODELOS_RESPALDO if m != MODELO]
    ultimo_error = None
    respuesta = None
    for modelo in modelos:
        for intento, espera in enumerate(ESPERAS_REINTENTO + [None], 1):
            try:
                respuesta = cliente.models.generate_content(
                    model=modelo, contents=prompt, config=config
                )
                print(f"[ok] Gemini respondio con {modelo} (intento {intento})")
                break
            except Exception as e:  # noqa: BLE001
                codigo = getattr(e, "code", None) or getattr(e, "status_code", None)
                ultimo_error = e
                temporal = codigo in (429, 500, 502, 503, 504)
                print(f"[aviso] {modelo}, intento {intento}: {type(e).__name__} {codigo}")
                if codigo == 404:
                    break  # modelo retirado o inexistente: no se reintenta, pasa al siguiente modelo
                if not temporal:
                    raise  # error no temporal (clave invalida, modelo inexistente...): no tiene sentido reintentar
                if espera is None:
                    break  # agotados los reintentos de este modelo
                time.sleep(espera)
        if respuesta is not None:
            break

    if respuesta is None:
        raise RuntimeError(f"Gemini no respondio con ningun modelo: {ultimo_error}")

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
        try:
            items = elegir_con_gemini(candidatas)
        except Exception as e:  # noqa: BLE001
            # Aviso por Telegram para enterarte de que hoy fallo, y se marca la ejecucion como fallida.
            if "--seco" not in args:
                try:
                    enviar_telegram(f"El agente de noticias fallo hoy: {type(e).__name__}. Revisa la pestana Actions en GitHub.")
                except Exception:  # noqa: BLE001
                    pass
            raise
        mensaje = armar_mensaje(items) if items else "Gemini no devolvio noticias validas hoy."

    if "--seco" in args:
        print("\n--- VISTA PREVIA (no se envia) ---\n")
        print(mensaje)
        return

    enviar_telegram(mensaje)
    print("Reporte enviado a Telegram.")


if __name__ == "__main__":
    main()
