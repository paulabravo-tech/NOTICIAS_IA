"""
Agente diario de noticias de ECONOMIA -> Telegram.
2 noticias de Colombia + 3 del mundo.

Uso:
    python noticias_economia.py --hola   # solo envia un mensaje de prueba a Telegram
    python noticias_economia.py --seco   # lee noticias y Gemini elige, pero lo IMPRIME (no envia)
    python noticias_economia.py          # flujo completo, envia a Telegram

Variables de entorno necesarias:
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID      (siempre)
    GEMINI_API_KEY                        (--seco y flujo completo)
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

# Si alguna fuente falla o cambia de direccion, el programa la omite.
# Para agregar o quitar fuentes, edita estas listas.
FEEDS_COLOMBIA = {
    "Portafolio": "https://www.portafolio.co/rss/economia.xml",
    "La Republica": "https://www.larepublica.co/rss/economia.xml",
    "El Tiempo Economia": "https://www.eltiempo.com/rss/economia.xml",
    "Valora Analitik": "https://www.valoraanalitik.com/feed/",
    "Semana Economia": "https://www.semana.com/rss/economia.xml",
}

FEEDS_MUNDO = {
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "The Guardian Business": "https://www.theguardian.com/uk/business/rss",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "El Pais Economia": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada",
}

HORAS_ATRAS = 24
MAX_POR_REGION = 30       # maximo de candidatas por region que se le pasan a Gemini
CUANTAS_COLOMBIA = 2
CUANTAS_MUNDO = 3
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODELOS_RESPALDO = ["gemini-3.6-flash"]
ESPERAS_REINTENTO = [20, 60, 120]
USER_AGENT = "Mozilla/5.0 (compatible; NoticiasEconomia/1.0)"


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
            "text": texto[:4000],
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


def recolectar(feeds: dict) -> list[dict]:
    limite = datetime.now(timezone.utc) - timedelta(hours=HORAS_ATRAS)
    noticias = []

    for fuente, url in feeds.items():
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

    vistos_enlace, vistos_titulo, unicas = set(), set(), []
    for n in sorted(noticias, key=lambda x: x["fecha"], reverse=True):
        clave_t = normalizar_titulo(n["titulo"])
        if n["enlace"] in vistos_enlace or clave_t in vistos_titulo:
            continue
        vistos_enlace.add(n["enlace"])
        vistos_titulo.add(clave_t)
        unicas.append(n)

    return unicas[:MAX_POR_REGION]


# ---------------------------------------------------------------------------
# Seleccion con Gemini
# ---------------------------------------------------------------------------

def _llamar_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )

    modelos = [MODELO] + [m for m in MODELOS_RESPALDO if m != MODELO]
    ultimo_error = None
    for modelo in modelos:
        for intento, espera in enumerate(ESPERAS_REINTENTO + [None], 1):
            try:
                respuesta = cliente.models.generate_content(
                    model=modelo, contents=prompt, config=config
                )
                print(f"[ok] Gemini respondio con {modelo} (intento {intento})")
                return respuesta.text
            except Exception as e:  # noqa: BLE001
                codigo = getattr(e, "code", None) or getattr(e, "status_code", None)
                ultimo_error = e
                temporal = codigo in (429, 500, 502, 503, 504)
                print(f"[aviso] {modelo}, intento {intento}: {type(e).__name__} {codigo}")
                if codigo == 404:
                    break
                if not temporal:
                    raise
                if espera is None:
                    break
                time.sleep(espera)
    raise RuntimeError(f"Gemini no respondio con ningun modelo: {ultimo_error}")


def _formatear_lista(candidatas: list[dict]) -> str:
    return "\n".join(
        f"[{i}] ({n['fuente']}) {n['titulo']} -- {n['resumen']}"
        for i, n in enumerate(candidatas)
    )


def elegir_con_gemini(colombia: list[dict], mundo: list[dict]) -> dict:
    prompt = f"""Eres un editor de noticias economicas para una persona en Colombia que sigue la economia por trabajo.
Tienes dos listas de noticias de las ultimas 24 horas: COLOMBIA y MUNDO.

Elige {CUANTAS_COLOMBIA} noticias de la lista COLOMBIA y {CUANTAS_MUNDO} de la lista MUNDO.
Los indices de cada lista son independientes (cada una empieza en 0).

Criterios de relevancia, en orden:
1. Datos macroeconomicos y decisiones de bancos centrales (tasas, inflacion, PIB, empleo, dolar, petroleo).
2. Politica economica, fiscal o regulatoria con impacto real.
3. Movimientos de mercados y empresas grandes (resultados, fusiones, inversiones).
4. Comercio, aranceles y geopolitica con efecto economico.
Para MUNDO, prefiere lo que pueda afectar a Colombia o a America Latina.
Busca variedad: no repitas el mismo tema; si varias noticias hablan de lo mismo, elige solo la mejor.
Descarta rumores, opinion sin hechos, contenido patrocinado y publicidad.

Para cada noticia devuelve:
- "indice": el numero entre corchetes de su lista.
- "titulo": titulo en espanol, claro y breve.
- "resumen": 2 a 3 lineas en espanol que expliquen que paso y por que importa.
  No repitas la misma frase de cierre en varios resumenes.
Basate SOLO en la informacion de las listas; no inventes datos ni cifras.

Responde unicamente con un objeto JSON con esta forma:
{{"colombia": [ {{"indice": 0, "titulo": "...", "resumen": "..."}} ],
  "mundo": [ {{"indice": 0, "titulo": "...", "resumen": "..."}} ]}}

COLOMBIA:
{_formatear_lista(colombia)}

MUNDO:
{_formatear_lista(mundo)}
"""

    datos = json.loads(_llamar_gemini(prompt))

    def convertir(elegidas: list, base: list[dict], cuantas: int) -> list[dict]:
        resultado = []
        for item in elegidas[:cuantas]:
            try:
                i = int(item["indice"])
            except (KeyError, ValueError, TypeError):
                continue
            if not 0 <= i < len(base):
                continue  # indice invalido: se descarta en vez de inventar
            resultado.append(
                {
                    "titulo": item["titulo"],
                    "resumen": item["resumen"],
                    "enlace": base[i]["enlace"],  # el enlace sale de nuestros datos, no de Gemini
                    "fuente": base[i]["fuente"],
                }
            )
        return resultado

    return {
        "colombia": convertir(datos.get("colombia", []), colombia, CUANTAS_COLOMBIA),
        "mundo": convertir(datos.get("mundo", []), mundo, CUANTAS_MUNDO),
    }


# ---------------------------------------------------------------------------
# Mensaje
# ---------------------------------------------------------------------------

def _bloque(items: list[dict], inicio: int) -> list[str]:
    partes = []
    for n, item in enumerate(items, inicio):
        partes.append(
            f"<b>{n}. {html.escape(item['titulo'], quote=False)}</b>\n"
            f"{html.escape(item['resumen'], quote=False)}\n"
            f"<a href=\"{html.escape(item['enlace'], quote=True)}\">Leer en {html.escape(item['fuente'], quote=False)}</a>\n"
        )
    return partes


def armar_mensaje(sel: dict) -> str:
    hoy = datetime.now().strftime("%d/%m/%Y")
    partes = [f"<b>Noticias economicas de hoy</b> ({hoy})\n"]
    if sel["colombia"]:
        partes.append("<b>COLOMBIA</b>\n")
        partes += _bloque(sel["colombia"], 1)
    if sel["mundo"]:
        partes.append("<b>MUNDO</b>\n")
        partes += _bloque(sel["mundo"], len(sel["colombia"]) + 1)
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> None:
    args = set(sys.argv[1:])

    if "--hola" in args:
        enviar_telegram("Hola. Si lees esto, el bot de noticias economicas funciona.")
        print("Mensaje de prueba enviado.")
        return

    print("--- COLOMBIA ---")
    colombia = recolectar(FEEDS_COLOMBIA)
    print("--- MUNDO ---")
    mundo = recolectar(FEEDS_MUNDO)
    print(f"\nCandidatas tras limpiar duplicados: Colombia {len(colombia)}, Mundo {len(mundo)}")

    if not colombia and not mundo:
        mensaje = "Hoy no encontre noticias economicas recientes en las fuentes configuradas."
    else:
        try:
            sel = elegir_con_gemini(colombia, mundo)
        except Exception as e:  # noqa: BLE001
            # Plan B: Gemini no respondio. Se envian las mas recientes tal cual, sin resumen de IA.
            print(f"[aviso] Gemini fallo ({type(e).__name__}); se envia version sin IA")

            def crudo(lista: list[dict], cuantas: int) -> list[dict]:
                return [
                    {
                        "titulo": n["titulo"],
                        "resumen": n["resumen"][:200],
                        "enlace": n["enlace"],
                        "fuente": n["fuente"],
                    }
                    for n in lista[:cuantas]
                ]

            sel = {
                "colombia": crudo(colombia, CUANTAS_COLOMBIA),
                "mundo": crudo(mundo, CUANTAS_MUNDO),
            }
            mensaje = armar_mensaje(sel) + "\n<i>Nota: hoy Gemini no estuvo disponible; estas son las mas recientes, sin seleccion ni resumen por IA.</i>"
        else:
            if sel["colombia"] or sel["mundo"]:
                mensaje = armar_mensaje(sel)
            else:
                mensaje = "Gemini no devolvio noticias validas hoy."

    if "--seco" in args:
        print("\n--- VISTA PREVIA (no se envia) ---\n")
        print(mensaje)
        return

    enviar_telegram(mensaje)
    print("Reporte enviado a Telegram.")


if __name__ == "__main__":
    main()
