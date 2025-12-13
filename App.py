# ======================================================
# 🤖 ChatAI Pro FLUX 2025 — Groq + Voce + Immagini (Render Version)
# ======================================================

!pip install -q gradio deep-translator Pillow requests gtts SpeechRecognition pydub
import gradio as gr
import requests, io, base64, tempfile, os
from deep_translator import GoogleTranslator
from PIL import Image
import speech_recognition as sr
from gtts import gTTS

# ======================================================
# 🔑 CHIAVI API (variabili d’ambiente per sicurezza)
# ======================================================
API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

conversation_history = []

# ======================================================
# 🔍 Verifica Token Hugging Face
# ======================================================
def check_hf_token():
    test_prompt = "a smiling cat wearing sunglasses"
    url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    r = requests.post(url, headers=headers, json={"inputs": test_prompt})
    if r.status_code == 200:
        return "✅ Token Hugging Face valido!"
    elif r.status_code == 401:
        return "❌ Token Hugging Face non valido o senza permessi Inference API!"
    else:
        return f"⚠️ Errore Hugging Face: {r.status_code} - {r.text[:150]}"

# ======================================================
# 💬 Chat AI con memoria e traduzione
# ======================================================
def chat_ai(message, language):
    global conversation_history
    try:
        if any(word in message.lower() for word in ["immagine", "disegna", "crea", "picture", "foto"]):
            img = generate_image(message)
            return img

        conversation_history.append({"role": "user", "content": message})
        headers = {"Authorization": f"Bearer {API_KEY}"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Rispondi come un assistente amichevole e utile."}
            ] + conversation_history,
            "temperature": 0.7,
            "max_tokens": 700
        }

        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=data, timeout=60)

        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            if language != "it":
                text = GoogleTranslator(source="auto", target=language).translate(text)
            conversation_history.append({"role": "assistant", "content": text})
            return text
        else:
            return f"❌ Errore {r.status_code}: {r.text}"
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# ======================================================
# 🎨 Generatore immagini (HuggingFace Router)
# ======================================================
def generate_image(prompt):
    try:
        hf_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt}
        r = requests.post(hf_url, headers=hf_headers, json=payload, timeout=120)
        if r.status_code == 200:
            try:
                return Image.open(io.BytesIO(r.content))
            except:
                img_b64 = r.json().get("image_base64")
                if img_b64:
                    return Image.open(io.BytesIO(base64.b64decode(img_b64)))
                else:
                    return "⚠️ Nessuna immagine restituita"
        elif r.status_code == 401:
            return "❌ Errore: token Hugging Face non autorizzato! Controlla i permessi (deve avere Access Inference API)."
        else:
            return f"❌ Errore Hugging Face: {r.status_code} - {r.text}"
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# ======================================================
# 🎙️ Speech → Text
# ======================================================
def voice_to_text(audio_file):
    if not audio_file:
        return ""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# ======================================================
# 🗣️ Text → Speech
# ======================================================
def text_to_speech(text, language):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang=language if language != "pt" else "pt-br")
        tts.save(fp.name)
        return fp.name

# ======================================================
# 💬 Gestione input principale (chat + voce + immagini)
# ======================================================
def process_inputs(text, lang, audio):
    if audio:
        text = voice_to_text(audio)
    response = chat_ai(text, lang)
    if isinstance(response, Image.Image):
        return None, None, response
    audio_resp = text_to_speech(response, lang)
    return response, audio_resp, None

# ======================================================
# 🧹 Pulisci memoria
# ======================================================
def clear_history():
    global conversation_history
    conversation_history = []
    return "🧠 Memoria chat cancellata!"

# ======================================================
# 🖥️ Interfaccia Gradio
# ======================================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 ChatAI Pro FLUX 2025 — Groq + Voce + Immagini")
    with gr.Tab("💬 Chat"):
        with gr.Row():
            inp_text = gr.Textbox(label="✍️ Messaggio", placeholder="Scrivi o parla...", lines=3)
            inp_audio = gr.Audio(label="🎤 Parla (opzionale)", type="filepath")
        with gr.Row():
            lang = gr.Dropdown(
                ["it", "en", "fr", "es", "de", "pt", "ar", "hi", "ja", "zh"],
                label="🌍 Lingua", value="it"
            )
            btn = gr.Button("🚀 Invia", variant="primary")

        out_text = gr.Textbox(label="💬 Risposta AI", lines=8, interactive=False)
        out_audio = gr.Audio(label="🔊 Ascolta la risposta")
        out_image = gr.Image(label="🖼️ Immagine generata", visible=True, height=400)

        btn.click(process_inputs, inputs=[inp_text, lang, inp_audio],
                  outputs=[out_text, out_audio, out_image])

    with gr.Accordion("⚙️ Impostazioni", open=False):
        with gr.Row():
            clear_btn = gr.Button("🧹 Pulisci memoria chat")
            check_btn = gr.Button("🔍 Verifica token Hugging Face")
        status = gr.Textbox(label="Stato", interactive=False)
        clear_btn.click(clear_history, outputs=status)
        check_btn.click(check_hf_token, outputs=status)

    gr.Markdown("💳 **Versione Premium:** [paypal.me/bobbob1979](https://www.paypal.me/bobbob1979)")

import gradio as gr

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=10000)
