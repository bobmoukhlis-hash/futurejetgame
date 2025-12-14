import gradio as gr
import requests, io, base64, tempfile, os
from deep_translator import GoogleTranslator
from PIL import Image
import speech_recognition as sr
from gtts import gTTS

# ========================
# 🔑 CHIAVI API
# ========================
API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
conversation_history = []

# ========================
# ✅ Verifica Token Hugging Face
# ========================
def check_hf_token():
    try:
        test_prompt = "a cute robot cat smiling"
        url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        data = {"inputs": test_prompt}
        r = requests.post(url, headers=headers, json=data, timeout=30)
        if r.status_code == 200:
            return "✅ Token Hugging Face valido!"
        elif r.status_code == 401:
            return "❌ Token non valido o senza permessi!"
        else:
            return f"⚠️ Errore: {r.status_code}"
    except Exception as e:
        return f"⚠️ Errore: {str(e)}"

# ========================
# 💬 Chat AI
# ========================
def chat_ai(message, language):
    global conversation_history
    try:
        conversation_history.append({"role": "user", "content": message})
        headers = {"Authorization": f"Bearer {API_KEY}"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Sei un assistente utile e simpatico."}
            ] + conversation_history,
            "temperature": 0.7,
            "max_tokens": 500
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        text = r.json()["choices"][0]["message"]["content"].strip()
        if language == "it":
            text = GoogleTranslator(source="auto", target="it").translate(text)
        conversation_history.append({"role": "assistant", "content": text})
        return text
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# ========================
# 🖼️ Generatore Immagini
# ========================
def generate_image(prompt):
    try:
        url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            try:
                return Image.open(io.BytesIO(r.content))
            except:
                img_b64 = r.json().get("image_base64")
                if img_b64:
                    return Image.open(io.BytesIO(base64.b64decode(img_b64)))
                else:
                    return "⚠️ Nessuna immagine generata."
        else:
            return f"❌ Errore HuggingFace: {r.status_code}"
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# ========================
# 🎙️ Speech Recognition
# ========================
def voice_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# ========================
# 🔊 Text to Speech
# ========================
def text_to_speech(text, lang):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang=lang)
        tts.save(fp.name)
        return fp.name

# ========================
# 🧩 Interfaccia Gradio
# ========================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🤖 Chat AI FutureJetGame (Voce + Immagini)")
    with gr.Tab("💬 Chat"):
        text = gr.Textbox(label="Scrivi qui...", placeholder="Scrivi un messaggio...")
        lang = gr.Dropdown(["it", "en", "fr"], label="Lingua", value="it")
        audio = gr.Audio(label="🎤 Parla (opzionale)", type="filepath")
        btn = gr.Button("Invia")
        output = gr.Textbox(label="Risposta AI")
        btn.click(chat_ai, inputs=[text, lang], outputs=output)
    with gr.Tab("🖼️ Immagine"):
        img_prompt = gr.Textbox(label="Descrivi un'immagine")
        btn_img = gr.Button("Genera Immagine")
        img_out = gr.Image(label="Risultato")
        btn_img.click(generate_image, inputs=img_prompt, outputs=img_out)
    with gr.Tab("⚙️ Impostazioni"):
        test_btn = gr.Button("Verifica Token Hugging Face")
        status = gr.Textbox(label="Stato")
        test_btn.click(check_hf_token, outputs=status)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 10000)))
