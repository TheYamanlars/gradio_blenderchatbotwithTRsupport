import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Model ve tokenizer'ı yükle
model_name = "facebook/blenderbot-1B-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Sohbet fonksiyonu
def chat_fn(user_message, history):
    if history is None:
        history = []

    # Geçmişi tek stringe çevir
    conversation = ""
    for h in history:
        conversation += f"User: {h['user']} Bot: {h['bot']} "
    conversation += f"User: {user_message}"

    # Tokenize et
    inputs = tokenizer(conversation, return_tensors="pt")

    # Cevap üret
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )

    # Decode et
    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Geçmişe ekle
    history.append({"user": user_message, "bot": bot_reply})

    # Gradio'nun `messages` formatı
    messages = []
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["bot"]})

    return messages, history

# 3. Temizleme fonksiyonu
def clear_fn():
    return [], []

# 4. Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Chat with Meta AI's BlenderBot via Hugging Face")

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State([])  # Geçmişi saklar

    with gr.Row():
        txt = gr.Textbox(placeholder="Mesaj yazın...", show_label=False)
        send_btn = gr.Button("Gönder")
        clear_btn = gr.Button("Temizle")

    send_btn.click(chat_fn, [txt, state], [chatbot, state])
    txt.submit(chat_fn, [txt, state], [chatbot, state])
    clear_btn.click(clear_fn, [], [chatbot, state])

# 5. Uygulamayı başlat
demo.launch()
