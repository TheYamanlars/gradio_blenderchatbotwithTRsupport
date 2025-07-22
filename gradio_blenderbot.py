import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Upload model and tokenizer
model_name = "facebook/blenderbot-1B-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Chat function
def chat_fn(user_message, history):
    if history is None:
        history = []

    # convert to one string
    conversation = ""
    for h in history:
        conversation += f"User: {h['user']} Bot: {h['bot']} "
    conversation += f"User: {user_message}"

    # Tokenize 
    inputs = tokenizer(conversation, return_tensors="pt")

    # response 
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )

    # Decode 
    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # past history append
    history.append({"user": user_message, "bot": bot_reply})

    # Gradio `messages` format
    messages = []
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["bot"]})

    return messages, history

# 3. cleaning func
def clear_fn():
    return [], []

# 4. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Chat with Meta AI's BlenderBot via Hugging Face")

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State([])  # Hide past conversation

    with gr.Row():
        txt = gr.Textbox(placeholder="Enter message...", show_label=False)
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")

    send_btn.click(chat_fn, [txt, state], [chatbot, state])
    txt.submit(chat_fn, [txt, state], [chatbot, state])
    clear_btn.click(clear_fn, [], [chatbot, state])

# 5. start the app
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
