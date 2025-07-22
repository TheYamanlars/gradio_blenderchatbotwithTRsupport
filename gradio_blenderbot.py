import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Load model and tokenizer
model_name = "facebook/blenderbot-1B-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Chat function
def chat_fn(user_message, history):
    if history is None:
        history = []

    # Convert history into a single conversation string
    conversation = ""
    for h in history:
        conversation += f"User: {h['user']} Bot: {h['bot']} "
    conversation += f"User: {user_message}"

    # Tokenize input
    inputs = tokenizer(conversation, return_tensors="pt")

    # Generate model response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )

    # Decode token IDs into string
    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Append to history
    history.append({"user": user_message, "bot": bot_reply})

    # Convert history into Gradio messages format
    messages = []
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["bot"]})

    return messages, history

# 3. Clear conversation function
def clear_fn():
    return [], []

# 4. Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Chat with Meta AI's BlenderBot via Hugging Face")

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State([])  # Keeps chat history

    with gr.Row():
        txt = gr.Textbox(placeholder="Type your message...", show_label=False)
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")

    send_btn.click(chat_fn, [txt, state], [chatbot, state])
    txt.submit(chat_fn, [txt, state], [chatbot, state])
    clear_btn.click(clear_fn, [], [chatbot, state])

# 5. Launch the app
demo.launch()
