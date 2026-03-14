import gradio as gr
from .chat_model import chat

with open("app/styles.css") as f:
    css = f.read()

def launch_ui():
    with gr.Blocks(css=css) as demo:
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Type your message...", label="")
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()
