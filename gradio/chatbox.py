import gradio as gr
from hugchat import hugchat
from hugchat.login import Login

# Global variables
messages = [{"role": "assistant", "content": "How may I help you?"}]
logged_in = False
chatbot = None


def login(email, password):
    global logged_in, chatbot
    try:
        sign = Login(email, password)
        cookies = sign.login()
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        logged_in = True
        return "Login successful. You can now start chatting!"
    except Exception as e:
        return f"Login failed: {str(e)}"


def chat(message, history):
    global messages, logged_in, chatbot

    if not logged_in:
        return "Please log in first.", history

    messages.append({"role": "user", "content": message})
    response = chatbot.chat(message)
    messages.append({"role": "assistant", "content": response})

    return response, history + [[message, response]]


with gr.Blocks() as demo:
    gr.Markdown("# Simple ChatBot")

    with gr.Tab("Login"):
        email_input = gr.Textbox(label="Enter E-mail:")
        password_input = gr.Textbox(label="Enter Password:", type="password")
        login_button = gr.Button("Login")
        login_output = gr.Textbox(label="Login Status")

        login_button.click(
            login, inputs=[email_input, password_input], outputs=login_output)

    with gr.Tab("Chat"):
        chatbot_interface = gr.ChatInterface(chat)

if __name__ == "__main__":
    demo.launch()
