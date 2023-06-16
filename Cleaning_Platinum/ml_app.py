import re
import gradio as gr

def data_processing(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

gradio_ui = gr.Interface(
    fn=data_processing,
    title="Data Processing and Modeling",
    description="Aplikasi Web Data Processing dan Modeling",
    inputs=gr.inputs.Textbox(lines=10, label="Paste some text here"),
    outputs=[
        gr.outputs.Textbox(label="Result"),
    ],
)

gradio_ui.launch()

