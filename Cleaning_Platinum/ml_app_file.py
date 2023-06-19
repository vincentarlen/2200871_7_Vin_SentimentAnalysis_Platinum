import re
import gradio as gr
import pandas as pd

def data_processing(file_obj):

    # Import file csv ke Pandas
    df = pd.read_csv(file_obj.name)

    # Ambil teks yang akan diproses dalam format list
    texts = df.text.to_list()

    # Lakukan cleansing pada teks
    cleaned_text = []
    for text in texts:
        cleaned_text.append(re.sub(r'[^a-zA-Z0-9]', ' ', text))

    return cleaned_text


gradio_ui = gr.Interface(data_processing, 
                            title="Data Processing and Modeling",
                            description="Aplikasi Web Data Processing dan Modeling",
                            inputs="file", 
                            outputs="json")

gradio_ui.launch()

