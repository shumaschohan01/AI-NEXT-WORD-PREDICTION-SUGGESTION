import gradio as gr
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Essential for loading older .h5 models in newer TensorFlow versions
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Load model and tokenizer
model = load_model("Next_Word_Pred_model.h5", compile=False)
with open("tokenizer (1).pkl", "rb") as f:
    tok = pickle.load(f)

max_len = 20 

def get_suggestions(seed_text):
    if not seed_text or len(seed_text.strip()) < 2:
        return ["", "", "", "", ""]

    # Process the input
    token_list = tok.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len, padding="pre")

    # Get probabilities for all words
    preds = model.predict(token_list, verbose=0)[0]
    
    # Get indices of the top 5 highest probabilities
    top_5_indices = np.argsort(preds)[-5:][::-1]
    
    suggestions = []
    for idx in top_5_indices:
        word = tok.index_word.get(idx, "")
        if word:
            # Show the full suggested sentence for each
            suggestions.append(f"{seed_text.strip()} {word}")
        else:
            suggestions.append("")
            
    # Ensure we always return exactly 5 items
    while len(suggestions) < 5:
        suggestions.append("")
        
    return suggestions

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 AI Autocomplete")
    gr.Markdown("Type below to see 5 real-time predictions.")
    
    search_input = gr.Textbox(
        show_label=False, 
        placeholder="Start typing...", 
        container=False
    )
    
    # Create 5 output boxes that look like a dropdown list
    with gr.Column():
        s1 = gr.Button("", variant="secondary", interactive=True)
        s2 = gr.Button("", variant="secondary", interactive=True)
        s3 = gr.Button("", variant="secondary", interactive=True)
        s4 = gr.Button("", variant="secondary", interactive=True)
        s5 = gr.Button("", variant="secondary", interactive=True)

    # This triggers the function EVERY time the user types (on change)
    search_input.change(
        fn=get_suggestions,
        inputs=search_input,
        outputs=[s1, s2, s3, s4, s5],
        show_progress="hidden" # Hidden to keep it feeling snappy
    )

    # Optional: Clicking a suggestion fills the search bar
    for btn in [s1, s2, s3, s4, s5]:
        btn.click(fn=lambda x: x, inputs=btn, outputs=search_input)

if __name__ == "__main__":
    demo.launch()