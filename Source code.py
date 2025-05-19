import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load and clean dataset
# Enclose the filename in quotes to treat it as a string
df = pd.read_csv("DATASET.csv")  # Use your path if needed

# ✅ Drop rows with missing 'response'
df = df.dropna(subset=['response'])

# ✅ Extract instructions and responses
questions = df['instruction'].fillna("").tolist()
answers = df['response'].fillna("").tolist()

# ✅ TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# ✅ Get best match from user's query
def get_answer(user_input, history):
    if not user_input.strip():
        return "", history

    input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vec, question_vectors)
    best_idx = similarities.argmax()
    best_score = similarities[0][best_idx]

    if best_score < 0.3:
        bot_response = "🤖: Sorry, I couldn't understand that. Can you try rephrasing?"
    else:
        bot_response = f"🤖: {answers[best_idx]}"

    history.append((f"🧑: {user_input}", bot_response))
    return "", history

# ✅ Gradio UI
with gr.Blocks(css="""
    .gradio-container {background-color: #FAFAFA; font-family: 'Segoe UI', sans-serif;}
    .chatbox {min-height: 400px;}
    .message-input {margin-top: 10px;}
    .btn-clear {margin-top: 10px;}
""") as demo:
    gr.Markdown("## 🤖 Intelligent Customer Support Chatbot")

    chatbot = gr.Chatbot(label="💬 Chat History", elem_classes="chatbox")
    msg = gr.Textbox(placeholder="💬 Ask me anything about your order...", show_label=False, elem_classes="message-input")
    clear = gr.Button("🧹 Clear Chat", elem_classes="btn-clear")

    state = gr.State([])

    msg.submit(fn=get_answer, inputs=[msg, state], outputs=[msg, chatbot])
    clear.click(fn=lambda: ([], ""), inputs=[], outputs=[chatbot, state])

demo.launch()
