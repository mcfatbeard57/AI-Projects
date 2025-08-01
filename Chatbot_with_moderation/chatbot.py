#!/usr/bin/env python3
"""
chatbot_moderation.py

A simple CLI & Streamlit web chatbot with moderation and pluggable models (OpenAI or HuggingFace).  

Requirements:
- Python 3.7+
- openai (`pip install openai`)
- streamlit (`pip install streamlit`)
- transformers (`pip install transformers`)
- Set environment variables:
    - `OPENAI_API_KEY` for OpenAI
    - `MODEL_NAME` for choice of model. E.g.:
        - `openai:gpt-4` (default)
        - `openai:gpt-3.5-turbo`
        - `huggingface:llama-2-7b`

Usage:
  # CLI mode:
  $ MODEL_NAME=openai:gpt-4 python chatbot_moderation.py

  # Web UI mode:
  $ MODEL_NAME=huggingface:llama-2-7b streamlit run chatbot_moderation.py

This script will:
1. Take user input via CLI or Streamlit.
2. Run OpenAI's Moderation check.
3. If flagged, warn (no LLM call).
4. Otherwise, dispatch to the chosen model backend:
    - OpenAI API for `openai:` models
    - HuggingFace Transformers for `huggingface:` models
5. Display and maintain a short history of interactions.
"""
import os
import sys
import openai
from collections import deque

# Optional UI import
try:
    import streamlit as st
except ImportError:
    st = None

# Load OpenAI key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: Set OPENAI_API_KEY environment variable.")
    sys.exit(1)
openai.api_key = api_key

# Model selection via env
MODEL_NAME = os.getenv("MODEL_NAME", "openai:gpt-4")

# History size\HISTORY_SIZE = 10


def moderate_content(text: str) -> bool:
    try:
        result = openai.Moderation.create(input=text)
        return not result['results'][0]['flagged']
    except Exception as e:
        print(f"Moderation error: {e}")
        return False


def get_response(messages: list) -> str:
    """
    Dispatch to OpenAI or HuggingFace based on MODEL_NAME.
    """
    # OpenAI backend
    if MODEL_NAME.startswith("openai:"):
        model_id = MODEL_NAME.split(":", 1)[1]
        try:
            resp = openai.ChatCompletion.create(
                model=model_id,
                messages=messages
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI API error: {e}"

    # HuggingFace backend
    elif MODEL_NAME.startswith("huggingface:"):
        hf_model = MODEL_NAME.split(":", 1)[1]
        try:
            from transformers import pipeline
            gen = pipeline("text-generation", model=hf_model)
            # Flatten history into a prompt
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            prompt += "\nassistant:"
            output = gen(prompt, max_length=512, do_sample=True)
            return output[0]['generated_text']
        except Exception as e:
            return f"HuggingFace error: {e}"

    else:
        return f"Unrecognized MODEL_NAME: {MODEL_NAME}"


def run_cli():
    history = deque([], maxlen=HISTORY_SIZE)
    print(f"Chatbot using model {MODEL_NAME}. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        if not moderate_content(user_input):
            print("Warning: flagged as inappropriate.")
            continue

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(history)
        reply = get_response(messages)
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})


def run_streamlit():
    if st is None:
        raise ImportError("Install streamlit to run web UI.")

    st.title(f"Chatbot (Model: {MODEL_NAME})")
    if 'history' not in st.session_state:
        st.session_state.history = []

    user = st.text_input("You:", key="input")
    if st.button("Send") and user:
        if not moderate_content(user):
            st.warning("Flagged as inappropriate.")
        else:
            st.session_state.history.append({"role": "user", "content": user})
            msgs = [{"role": "system", "content": "You are a helpful assistant."}] + st.session_state.history
            resp = get_response(msgs)
            st.session_state.history.append({"role": "assistant", "content": resp})

    for m in st.session_state.history:
        prefix = "You:" if m['role']=='user' else "Assistant:"
        st.markdown(f"**{prefix}** {m['content']}")


if __name__ == "__main__":
    if "STREAMLIT_SERVER_PORT" in os.environ:
        run_streamlit()
    else:
        run_cli()
