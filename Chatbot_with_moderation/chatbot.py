#!/usr/bin/env python3
"""
chatbot_moderation.py

A simple CLI & Streamlit web chatbot with OpenAI moderation and GPT-4.

Requirements:
- Python 3.7+
- openai package (install with `pip install openai`)
- streamlit package (install with `pip install streamlit`)
- Set the environment variable OPENAI_API_KEY with your OpenAI API key.

Usage:
  # CLI mode:
  $ python chatbot_moderation.py

  # Web UI mode:
  $ streamlit run chatbot_moderation.py

This script will:
1. Prompt the user for input (CLI) or show a web UI (Streamlit).
2. Check the input with OpenAI's Moderation API.
3. If the content is flagged, warn and skip the GPT-4 call.
4. Otherwise, send the conversation history + new user message to GPT-4.
5. Display the assistant's response and maintain a short history of interactions.
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

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: The OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)
openai.api_key = api_key

# Maximum number of past messages to keep in history
HISTORY_SIZE = 10


def moderate_content(text: str) -> bool:
    """
    Returns True if content is safe, False if flagged by moderation.
    """
    try:
        response = openai.Moderation.create(input=text)
        return not response['results'][0]['flagged']
    except Exception as e:
        print(f"Moderation API error: {e}")
        return False


def get_gpt4_response(messages: list) -> str:
    """
    Sends the message history to GPT-4 and returns the assistant's reply.
    """
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling GPT-4 API: {e}"


def run_cli():
    history = deque([], maxlen=HISTORY_SIZE)
    print("Welcome to the GPT-4 Chatbot with Moderation (CLI). Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not moderate_content(user_input):
            print("Warning: Your message was flagged and not sent to the model.")
            continue

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(history)
        reply = get_gpt4_response(messages)
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})


def run_streamlit():
    if st is None:
        raise ImportError("Streamlit is not installed. Install with `pip install streamlit`.")

    st.set_page_config(page_title="GPT-4 Chatbot with Moderation")
    st.title("GPT-4 Chatbot with Moderation")

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:", key="input")
    if st.button("Send", key="send") and user_input:
        if not moderate_content(user_input):
            st.warning("Your message was flagged as inappropriate.")
        else:
            st.session_state.history.append({"role": "user", "content": user_input})
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + st.session_state.history
            reply = get_gpt4_response(messages)
            st.session_state.history.append({"role": "assistant", "content": reply})

    # Display chat history
    for msg in st.session_state.history:
        if msg['role'] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")


if __name__ == "__main__":
    # Detect Streamlit environment by its server port var
    if "STREAMLIT_SERVER_PORT" in os.environ:
        run_streamlit()
    else:
        run_cli()
