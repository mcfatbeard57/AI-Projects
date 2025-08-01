# GPT-4 Chatbot with Moderation

A combined CLI & web UI chatbot that checks user input via OpenAI’s Moderation API before sending it to GPT-4.

## Requirements

- Python 3.7+  
- [openai](https://pypi.org/project/openai/)  
- [streamlit](https://pypi.org/project/streamlit/)  
- An OpenAI API key set in `OPENAI_API_KEY`

```bash
pip install openai streamlit
export OPENAI_API_KEY="your_api_key_here"

# CLI mode
python chatbot_moderation.py

# Web UI mode
streamlit run chatbot_moderation.py
