import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json
import re

load_dotenv()

CATEGORIES = [
    "Solution Requests",
    "Pain & Anger",
    "Self-Promotion",
    "News",
    "Money Talk",
    "Advice Requests",
    "Ideas",
    "AI/Technology",
    "Help",
    "Something Else"
]

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

class PostCategories(BaseModel):
    categories: List[str] = Field(description="List of categories that apply to the post")

def extract_categories_fallback(output):
    # Look for anything that looks like a list of categories
    match = re.search(r'\[.*?\]', output)
    if match:
        try:
            categories = json.loads(match.group(0))
            return [cat for cat in categories if cat in CATEGORIES]
        except json.JSONDecodeError:
            pass
    return ["Uncategorized"]

def categorize_post(post):
    prompt = PromptTemplate(
        template="Analyze the following Reddit post and determine which categories it belongs to. Respond with a JSON object containing only a 'categories' key with a list of applicable categories. Do not include any additional text or explanation.\n\nPost Title: {title}\nPost Content: {content}\n\nCategories:\n{categories}\n\nResponse format:\n{{'categories': ['Category1', 'Category2', ...]}}\n\nYour response (JSON only):",
        input_variables=["title", "content", "categories"],
    )

    # Format the prompt with our data
    _input = prompt.format_prompt(title=post['title'], 
                                  content=post['content'], 
                                  categories=", ".join(CATEGORIES))

    # Initialize the Ollama model
    model = Ollama(model=OLLAMA_MODEL)

    # Generate the output
    output = model.invoke(_input.to_string())
    
    print(f"Raw output from model: {output}")  # Debug print

    try:
        parsed_output = json.loads(output)
        if 'categories' in parsed_output:
            return parsed_output['categories']
        else:
            return extract_categories_fallback(output)
    except json.JSONDecodeError:
        return extract_categories_fallback(output)
    except Exception as e:
        print(f"Error processing output: {e}")
        print(f"Raw output: {output}")
        return ["Uncategorized"]

def categorize_posts(posts):
    for post in posts:
        post['categories'] = categorize_post(post)
    return posts

# Done
