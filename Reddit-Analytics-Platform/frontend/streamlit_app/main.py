import streamlit as st
import re
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from supabase_ops import store_subreddit, get_stored_subreddits, get_categorized_posts, delete_subreddit
from subreddit_collections import collections_page
from categorization import CATEGORIES

# Default subreddits
DEFAULT_SUBREDDITS = [
    {"name": "ollama", "description": "Discussions about Ollama, an open-source AI model runner"},
    {"name": "openai", "description": "News and discussions about OpenAI and its technologies"},
    {"name": "python", "description": "A community for Python programmers"},
    {"name": "datascience", "description": "A place for data science practitioners and enthusiasts"},
]

def is_valid_subreddit_url(url):
    # Basic validation for Reddit URL
    pattern = r'^https?://(?:www\.)?reddit\.com/r/[\w-]+/?$'
    return re.match(pattern, url) is not None

def extract_subreddit_name(url):
    # Extract subreddit name from URL
    match = re.search(r'/r/([\w-]+)', url)
    return match.group(1) if match else None

def display_themes(subreddit_name):
    st.title(f"r/{subreddit_name} Themes")
    
    # Fetch categorized posts for the subreddit
    posts = get_categorized_posts(subreddit_name)
    
    # Group posts by category
    categorized_posts = {category: [] for category in CATEGORIES}
    for post in posts:
        for category in post.get('categories', []):
            if category in CATEGORIES:
                categorized_posts[category].append(post)
    
    # Display posts by category
    for category in CATEGORIES:
        with st.expander(f"{category} ({len(categorized_posts[category])})"):
            if categorized_posts[category]:
                for post in categorized_posts[category]:
                    st.write(f"**{post['title']}**")
                    st.write(f"Score: {post.get('score', 'N/A')} | Comments: {post.get('num_comments', 'N/A')}")
                    st.write(f"[Link to post]({post.get('url', '#')})")
                    st.write("---")
            else:
                st.write("No posts in this category.")

def main():
    st.set_page_config(page_title="Reddit Analytics Platform", layout="wide")

    # Initialize session state for current subreddit if not exists
    if 'current_subreddit' not in st.session_state:
        st.session_state.current_subreddit = None

    # Initialize session state for subreddits if not exists
    if 'subreddits' not in st.session_state:
        stored_subreddits = get_stored_subreddits()
        # Combine stored subreddits with default subreddits
        all_subreddits = DEFAULT_SUBREDDITS + [s for s in stored_subreddits if s['name'] not in [d['name'] for d in DEFAULT_SUBREDDITS]]
        st.session_state.subreddits = all_subreddits

    # Update the sidebar to include the new Collections page
    st.sidebar.title("Reddit Analytics Platform")
    page = st.sidebar.radio("Navigate", ["Home", "Collections"])

    if page == "Home":
        show_main_page()
    elif page == "Collections":
        collections_page()

def show_main_page():
    st.title("Reddit Analytics Platform")

    # Add Reddit button
    if st.button("Add Reddit"):
        st.session_state.add_reddit = True

    # Modal for adding new subreddit
    if st.session_state.get('add_reddit', False):
        with st.form("add_subreddit_form"):
            st.subheader("Add New Subreddit")
            subreddit_url = st.text_input("Paste Subreddit URL")
            submitted = st.form_submit_button("Submit")

            if submitted:
                if is_valid_subreddit_url(subreddit_url):
                    subreddit_name = extract_subreddit_name(subreddit_url)
                    new_subreddit = {"name": subreddit_name, "description": f"User-added subreddit: r/{subreddit_name}"}
                    store_subreddit(new_subreddit)
                    stored_subreddits = get_stored_subreddits()
                    # Add default description to stored subreddits if missing
                    for subreddit in stored_subreddits:
                        if 'description' not in subreddit:
                            subreddit['description'] = f"Subreddit: r/{subreddit['name']}"
                    all_subreddits = DEFAULT_SUBREDDITS + [s for s in stored_subreddits if s['name'] not in [d['name'] for d in DEFAULT_SUBREDDITS]]
                    st.session_state.subreddits = all_subreddits
                    st.success(f"Successfully added r/{subreddit_name}")
                    st.session_state.add_reddit = False
                else:
                    st.error("Invalid subreddit URL. Please enter a valid Reddit URL.")

    st.header("Available Subreddits")

    # Create a 2-column layout
    col1, col2 = st.columns(2)

    # Display subreddit cards
    for i, subreddit in enumerate(st.session_state.subreddits):
        with col1 if i % 2 == 0 else col2:
            st.subheader(f"r/{subreddit['name']}")
            st.write(f"**Description:** {subreddit.get('description', 'No description available')}")
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button(f"View Analytics for r/{subreddit['name']}", key=f"view_{subreddit['name']}"):
                    st.session_state.current_subreddit = subreddit['name']
            with button_col2:
                if st.button(f"Delete r/{subreddit['name']}", key=f"delete_{subreddit['name']}"):
                    delete_subreddit(subreddit['name'])
                    st.session_state.subreddits = [s for s in st.session_state.subreddits if s['name'] != subreddit['name']]
                    st.success(f"Deleted r/{subreddit['name']}")
                    st.experimental_rerun()

    # Display themes for the selected subreddit
    if st.session_state.current_subreddit:
        display_themes(st.session_state.current_subreddit)

if __name__ == "__main__":
    main()
