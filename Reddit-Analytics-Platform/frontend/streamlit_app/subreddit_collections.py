import streamlit as st
from supabase_ops import create_collection, get_collections, add_subreddit_to_collection, get_collection_subreddits, get_categorized_posts, remove_subreddit_from_collection
from categorization import CATEGORIES

def display_collection_themes(collection_name, subreddits):
    st.subheader(f"Themes for Collection: {collection_name}")
    
    all_posts = []
    for subreddit in subreddits:
        posts = get_categorized_posts(subreddit['name'])
        for post in posts:
            post['subreddit_name'] = subreddit['name']  # Add subreddit name to each post
        all_posts.extend(posts)
    
    # Group posts by category
    categorized_posts = {category: [] for category in CATEGORIES}
    for post in all_posts:
        for category in post.get('categories', []):
            if category in CATEGORIES:
                categorized_posts[category].append(post)
    
    # Display posts by category without nesting expanders
    for category in CATEGORIES:
        st.markdown(f"### {category} ({len(categorized_posts[category])})")
        if categorized_posts[category]:
            for post in categorized_posts[category]:  # Show all posts in the category
                st.markdown(f"**{post['title']}** (r/{post['subreddit_name']})")
                st.markdown(f"Score: {post.get('score', 'N/A')} | Comments: {post.get('num_comments', 'N/A')}")
                st.markdown(f"[Link to post]({post.get('url', '#')})")
                st.write("---")
        else:
            st.write("No posts in this category.")
        st.write("---")

def collections_page():
    st.title("Subreddit Collections")

    # Create new collection
    st.header("Create New Collection")
    new_collection_name = st.text_input("Collection Name")
    if st.button("Create Collection"):
        if new_collection_name:
            create_collection(new_collection_name)
            st.success(f"Collection '{new_collection_name}' created successfully!")
            st.experimental_rerun()
        else:
            st.error("Please enter a name for the collection.")

    # Add subreddit to collection
    st.header("Add Subreddit to Collection")
    collections = get_collections()
    collection_select = st.selectbox("Select Collection", [c['name'] for c in collections])
    subreddit_name = st.text_input("Subreddit Name")
    if st.button("Add to Collection"):
        if collection_select and subreddit_name:
            collection_id = next(c['id'] for c in collections if c['name'] == collection_select)
            add_subreddit_to_collection(collection_id, subreddit_name)
            st.success(f"Added r/{subreddit_name} to collection '{collection_select}'")
            st.experimental_rerun()
        else:
            st.error("Please select a collection and enter a subreddit name.")

    st.markdown("---")

    # Display existing collections
    st.header("Existing Collections")
    for collection in collections:
        with st.expander(f"Collection: {collection['name']}", expanded=False):
            subreddits = get_collection_subreddits(collection['id'])
            st.write("Subreddits in this collection:")
            for subreddit in subreddits:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"- r/{subreddit['name']}")
                with col2:
                    if st.button(f"Remove", key=f"remove_{collection['id']}_{subreddit['name']}"):
                        remove_subreddit_from_collection(collection['id'], subreddit['name'])
                        st.success(f"Removed r/{subreddit['name']} from collection '{collection['name']}'")
                        st.experimental_rerun()
            
            # Display themes for the collection
            display_collection_themes(collection['name'], subreddits)

if __name__ == "__main__":
    collections_page()
