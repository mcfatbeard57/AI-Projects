import streamlit as st
import pandas as pd
from reddit_fetcher import fetch_top_posts
from supabase_ops import store_categorized_posts, get_categorized_posts, store_subreddit
from categorization import categorize_posts, CATEGORIES

def subreddit_detail(subreddit_name):
    st.title(f"r/{subreddit_name}")

    # Create tabs for Top Posts and Themes
    tab1, tab2 = st.tabs(["Top Posts", "Themes"])

    with tab1:
        st.header("Top Posts")
        if 'top_posts' not in st.session_state or st.button('Refresh Top Posts'):
            with st.spinner('Fetching and categorizing top posts...'):
                subreddit_id = store_subreddit({"name": subreddit_name})
                
                # Fetch stored posts first
                stored_posts = get_categorized_posts(subreddit_id)
                stored_urls = {post['url'] for post in stored_posts}
                
                # Fetch new posts
                new_posts = fetch_top_posts(subreddit_name)
                
                # Identify which posts need categorization
                posts_to_categorize = [post for post in new_posts if post['url'] not in stored_urls]
                
                if posts_to_categorize:
                    categorized_new_posts = categorize_posts(posts_to_categorize)
                    store_categorized_posts(categorized_new_posts, subreddit_id)
                
                # Combine stored and new categorized posts
                all_posts = stored_posts + [post for post in new_posts if post['url'] not in stored_urls]
                
                st.session_state.top_posts = all_posts

        if hasattr(st.session_state, 'top_posts') and st.session_state.top_posts:
            df = pd.DataFrame(st.session_state.top_posts)
            st.dataframe(df[['title', 'score', 'num_comments', 'categories']].sort_values('score', ascending=False), use_container_width=True)
        else:
            st.info("No posts fetched yet. Click 'Refresh Top Posts' to fetch and categorize the latest top posts.")

    with tab2:
        st.header("Themes")
        if hasattr(st.session_state, 'top_posts') and st.session_state.top_posts:
            posts = st.session_state.top_posts
        else:
            subreddit_id = store_subreddit({"name": subreddit_name})
            posts = get_categorized_posts(subreddit_id)
        
        if posts:
            # Create a 3-column layout for theme cards
            col1, col2, col3 = st.columns(3)
            for i, category in enumerate(CATEGORIES):
                category_posts = [post for post in posts if category in post['categories']]
                with [col1, col2, col3][i % 3]:
                    with st.expander(f"{category} ({len(category_posts)})", expanded=False):
                        if len(category_posts) > 0:
                            for post in category_posts:
                                st.markdown(f"**[{post['title']}](https://www.reddit.com{post['url']})**")
                                st.markdown(f"Score: {post['score']} | Comments: {post['num_comments']}")
                                st.markdown(post['content'][:200] + "..." if len(post['content']) > 200 else post['content'])
                                st.markdown("---")
                        else:
                            st.info("No posts in this category.")
        else:
            st.info("No categorized posts available. Refresh the Top Posts to fetch and categorize new posts.")

    # Back button to return to the main page
    if st.button("Back to Subreddit List"):
        st.session_state.current_page = "main"
        st.experimental_rerun()
