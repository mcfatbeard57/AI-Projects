import praw
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

def fetch_top_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter='week', limit=limit):
        posts.append({
            'reddit_id': post.id,
            'title': post.title,
            'score': post.score,
            'content': post.selftext[:500] + '...' if len(post.selftext) > 500 else post.selftext,
            'url': post.url,
            'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
            'num_comments': post.num_comments
        })
    return posts
