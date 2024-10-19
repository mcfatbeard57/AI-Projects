from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def store_posts(posts, subreddit_id):
    for post in posts:
        # Check if the post already exists
        existing = supabase.table("posts").select("*").eq("url", post['url']).execute()
        
        if len(existing.data) == 0:
            # If the post doesn't exist, insert it
            new_post = {
                'title': post['title'],
                'content': post['content'],
                'score': post['score'],
                'url': post['url'],
                'created_utc': post['created_utc'],
                'num_comments': post['num_comments'],
                'subreddit_id': subreddit_id,
                'category_name': 'Uncategorized',  # Default category, to be updated later
                'fetched_at': datetime.utcnow().isoformat()
            }
            supabase.table("posts").insert(new_post).execute()
        else:
            # If the post exists, update it
            supabase.table("posts").update({
                'title': post['title'],
                'content': post['content'],
                'score': post['score'],
                'num_comments': post['num_comments'],
                'fetched_at': datetime.utcnow().isoformat()
            }).eq("url", post['url']).execute()

def get_stored_posts(subreddit_id):
    response = supabase.table("posts").select("*").eq("subreddit_id", subreddit_id).execute()
    posts = response.data
    for post in posts:
        post['categories'] = post['category_name'].split(',') if post['category_name'] else []
    return posts

def store_subreddit(subreddit):
    # Check if the subreddit already exists
    existing = supabase.table("subreddits").select("*").eq("name", subreddit['name']).execute()
    
    if len(existing.data) == 0:
        # If the subreddit doesn't exist, insert it
        new_subreddit = {
            'name': subreddit['name'],
            'url': f"https://www.reddit.com/r/{subreddit['name']}/",
            'created_at': datetime.utcnow().isoformat(),
            'last_fetched': None
        }
        result = supabase.table("subreddits").insert(new_subreddit).execute()
        return result.data[0]['id']
    else:
        # If the subreddit exists, update last_fetched
        supabase.table("subreddits").update({
            'last_fetched': datetime.utcnow().isoformat()
        }).eq("name", subreddit['name']).execute()
        return existing.data[0]['id']

def get_stored_subreddits():
    response = supabase.table("subreddits").select("*").execute()
    return response.data

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

def store_categorized_posts(posts, subreddit_id):
    for post in posts:
        # Check if the post already exists
        existing = supabase.table("posts").select("*").eq("url", post['url']).execute()
        
        if len(existing.data) == 0:
            # If the post doesn't exist, insert it
            new_post = {
                'title': post['title'],
                'content': post['content'],
                'score': post['score'],
                'url': post['url'],
                'created_utc': post['created_utc'],
                'num_comments': post['num_comments'],
                'subreddit_id': subreddit_id,
                'category_name': ','.join(post['categories']),
                'fetched_at': datetime.utcnow().isoformat()
            }
            supabase.table("posts").insert(new_post).execute()
        else:
            # If the post exists, update it
            supabase.table("posts").update({
                'title': post['title'],
                'content': post['content'],
                'score': post['score'],
                'num_comments': post['num_comments'],
                'category_name': ','.join(post['categories']),
                'fetched_at': datetime.utcnow().isoformat()
            }).eq("url", post['url']).execute()

def get_categorized_posts(subreddit_name):
    print(f"Debug: Fetching posts for {subreddit_name}")  # Console debug output
    subreddit = supabase.table("subreddits").select("id").eq("name", subreddit_name).execute()
    if not subreddit.data:
        print(f"Debug: Subreddit {subreddit_name} not found")  # Console debug output
        return []
    
    subreddit_id = subreddit.data[0]['id']
    response = supabase.table("posts").select("*").eq("subreddit_id", subreddit_id).execute()
    posts = response.data
    print(f"Debug: Fetched {len(posts)} posts for {subreddit_name}")  # Console debug output
    
    for post in posts:
        post['categories'] = post['category_name'].split(',') if post['category_name'] else []
    
    return posts

# Add these new functions to the existing file

def create_collection(name):
    new_collection = {
        'name': name,
        'created_at': datetime.utcnow().isoformat()
    }
    result = supabase.table("collections").insert(new_collection).execute()
    return result.data[0]['id']

def get_collections():
    response = supabase.table("collections").select("*").execute()
    return response.data

def add_subreddit_to_collection(collection_id, subreddit_name):
    subreddit_id = store_subreddit({"name": subreddit_name})
    new_relation = {
        'collection_id': collection_id,
        'subreddit_id': subreddit_id
    }
    supabase.table("collection_subreddits").insert(new_relation).execute()

def get_collection_subreddits(collection_id):
    response = supabase.table("collection_subreddits")\
        .select("subreddits(*)")\
        .eq("collection_id", collection_id)\
        .execute()
    return [item['subreddits'] for item in response.data]

# Add this new function to the existing file

def delete_subreddit(subreddit_name):
    # Delete the subreddit from the subreddits table
    supabase.table("subreddits").delete().eq("name", subreddit_name).execute()
    
    # Delete all posts associated with this subreddit
    subreddit = supabase.table("subreddits").select("id").eq("name", subreddit_name).execute()
    if subreddit.data:
        subreddit_id = subreddit.data[0]['id']
        supabase.table("posts").delete().eq("subreddit_id", subreddit_id).execute()
    
    # Remove the subreddit from all collections
    supabase.table("collection_subreddits").delete().eq("subreddit_id", subreddit_id).execute()

def remove_subreddit_from_collection(collection_id, subreddit_name):
    subreddit = supabase.table("subreddits").select("id").eq("name", subreddit_name).execute()
    if subreddit.data:
        subreddit_id = subreddit.data[0]['id']
        supabase.table("collection_subreddits").delete().eq("collection_id", collection_id).eq("subreddit_id", subreddit_id).execute()
