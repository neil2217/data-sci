import config
import praw
import praw.models
import pandas as pd

reddit = praw.Reddit(client_id=config.REDDIT_CLIENT_ID, client_secret=config.REDDIT_CLIENT_SECRET, \
                     user_agent=config.REDDIT_USER_AGENT, username=config.REDDIT_USERNAME, password=config.REDDIT_PASSWORD)
reddit.read_only = True

sub = reddit.subreddit("todayilearned")
top_posts = []

for post in sub.hot(limit=50):
    title = post.title
    score = post.score
    url = post.url

    post.comments.replace_more(limit=0)
    top_level_comments = list(post.comments)
    if top_level_comments:
        top_comment = top_level_comments[0]
        if isinstance(top_comment, praw.models.Comment):
            top_comment_author = top_comment.author.name if top_comment.author else "Deleted"
            top_comment_text = top_comment.body
        else:
            top_comment_author = "Deleted"
            top_comment_text = "No Comments"
    else:
        top_comment_author = "Deleted"
        top_comment_text = "No Comments"

    top_posts.append({
        "Title": title, "Score": score, "URL": url, "Top comment Author": top_comment_author, "Top comment content": top_comment_text})


posts_df = pd.DataFrame(top_posts)
posts_df.to_csv("top_posts_todayilearned.csv", index=False)
print(posts_df)