# src/reddit_scraper.py
import os, json, datetime as dt, praw, argparse

def scrape(sub, limit=1000, days=30):
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent=os.getenv("REDDIT_AGENT")
    )
    # Timestamp filtro
    after = int((dt.datetime.utcnow() - dt.timedelta(days=days)).timestamp())
    items = []
    for post in reddit.subreddit(sub).new(limit=limit):
        if post.created_utc < after: break
        items.append({"id": post.id, "text": post.title + " " + post.selftext})
    return items

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sub", default="brasil")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()
    data = scrape(args.sub, args.limit, args.days)
    out = f".data/raw/reddit_{args.sub}.jsonl"
    with open(out, "w") as f:
        for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Salvo {len(data)} posts em {out}")
