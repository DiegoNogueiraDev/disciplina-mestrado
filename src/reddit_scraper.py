#!/usr/bin/env python3
"""Simple Reddit scraper ‑ saves submissions into JSONL

Usage:
  python src/reddit_scraper.py --sub brasil --limit 2000 --days 60

Env vars (from .env or shell):
  REDDIT_ID, REDDIT_SECRET, REDDIT_AGENT
"""

import os, json, argparse, datetime as dt, sys
from pathlib import Path
import praw
from dotenv import load_dotenv
from prawcore.exceptions import ResponseException

load_dotenv()  # lê REDDIT_* do .env

parser = argparse.ArgumentParser(description="Scrape subreddit posts → JSONL")
parser.add_argument("--sub", required=True, help="Nome do subreddit, ex.: brasil")
parser.add_argument("--limit", type=int, default=1000, help="Máximo de posts")
parser.add_argument("--days", type=int, default=30, help="Quantos dias para trás")
parser.add_argument("--output", default=None, help="Caminho JSONL de saída")
args = parser.parse_args()

try:
    # Usa client credentials apenas (read-only)
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent=os.getenv("REDDIT_AGENT", "sentimento-bot/0.1"),
        read_only=True,
    )
    
    # Testa a autenticação fazendo uma requisição simples
    reddit.subreddit("test").display_name
    
except ResponseException as e:
    if e.response.status_code == 401:
        print(
            "Erro de autenticação (401 Unauthorized).\n"
            "Soluções:\n"
            "1. Se o app é 'personal use script': adicione seu username/password Reddit no .env\n"
            "2. Ou mude o app para tipo 'web app' em https://www.reddit.com/prefs/apps\n"
            "3. Verifique se REDDIT_ID é o código longo que aparece abaixo do nome do app",
            file=sys.stderr,
        )
    else:
        print(f"Erro na API do Reddit: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Erro inesperado: {e}", file=sys.stderr)
    sys.exit(1)

cutoff_ts = dt.datetime.utcnow() - dt.timedelta(days=args.days)
cutoff_ts = int(cutoff_ts.timestamp())

items = []
print(f"Buscando posts em r/{args.sub}...")
for post in reddit.subreddit(args.sub).new(limit=args.limit):
    if post.created_utc < cutoff_ts:
        break  # parou na janela de tempo
    items.append(
        {
            "id": post.id,
            "text": f"{post.title}\n{post.selftext}".strip(),
            "score": post.score,
            "created_utc": post.created_utc,
            "num_comments": post.num_comments,
        }
    )

out_path = args.output or f".data/raw/reddit_{args.sub}.jsonl"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    # Each object on a new line
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Salvo {len(items)} posts → {out_path}")