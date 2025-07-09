# src/preprocess.py
import re, spacy, json, argparse, unicodedata as ud
nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])

def clean(text):
    text = ud.normalize("NFKD", text)  # remove acentos p/ BoW se quiser
    text = re.sub(r"http\S+", "", text)  # links
    text = re.sub(r"@\w+|#\w+", "", text)  # mentions & hashtags
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t) > 2])

def batch(in_path, out_path):
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            j = json.loads(line)
            j["clean"] = clean(j["text"])
            f_out.write(json.dumps(j, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-path")
    p.add_argument("--out-path")
    args = p.parse_args()
    batch(args.in_path, args.out_path)
