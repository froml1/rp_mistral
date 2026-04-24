import json
import argparse
from datetime import datetime
from pathlib import Path


def load_messages(filepath):
    """
    Charge un fichier JSON.
    Accepte :
    - liste directe [...]
    - objet {"messages":[...]}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "messages" in data:
        return data["messages"]

    raise ValueError(f"Format JSON non reconnu : {filepath}")


def parse_timestamp(msg, field):
    ts = msg.get(field)

    if not ts:
        return datetime.min

    formats = [
        "%Y-%m-%d %H:%M:%S",      # 2023-10-21 10:55:27
        "%Y-%m-%dT%H:%M:%S",      # ISO sans timezone
        "%Y-%m-%dT%H:%M:%S.%f",   # ISO ms
    ]

    for fmt in formats:
        try:
            return datetime.strptime(ts, fmt)
        except:
            pass

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except:
        return datetime.min


def normalize_author(msg, default_name):
    """
    Tente de récupérer auteur Discord.
    """
    if "author" in msg:
        if isinstance(msg["author"], dict):
            return msg["author"].get("name", default_name)
        return msg["author"]

    return default_name


def main():
    parser = argparse.ArgumentParser(
        description="Fusionner 2 exports Discord triés par timestamp"
    )

    parser.add_argument("--file-a", required=True, help="JSON auteur A")
    parser.add_argument("--file-b", required=True, help="JSON auteur B")
    parser.add_argument("--output", default="merged.json", help="Fichier final")
    parser.add_argument("--timestamp-field", default="timestamp", help="Champ timestamp")
    parser.add_argument("--name-a", default="Auteur A")
    parser.add_argument("--name-b", default="Auteur B")

    args = parser.parse_args()

    msgs_a = load_messages(args.file_a)
    msgs_b = load_messages(args.file_b)

    merged = msgs_a + msgs_b
    # injecte auteur si absent
    clean_merged = []
    for m in merged:
        mm = {}
        for k in m:
            if "content" in k.lower(): mm["content"] = m[k]
            else: mm[k.lower()] = m[k]
        mm["author"] = normalize_author(m, args.name_a)
        clean_merged.append(mm)

    # tri chrono
    clean_merged.sort(key=lambda x: parse_timestamp(x, args.timestamp_field))

    # export
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean_merged, f, indent=2, ensure_ascii=False, default=str)

    print(f"Fusion terminée : {args.output}")
    print(f"Messages totaux : {len(clean_merged)}")


if __name__ == "__main__":
    main()