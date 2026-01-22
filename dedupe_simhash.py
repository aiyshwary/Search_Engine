# dedupe_simhash.py
import json
from simhash import Simhash

def simhash_of_text(text):
    return Simhash(text).value

def dedupe_docs(input_docs_jsonl, output_jsonl, threshold=3):
    seen = {}
    with open(input_docs_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = "".join([p.get("text","") for p in rec.get("pages",[])])
            h = simhash_of_text(text)
            found = False
            for key, val in seen.items():
                # compute hamming distance
                dist = Simhash(val).distance(Simhash(text))
                if dist <= threshold:
                    # mark duplicate
                    rec["duplicate_of"] = key
                    found = True
                    break
            if not found:
                seen[rec["doc_id"]] = text
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Hard-coded paths (edit these as you need)
    input_path = r"C:\Users\cmd\OneDrive\Documents\Search_Engine\output\documents.jsonl"
    output_path = r"C:\Users\cmd\OneDrive\Documents\Search_Engine\output\deduped_docs.jsonl"
    threshold = 3

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Threshold: {threshold}")

    dedupe_docs(input_path, output_path, threshold)

    print("[INFO] Dedupe completed.")
