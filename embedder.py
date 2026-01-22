# embedder.py
import json, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_VERSION = "v1-paraphrase-multi-miniLM"
BATCH = 256

model = SentenceTransformer(MODEL_NAME)

def run(chunks_jsonl, out_vectors_np, out_meta_jsonl):
    texts = []
    metas = []
    with open(chunks_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append({"chunk_id": rec["chunk_id"], "doc_id": rec["doc_id"]})
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in tqdm(range(0, n, BATCH)):
        batch_texts = texts[i:i+BATCH]
        emb = model.encode(
            batch_texts,
            convert_to_numpy=True,
            batch_size=BATCH,
            normalize_embeddings=True   # 🔥 improves FAISS cosine
        )

        vectors[i:i+len(emb)] = emb

    # save vectors
    np.save(out_vectors_np, vectors)

    # save meta with model version and index positions
    with open(out_meta_jsonl, "w", encoding="utf-8") as fout:
        for i, m in enumerate(metas):
            m.update({"index": i, "model_version": MODEL_VERSION})
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run(r"C:\Users\cmd\OneDrive\Documents\Search_Engine\output\chunks.jsonl", r"C:\Users\cmd\OneDrive\Documents\Search_Engine\output\chunks_embeddings.npy", r"C:\Users\cmd\OneDrive\Documents\Search_Engine\output\embeddings_meta.jsonl")
