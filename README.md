# 📰 NLP RAG on 30,000 World News Articles


 [Link to Kaggle notebook](https://www.kaggle.com/code/musarajput/nlp-rag-pipeline/) You can play around and tweak if you want.

**🔗 Live Demo:** [Ask the News - Hugging Face Space](https://huggingface.co/spaces/MusaR/NLP-RAG-world-news)  
*⚠️ Note: The demo uses a free CPU tier and TinyLlama so answers quality is tanked, and response may take 10 - 15 min.*

---

## 📌 Project Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline from scratch to answer queries based on this [All the News Dataset](https://www.kaggle.com/datasets/davidmckinley/all-the-news-dataset) consisting of over 2.2M+ articles though this project is done on **30,000 world news articles**.

Unlike typical RAG systems that rely on LangChain or LlamaIndex, this project uses only low-level libraries like **PyTorch**, **Transformers**, and **FAISS**, showcasing the core mechanics and allowing maximum control over the architecture.

You can explore the full development workflow in the provided notebook: `nlp-rag-pipeline.ipynb`.

The notebook contains all cell codes, from processing the CSV, each article is broken down into smaller, overlapping text chunks, then embedder model is loaded, where each chunk is converted into 'embedding' and stored in FAISS index. Then comes the Cross-Encoder model for Re-ranking, to pass the best context to the LLM. And lastly, the top 3 - 5 ranked chunks are taken and fed into the LLM for output generation. 

---


## ✅ Key Features

- 🔧 **Custom From-Scratch RAG:** 
- 🧠 **Hybrid Retrieval:**
  - **BM25** (via `rank_bm25`) for fast keyword-based matches.
  - **FAISS** + Sentence Embeddings for deep semantic similarity.
- 🎯 **Cross-Encoder Reranking:** Uses a cross-encoder to sort retrieved documents by contextual relevance.
- ⚡ **Optimized Deployment:** Runs on Hugging Face Spaces using `TinyLlama` and CPU-friendly optimizations.

---

## 🔁 RAG Pipeline Flow

```text
User Query → BM25 + FAISS Retrieval → Candidate Pooling → Cross-Encoder Reranking → Top-N Chunk Selection → LLM Answer Generation → Output with Source URLs
````

1. **User inputs question** via Gradio UI.
2. **BM25** retrieves keyword-based matches.
3. **FAISS** retrieves semantically similar chunks.
4. **Candidates merged + deduplicated.**
5. **Cross-Encoder** ranks each chunk by relevance to query.
6. **Top 3 chunks** are combined into the LLM prompt context.
7. **LLM (TinyLlama)** generates answer strictly from provided context. (quality is not good, running on better hardware is advised)
8. **Answer + Source URLs** displayed.

---

## 🧱 Technology Stack

| Component      | Model / Library                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------- |
| 💬 LLM         | `Microsoft Phi-3`                                                                      |
| 🔍 Embeddings  | `multi-qa-MiniLM-L6-cos-v1`                                                                               |
| 📊 Reranker    | `cross-encoder/ms-marco-MiniLM-L-6-v2`                                                                    |
| 🛠️ Core Tools | `PyTorch`, `Transformers`, `Gradio`, `faiss-cpu`, `rank-bm25`, `sentence-transformers`, `numpy`, `pandas` |
| ☁️ Platform    | Hugging Face Spaces                                                                                       |

---

## 🚀 Steps to run locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rajput-musa/NLP-RAG-World-News
   cd NLP-RAG-World-News
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   
3. **Let the cells run, it may take some time depending on your hardware and internet, models will be downloaded.**

 
4. **Download RAG artifacts** and place in root directory:

   * `chunks_df.parquet`
   * `bm25_index.pkl`
   * `news_chunks.faiss_index`

5. **Run the app:**

   ```bash
   python app.py
   ```


## 📄 License

MIT License

