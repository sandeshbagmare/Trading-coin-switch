"""!
@file knowledge_base.py
@brief Contains classes and methods for creating and managing a persistant vector database of trading strategies.
@details This module provides an end-to-end pipeline to extract text from trading literature (PDFs), 
         produce semantic chunks, embed them using local sentence-transformers, and store them 
         persistently in ChromaDB. 
         Additionally, it includes a QueryEngine wrapper that uses a custom OpenAI-compatible API 
         (e.g. BluesMinds) with DeepSeek or GPT models to perform Retrieval-Augmented Generation (RAG).
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Document:
    """!
    @class Document
    @brief A simple data class representing an extracted document or section.
    """
    def __init__(self, content: str, metadata: Dict[str, Any]):
        """!
        @brief Constructor for Document.
        @param content The text content of the document.
        @param metadata A dictionary containing source info, page number, etc.
        """
        self.content = content
        self.metadata = metadata


class Chunk:
    """!
    @class Chunk
    @brief A data class representing a chunk of a document.
    """
    def __init__(self, content: str, metadata: Dict[str, Any]):
        """!
        @brief Constructor for Chunk.
        @param content The chunked text.
        @param metadata Metadata carrying over from the original document and holding chunk-specific identifiers.
        """
        self.content = content
        self.metadata = metadata


class DocumentIngester:
    """!
    @class DocumentIngester
    @brief Manages reading and text extraction from various file types, primarily PDFs.
    """
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        """!
        @brief Constructor for DocumentIngester.
        @param knowledge_dir Path to the directory containing strategy PDFs and literature.
        """
        self.knowledge_dir = Path(knowledge_dir)

    def ingest_all(self) -> List[Document]:
        """!
        @brief Scans the knowledge directory for supported files and extracts their text.
        @return A list of Document objects extracted from the files.
        """
        documents = []
        pdf_files = list(self.knowledge_dir.rglob("*.pdf"))
        
        for file in pdf_files:
            print(f"📖 Loading: {file.name}")
            docs = self._load_pdf(file)
            documents.extend(docs)

        print(f"✅ Loaded {len(documents)} document pages/sections")
        return documents

    def _load_pdf(self, path: Path) -> List[Document]:
        """!
        @brief Extracts text page by page from a PDF using pdfplumber.
        @param path The pathlib.Path to the PDF file.
        @return A list of Document objects, one for each page with text.
        """
        documents = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:  # Skip virtually empty pages
                        documents.append(Document(
                            content=text,
                            metadata={
                                'source': path.name,
                                'page': i + 1,
                                'type': 'trading_research'
                            }
                        ))
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")
        return documents


class SemanticChunker:
    """!
    @class SemanticChunker
    @brief Implements chunking strategies using LangChain's RecursiveCharacterTextSplitter.
    @details Ensures that technical trading strategies and rules are not abruptly broken by using 
             semantically meaningful separators (paragraphs, newlines, spaces).
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """!
        @brief Constructor for SemanticChunker.
        @param chunk_size Target maximum characters per chunk.
        @param chunk_overlap Characters to overlap to preserve context between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """!
        @brief Splits a list of documents into highly cohesive overlapping chunks.
        @param documents A list of Document objects.
        @return A list of Chunk objects securely retaining their source metadata.
        """
        chunks = []
        for doc in documents:
            texts = self.splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                meta = doc.metadata.copy()
                meta['chunk_id'] = i
                chunks.append(Chunk(content=text, metadata=meta))
        return chunks


class ChunkEmbedder:
    """!
    @class ChunkEmbedder
    @brief Manages vector embeddings for text chunks using SentenceTransformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """!
        @brief Constructor initializing the local embedding model.
        @param model_name The HuggingFace sentence-transformers model to use.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[Chunk]) -> List[tuple]:
        """!
        @brief Encodes text chunks into high-density vectors.
        @param chunks List of Chunk objects.
        @return A list of tuples containing (Chunk, List[float] representing embedding).
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return list(zip(chunks, embeddings.tolist()))


class VectorStore:
    """!
    @class VectorStore
    @brief Interface to ChromaDB for persistent storage and semantic retrieval of chunks.
    """
    def __init__(self, persist_dir: str = "data/knowledge/chromadb"):
        """!
        @brief Constructor for VectorStore.
        @param persist_dir Directory where the ChromaDB SQLite files are stored persistently.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="trading_strategies",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks_with_embeddings: List[tuple]):
        """!
        @brief Inserts chunks and their respective embeddings into ChromaDB.
        @param chunks_with_embeddings List of tuples: (Chunk, embedding_vector).
        """
        if not chunks_with_embeddings:
            return

        ids = [f"{c.metadata['source']}_p{c.metadata['page']}_c{c.metadata['chunk_id']}" for c, _ in chunks_with_embeddings]
        documents = [c.content for c, _ in chunks_with_embeddings]
        embeddings = [e for _, e in chunks_with_embeddings]
        metadatas = [c.metadata for c, _ in chunks_with_embeddings]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"✅ Added {len(documents)} chunks to Vector Database.")

    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """!
        @brief Performs an approximate nearest neighbor search using Cosine Similarity.
        @param query_embedding The vectorized query.
        @param n_results K-amount of top results to retrieve.
        @return ChromaDB result dictionary containing distances, chunks, and metadata.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results


class RAGQueryEngine:
    """!
    @class RAGQueryEngine
    @brief Executes Retrieval-Augmented Generation workflows utilizing the stored VectorDB and external LLM.
    """
    def __init__(self, vector_store: VectorStore, embedder: ChunkEmbedder):
        """!
        @brief Constructor fetching base API settings for the BluesMinds gateway.
        @param vector_store Initialized VectorStore instance.
        @param embedder Initialized ChunkEmbedder instance.
        """
        self.vs = vector_store
        self.embedder = embedder
        
        # Configure the custom OpenAI client
        base_url = os.getenv("LLM_BASE_URL", "https://api.bluesminds.com/v1")
        api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "gpt-4o")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def query(self, question: str) -> str:
        """!
        @brief Synthesizes an answer by retrieving context and calling the LLM.
        @param question The user or agent's query about trading decisions.
        @return The generated string answer integrating knowledge base insights.
        """
        # Embed Query
        query_embedding = self.embedder.model.encode([question], normalize_embeddings=True)[0].tolist()
        
        # Search Top K from ChromaDB
        search_results = self.vs.search(query_embedding, n_results=5)
        
        if not search_results['documents'][0]:
            return "No relevant trading strategies found in the knowledge base."
            
        context_chunks = search_results['documents'][0]
        
        # Construct Prompt
        context_str = "\n\n".join(f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks))
        
        prompt = f"""You are a crypto trading expert. Answer the following question based on the provided technical analysis and strategy context.
        
Context documents:
{context_str}

Question: {question}

Please provide a concise, actionable trading strategy or insight.
"""
        
        # Call Custom OpenAI API
        print(f"🤖 Calling LLM ({self.model}) at {self.client.base_url}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency trading agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ LLM Error: {e}"


def build_knowledge_base():
    """!
    @brief Helper pipeline function to execute the full document ingestion process.
    """
    print("🚀 Starting Knowledge Base Build Pipeline...")
    
    ingester = DocumentIngester()
    docs = ingester.ingest_all()
    
    if not docs:
        print("⚠️ No documents found. Run `python download_books.py` first.")
        return
        
    chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(docs)
    print(f"🧩 Split into {len(chunks)} contextual chunks.")
    
    embedder = ChunkEmbedder()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    
    vector_store = VectorStore()
    vector_store.add_chunks(chunks_with_embeddings)
    
    print("✅ Knowledge Base Build Complete.")


if __name__ == "__main__":
    # If run directly, rebuild the database 
    build_knowledge_base()
    
    # Test query
    print("\n🧐 Testing retrieval...")
    vs = VectorStore()
    emb = ChunkEmbedder()
    rag = RAGQueryEngine(vs, emb)
    
    answer = rag.query("What are the best indicators for mean reversion trading in crypto?")
    print("\n--- Synthesis Result ---")
    print(answer)
    print("------------------------")
