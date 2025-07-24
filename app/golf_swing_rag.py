import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
import openai
from dotenv import load_dotenv
import os
import json
import pickle
from typing import List, Dict, Tuple
import re
from datetime import datetime

# Load environment variables
load_dotenv()

class GolfSwingRAG:
    def __init__(self, csv_file_path: str = None):
        """Initialize the Golf Swing RAG system"""
        # Set default CSV path based on current working directory
        if csv_file_path is None:
            if os.path.exists("golf_swing_articles_complete.csv"):
                csv_file_path = "golf_swing_articles_complete.csv"
            elif os.path.exists("../golf_swing_articles_complete.csv"):
                csv_file_path = "../golf_swing_articles_complete.csv"
            else:
                raise FileNotFoundError("golf_swing_articles_complete.csv not found in current or parent directory")
        
        self.csv_file_path = csv_file_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.metadata = []
        self.openai_client = None
        
        # Initialize OpenAI client using Streamlit secrets
        try:
            openai_key = st.secrets.get("openai", {}).get("api_key", "")
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
        except (KeyError, FileNotFoundError, AttributeError):
            # Fallback to environment variable if secrets not available
            if os.getenv("OPENAI_API_KEY"):
                self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def load_and_process_data(self):
        """Load CSV data and process it for RAG"""
        print("Loading golf swing data...")
        
        # Read CSV file
        df = pd.read_csv(self.csv_file_path)
        print(f"Loaded {len(df)} articles")
        
        # Process each article
        all_chunks = []
        all_metadata = []
        
        for idx, row in df.iterrows():
            # Parse text chunks if they exist
            text_chunks = []
            if pd.notna(row['text_chunks']) and row['text_chunks'].strip():
                try:
                    # Parse the text_chunks column (it appears to be a list in string format)
                    chunks_str = row['text_chunks']
                    if chunks_str.startswith('[') and chunks_str.endswith(']'):
                        # Remove brackets and split by quotes
                        chunks_str = chunks_str[1:-1]  # Remove outer brackets
                        # Split by quote patterns while preserving content
                        text_chunks = [chunk.strip().strip("'\"") for chunk in chunks_str.split("', '") if chunk.strip()]
                        if not text_chunks and chunks_str:
                            text_chunks = [chunks_str.strip().strip("'\"")]
                except:
                    # Fallback: use cleaned_text if text_chunks parsing fails
                    text_chunks = [row['cleaned_text']] if pd.notna(row['cleaned_text']) else []
            
            # If no chunks, create chunks from cleaned_text or text
            if not text_chunks:
                text_content = row['cleaned_text'] if pd.notna(row['cleaned_text']) else row['text']
                if pd.notna(text_content):
                    # Split into chunks of ~500 words
                    words = text_content.split()
                    chunk_size = 500
                    text_chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            
            # Add each chunk with metadata
            for chunk_idx, chunk in enumerate(text_chunks):
                if chunk and len(chunk.strip()) > 50:  # Only process substantial chunks
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'title': row['title'],
                        'url': row['url'],
                        'source': row['source'],
                        'publish_date': row['publish_date'],
                        'authors': row['authors'],
                        'chunk_index': chunk_idx,
                        'article_index': idx
                    })
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        print(f"Created {len(all_chunks)} text chunks")
        
    def create_embeddings(self, force_recreate: bool = False):
        """Create embeddings for all text chunks"""
        # Determine the correct base directory for embeddings files
        if os.path.exists("golf_swing_articles_complete.csv"):
            # Running from project root
            embeddings_file = "golf_swing_embeddings.pkl"
            index_file = "golf_swing_index.faiss"
        else:
            # Running from app directory
            embeddings_file = "../golf_swing_embeddings.pkl"
            index_file = "../golf_swing_index.faiss"
        
        if not force_recreate and os.path.exists(embeddings_file) and os.path.exists(index_file):
            print("Loading existing embeddings...")
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            self.index = faiss.read_index(index_file)
            print(f"Loaded {len(self.chunks)} chunks with embeddings")
            return
        
        print("Creating embeddings...")
        if not self.chunks:
            self.load_and_process_data()
        
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_chunks, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            print(f"Processed {min(i+batch_size, len(self.chunks))}/{len(self.chunks)} chunks")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save embeddings and index
        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        faiss.write_index(self.index, index_file)
        
        print(f"Created and saved embeddings for {len(self.chunks)} chunks")
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks using semantic similarity"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(score)
                })
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using OpenAI API with context"""
        if not self.openai_client:
            return self._generate_fallback_response(query, context_chunks)
        
        # Prepare context
        context = "\n\n".join([f"Source: {chunk['metadata']['title']}\nContent: {chunk['chunk']}" 
                              for chunk in context_chunks])
        
        # Create system prompt
        system_prompt = """You are a golf swing technique expert assistant. You help golfers improve their swing by providing detailed, accurate advice based on professional golf instruction content.

Instructions:
- Answer questions about golf swing technique, mechanics, common problems, and solutions
- Provide specific, actionable advice when possible
- Reference relevant technical concepts when appropriate
- Be encouraging and supportive
- If asked about physical limitations or injuries, recommend consulting with a TPI certified professional
- Always base your answers on the provided context from golf instruction materials

Context from golf instruction database:
{context}"""

        user_prompt = f"""Based on the golf instruction content provided, please answer this question about golf swing technique:

Question: {query}

Please provide a helpful, detailed response that addresses the specific question while drawing from the relevant information in the context."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback_response(query, context_chunks)
    
    def _generate_fallback_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate a fallback response when OpenAI API is not available"""
        if not context_chunks:
            return "I couldn't find specific information about that topic in the golf swing database. Could you try rephrasing your question or being more specific?"
        
        # Create a simple response based on the most relevant chunk
        best_chunk = context_chunks[0]
        chunk_content = best_chunk['chunk']
        title = best_chunk['metadata']['title']
        
        response = f"Based on the article '{title}', here's what I found:\n\n"
        response += chunk_content[:500] + "..."
        response += f"\n\nFor more detailed information, you can refer to the full article: {title}"
        
        return response
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Main query method that returns both response and sources"""
        # Search for relevant chunks
        relevant_chunks = self.search_similar_chunks(question, top_k)
        
        # Generate response
        response = self.generate_response(question, relevant_chunks)
        
        return {
            'response': response,
            'sources': relevant_chunks,
            'query': question,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Initialize and test the RAG system"""
    rag = GolfSwingRAG()
    rag.load_and_process_data()
    rag.create_embeddings()
    
    # Test query
    test_query = "What wrist motion happens during the downswing?"
    result = rag.query(test_query)
    
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Number of sources: {len(result['sources'])}")

if __name__ == "__main__":
    main() 