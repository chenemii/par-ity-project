import requests
from newspaper import Article
import pandas as pd
import time
from pathlib import Path
import re
from typing import List

def extract_article_text(urls):
    """Extract text content from a list of article URLs"""
    articles = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            articles.append({
                'url': url,
                'title': article.title,
                'text': article.text,
                'authors': article.authors,
                'publish_date': article.publish_date,
                'source': url.split('/')[2]  # Extract domain
            })
            time.sleep(1)  # Be respectful to servers
        except Exception as e:
            print(f"Failed to extract {url}: {e}")
    
    return pd.DataFrame(articles)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    # Remove extra whitespace, special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def process_articles(urls: List[str], save_path: str = None) -> pd.DataFrame:
    """Complete pipeline to extract, clean, and process articles"""
    print(f"Extracting text from {len(urls)} articles...")
    
    # Extract articles
    df = extract_article_text(urls)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Create chunks for each article
    df['text_chunks'] = df['cleaned_text'].apply(
        lambda x: chunk_text(x) if pd.notna(x) else []
    )
    
    # Save if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_urls = [
        "https://example.com/article1",
        "https://example.com/article2"
    ]
    
    # Process articles
    # df = process_articles(sample_urls, "extracted_articles.csv")
    # print(f"Extracted {len(df)} articles")
    
    print("Article extractor ready! Use process_articles() with your URLs.") 