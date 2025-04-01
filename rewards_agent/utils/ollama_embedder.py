import requests
from langchain_core.embeddings import Embeddings
from typing import List

class OllamaNomicEmbedder(Embeddings):
    def __init__(self, model_name: str = "nomic-embed", base_url: str = "http://ollama:11434"):
        self.model_name = model_name 
        self.base_url = base_url

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query using Ollama API."""
        response = self._request_ollama(query)

        return response.get("embedding")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Ollama API."""
        response = self._request_ollama(texts)

        return [item.get("embedding") for item in response]

    def _request_ollama(self, data):
        """Sends request to Ollama API to get embeddings."""
        url = f"{self.base_url}/v1/embeddings/{self.model_name}"
        headers = {"Content-Type": "application/json"}
        
        body = {"input": data} if isinstance(data, str) else {"inputs": data}
        
        response = requests.post(url, json=body, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Ollama API request failed: {response.status_code} - {response.text}")
