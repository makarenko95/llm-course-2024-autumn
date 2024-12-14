from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult

class FAISSSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать создание FAISS индекса
        
        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        pass

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса
        
        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        pass

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса
        
        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        pass

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск
        
        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        pass

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        TODO: Реализовать batch-поиск
        
        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        pass
