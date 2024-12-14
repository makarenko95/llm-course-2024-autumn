import pytest
import os
import numpy as np
from part1.search_engine import Document, Indexer, Searcher, load_documents

@pytest.fixture
def documents():
    return load_documents('Homework/09/data/articles.json')

def test_load_documents(documents):
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)

def test_indexer(documents, tmp_path):
    # Тестируем индексацию
    indexer = Indexer()
    indexer.add_documents(documents)
    
    # Проверяем что эмбеддинги создались
    assert len(indexer.documents) == len(documents)
    assert indexer.embeddings is not None
    assert indexer.embeddings.shape[0] == len(documents)
    assert indexer.embeddings.shape[1] == 384  # размерность для MiniLM-L6
    
    # Тестируем сохранение/загрузку
    index_path = tmp_path / "index.pkl"
    indexer.save(str(index_path))
    
    assert index_path.exists()
    
    new_indexer = Indexer()
    new_indexer.load(str(index_path))
    
    assert len(new_indexer.documents) == len(documents)
    assert new_indexer.embeddings.shape == indexer.embeddings.shape

def test_searcher(documents, tmp_path):
    # Создаем индекс
    index_path = tmp_path / "test_index.pkl"
    indexer = Indexer()
    indexer.add_documents(documents)
    indexer.save(str(index_path))
    
    # Тестируем поиск
    searcher = Searcher(str(index_path))
    results = searcher.search("machine learning", top_k=3)
    
    # Проверяем результаты
    assert len(results) <= 3
    assert all(isinstance(r.score, float) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    
    # Проверяем что документы про ML нашлись
    titles = [r.title.lower() for r in results]
    assert any('machine learning' in title for title in titles)

    # Тестируем другой запрос
    results = searcher.search("web development", top_k=3)
    assert len(results) <= 3
    titles = [r.title.lower() for r in results]
    assert any('web' in title for title in titles)