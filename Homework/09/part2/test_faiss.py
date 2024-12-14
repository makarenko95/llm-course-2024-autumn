import pytest
import numpy as np
import faiss
from part2.faiss_search import FAISSSearcher 
from part1.search_engine import Document, load_documents

@pytest.fixture
def documents():
    return load_documents('Homework/09/data/articles.json')

@pytest.fixture
def searcher(documents):
    searcher = FAISSSearcher()
    searcher.build_index(documents)
    return searcher

def test_build_index(documents):
    searcher = FAISSSearcher()
    searcher.build_index(documents)
    
    assert len(searcher.documents) == len(documents)
    assert searcher.index is not None
    assert isinstance(searcher.index, (faiss.IndexFlatIP, faiss.IndexIVFFlat))

def test_save_load(searcher, tmp_path):
    index_path = tmp_path / "faiss_index.pkl"
    searcher.save(str(index_path))
    
    new_searcher = FAISSSearcher()
    new_searcher.load(str(index_path))
    
    assert len(new_searcher.documents) == len(searcher.documents)
    assert isinstance(new_searcher.index, (faiss.IndexFlatIP, faiss.IndexIVFFlat))

def test_basic_search(searcher):
    results = searcher.search("machine learning", top_k=3)
    
    assert len(results) == 3
    assert all(0 <= r.score <= 1 for r in results)
    assert any('machine learning' in r.title.lower() for r in results)

def test_batch_search(searcher):
    queries = ["machine learning", "web development"]
    results = searcher.batch_search(queries, top_k=3)
    
    assert len(results) == len(queries)
    assert all(len(r) == 3 for r in results)
    assert any('machine learning' in r.title.lower() for r in results[0])
    assert any('web' in r.title.lower() for r in results[1])

def test_search_performance(searcher):
    import time
    
    start = time.time()
    for _ in range(10):
        searcher.search("test query")
    end = time.time()
    
    avg_time = (end - start) / 10
    assert avg_time < 0.1

def test_batch_search_performance(searcher):
    import time
    
    queries = ["query1", "query2", "query3"] * 3
    
    start = time.time()
    results = searcher.batch_search(queries)
    end = time.time()
    
    time_per_query = (end - start) / len(queries)
    assert time_per_query < 0.1