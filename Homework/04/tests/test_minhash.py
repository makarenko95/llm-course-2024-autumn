import sys
sys.path.append("Homework/04")
import unittest
import random
from minhash import MinHash

Docs = [
   'Я очень люблю читать книги, особенно перед сном. Это помогает мне расслабиться и отвлечься от повседневных забот.',

'Прогулки на свежем воздухе — это отличный способ улучшить настроение и укрепить здоровье. Они помогают мне чувствовать себя бодрым и энергичным.',

'Мне нравится слушать музыку, она вдохновляет меня и помогает сосредоточиться. Особенно я люблю классическую музыку.',

'Спорт играет важную роль в моей жизни. Он помогает мне поддерживать хорошую физическую форму и быть здоровым.',

'Общение с друзьями — это то, что делает мою жизнь яркой и насыщенной. Мы проводим время вместе, смеёмся и наслаждаемся общением.',
]



def sort_tuples_in_list(answer):
    sorted_answer = []

    for pair in answer:
        if pair[0] < pair[1]:
            sorted_answer.append(pair)
        else:
            sorted_answer.append((pair[1], pair[0]))
            
    return set(sorted_answer)


class TestMinhash(unittest.TestCase):
    def test_jaccard(self):
        min_hash = MinHash(num_permutations=5, threshold=0.0)
        jaccard_sim = min_hash.get_jaccard_similarity({1, 2, 3, 4}, {3,4})
        self.assertEqual(jaccard_sim, 0.5)
        
        jaccard_sim = min_hash.get_jaccard_similarity({1, 2, 3, 4}, {3,4,7,8,9,10})
        self.assertEqual(jaccard_sim, 0.25)

    def test_get_similar_pairs(self):
        
        min_hash = MinHash(num_permutations=2, threshold=0.0)
        answer = min_hash.run_minhash(Docs) 
        self.assertEqual(sort_tuples_in_list(answer), {(0, 4)})
        
        min_hash = MinHash(num_permutations=5, threshold=0.0)
        answer = min_hash.run_minhash(Docs)
        self.assertEqual(sort_tuples_in_list(answer), {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)})
        
        
        min_hash = MinHash(num_permutations=2, threshold=0.3)
        answer = min_hash.run_minhash(Docs)
        self.assertEqual(sort_tuples_in_list(answer), {(0, 4)})
        
        
        
        
