import sys
sys.path.append("Homework/04")
import unittest
import random
from minhashlsh import MinHashLSH

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


class TestMinhashLSH(unittest.TestCase):
    def test_get_similar_pairs(self):   
        all_pairs = {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
        
        min_hash = MinHashLSH(num_permutations=5, num_buckets=1, threshold=0.0)
        answer = min_hash.run_minhash_lsh(Docs) 
        self.assertEqual(sort_tuples_in_list(answer), set())
        
        min_hash = MinHashLSH(num_permutations=5, num_buckets=10, threshold=0.0)
        answer = min_hash.run_minhash_lsh(Docs) 
        self.assertEqual(sort_tuples_in_list(answer), all_pairs)
        
        min_hash = MinHashLSH(num_permutations=5, num_buckets=10000, threshold=0.0)
        answer = min_hash.run_minhash_lsh(Docs) 
        self.assertEqual(sort_tuples_in_list(answer), all_pairs)
                
        min_hash = MinHashLSH(num_permutations=5, num_buckets=3, threshold=0.0)
        answer = min_hash.run_minhash_lsh(Docs) 
        self.assertEqual(sort_tuples_in_list(answer), {(0, 2), (0, 3), (2, 3)})
        
        min_hash = MinHashLSH(num_permutations=5, num_buckets=3, threshold=0.0)
        answer = min_hash.run_minhash_lsh([Docs[1], Docs[1]]) 
        self.assertEqual(sort_tuples_in_list(answer), {(0, 1)})
        
