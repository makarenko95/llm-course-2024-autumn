import re
import pandas as pd
import numpy as np


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+"," ",text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)      
        return set(text.split(' '))
    
    def get_occurrence_matrix(self, corpus_of_texts: list[set]) -> pd.DataFrame:
        '''
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        '''
        # TODO:
        df.sort_index(inplace=True)
        return df
    
    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a
    
    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        '''
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        '''
        values_dict = {
            'a': [3, 4, 5, 7, 8],
            'b': [3, 4, 5, 7, 8] 
        }
        a = values_dict['a'][permutation_index]
        b = values_dict['b'][permutation_index]
        return (a*(x+1) + b) % prime_num_rows 
    
    
    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        '''
        Вовзращает расстояние Жаккарда для двух сетов. 
        '''
        # TODO:
        return 
    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        return 
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO: 
                
        return 
     
    
    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей.

        '''
        # TODO:
        return 

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs
    
    
    
