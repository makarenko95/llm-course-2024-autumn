import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor


class Collator:
    """
    Класс Collator используется для дополнения (padding) списков разной длины
    до одинаковой длины с использованием заданного значения padding_value.

    Аргументы:
        padding_value (int): Значение, которое будет использоваться для дополнения
        (padding) последовательностей до одинаковой длины.
    """
    def __init__(self, padding_value: int):
        """
        Инициализирует Collator с заданным значением для дополнения.

        Аргументы:
            padding_value (int): Значение для padding.
        """
        self.padding_value = padding_value

    def __call__(self, data: List[List[int]]) -> Tensor:
        """
        Применяет padding к входным данным, чтобы выровнять списки до одинаковой длины.

        Аргументы:
            data (List[List[int]]): Список списков целых чисел,
            которые необходимо дополнить до одинаковой длины.

        Возвращает:
            Tensor: Тензор с дополненными значениями, где все последовательности
            имеют одинаковую длину.

        Пример:
            >>> collator = Collator(padding_value=0)
            >>> data = [[1, 2, 3], [4, 5], [6]]
            >>> output = collator(data)
            >>> print(output)
            tensor([[1, 2, 3],
                    [4, 5, 0],
                    [6, 0, 0]])
        """
        data = [torch.tensor(x, dtype=torch.long) for x in data]
        data = pad_sequence(<YOUR CODE HERE>)
        return data
