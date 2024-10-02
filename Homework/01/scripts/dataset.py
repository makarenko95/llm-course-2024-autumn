from typing import List, Optional
from tqdm import tqdm
from torch.utils.data import Dataset
from scripts.tokenizer import ByteTokenizer


class MyDataset(Dataset):
    """
    Класс MyDataset представляет собой набор данных для работы с текстами, закодированными с помощью токенизатора.

    Атрибуты:
    ----------
    max_length : Optional[int]
        Максимальная длина последовательности токенов (по умолчанию None, что означает отсутствие ограничения).
    data : List[List[int]]
        Список последовательностей токенов для каждого текста.

    Параметры:
    ----------
    texts : List[str]
        Список текстов для токенизации.
    tokenizer : ByteTokenizer
        Токенизатор, который преобразует текст в последовательность токенов.
    max_length : Optional[int], по умолчанию None
        Максимальная длина последовательности токенов (опционально). Если задано, обрезает последовательность до этой длины.

    Методы:
    -------
    __getitem__(idx: int) -> List[int]
        Возвращает последовательность токенов для текста по индексу, обрезанную до max_length.
    __len__() -> int
        Возвращает количество текстов в наборе данных

    Пример:
    ----------
    >>> texts = ["Привет, мир!", "Это тест."]
    >>> tokenizer = ByteTokenizer()  # Предположим, что токенизатор уже реализован
    >>> dataset = MyDataset(texts, tokenizer, max_length=10)
    >>> len(dataset)
    2
    >>> dataset[0]
    [tokenizer.bos_token_id, 1, 2, 3, tokenizer.eos_token_id]  # Пример токенов
    """
    def __init__(self, texts: List[str], tokenizer: ByteTokenizer, max_length: Optional[int] = None):
        self.max_length = max_length
        self.data = []
        for text in tqdm(texts):
            # Получаем список токенов (номеров) для данного текста и добавляем к началу и концу спецтокены bos, eos (см. пример)
            token_ids = <YOUR CODE HERE>
            self.data.append(token_ids)

    def __getitem__(self, idx: int) -> List[int]:
        """
        Возвращает последовательность токенов для текста по индексу, обрезанную до max_length.

        Параметры:
        ----------
        idx : int
            Индекс элемента в наборе данных, который нужно вернуть

        Возвращает:
        -----------
        List[int]
            Усеченный список номеров токенов
        """
        return <YOUR CODE HERE>

    def __len__(self) -> int:
        """Возвращает количество текстов в наборе данных."""
        return <YOUR CODE HERE>
