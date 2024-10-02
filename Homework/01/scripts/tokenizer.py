from typing import List, Tuple, Dict
from tqdm import tqdm


class ByteTokenizer:
    """
    Класс для токенизации текста с использованием байтового представления символов.

    Этот класс реализует базовый байтовый токенизатор, который преобразует строки
    в последовательности байтов и обратно. Также поддерживаются специальные токены,
    такие как <pad>, <bos> (начало последовательности) и <eos> (конец последовательности).

    Атрибуты:
    ----------
    pad_token : bytes
        Токен для заполнения (padding) последовательности, по умолчанию b'<pad>'.
    bos_token : bytes
        Токен начала последовательности, по умолчанию b'<bos>'.
    eos_token : bytes
        Токен конца последовательности, по умолчанию b'<eos>'.
    pad_token_id : int или None
        Идентификатор токена заполнения, назначается при инициализации словаря.
    bos_token_id : int или None
        Идентификатор токена начала последовательности, назначается при инициализации словаря.
    eos_token_id : int или None
        Идентификатор токена конца последовательности, назначается при инициализации словаря.
    special_tokens : List[bytes]
        Список специальных токенов, включающий pad_token, bos_token и eos_token.
    vocab : dict
        Словарь для сопоставления индексов и байтовых значений символов.

    Методы:
    -------
    init_vocab() -> None
        Инициализирует словарь (vocab), где ключами являются индексы, а значениями — байтовые представления символов.
    train(texts: List[str], max_vocab: int) -> None
        Переинициализирует словарь (переопределяется в потомках).
    encode(text: str) -> List[int]
        Преобразует строку в список идентификаторов (байтов) с использованием кодировки UTF-8.
    decode(ids: List[int]) -> str
        Преобразует список идентификаторов (байтов) обратно в строку.
    get_vocab_size() -> int
        Возвращает размер словаря (количество уникальных символов и токенов).

    Пример использования:
    ---------------------
    >>> tokenizer = ByteTokenizer()
    >>> encoded = tokenizer.encode("Привет")
    >>> print(encoded)
    [208, 159, 209, 128, 208, 184, 208, 178, 208, 181, 209, 130]

    >>> decoded = tokenizer.decode(encoded)
    >>> print(decoded)
    'Привет'

    >>> vocab_size = tokenizer.get_vocab_size()
    >>> print(vocab_size)
    259
    """
    def __init__(self):
        self.pad_token = b'<pad>'
        self.bos_token = b'<bos>'
        self.eos_token = b'<eos>'
        self.pad_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token]
        self.vocab = {}
        self.init_vocab()

    def init_vocab(self) -> None:
        """Инициализирует словарь для токенизации, добавляя байтовые представления символов и специальные токены."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for token in self.special_tokens:
            idx = len(self.vocab)
            self.vocab[idx] = token
        token_to_id = {y: x for x, y in self.vocab.items()}
        self.pad_token_id = token_to_id[self.pad_token]
        self.bos_token_id = token_to_id[self.bos_token]
        self.eos_token_id = token_to_id[self.eos_token]

    def train(self, texts: List[str], max_vocab: int) -> None:
        """Тренирует токенизатор на данных текстах, переинициализируя словарь (пока без дополнительной логики)."""
        self.init_vocab()

    def encode(self, text: str) -> List[int]:
        """Преобразует строку в список байтов с использованием кодировки UTF-8.

        Параметры:
        ----------
        text : str
            Входная строка для преобразования.

        Возвращает:
        -----------
        List[int]
            Список идентификаторов (байтов), представляющих символы строки.
        """
        return list(text.encode('utf-8'))

    def decode(self, ids: List[int]) -> str:
        """Преобразует список идентификаторов обратно в строку.

        Параметры:
        ----------
        ids : List[int]
            Список байтовых идентификаторов.

        Возвращает:
        -----------
        str
            Декодированная строка.
        """
        text = b''.join(self.vocab[idx] for idx in ids).decode('utf-8', errors='replace')
        return text

    def get_vocab_size(self) -> int:
        """Возвращает количество уникальных символов и токенов в словаре.

        Возвращает:
        -----------
        int
            Размер словаря.
        """
        return len(self.vocab)


def count_pairs(data: List[List[int]]) -> Dict[Tuple[int, int], int]:
    """
    Считает, сколько раз встречается каждая пара последовательных элементов (стоящих на соседних позициях) во всех списках чисел.

    Параметры:
    ----------
    data: List[List[int]]
        Список, содержащий списки целых чисел.

    Возвращает:
    -----------
        Dict[Tuple[int, int], int]
            Словарь, где ключами являются пары элементов (кортежи), а значениями - количество их появлений в списках.

    Пример:
    -------
    >>> data = [[1, 2, 3], [2, 3, 4], [1, 2, 2]]
    >>> count_pairs(data)
    {(1, 2): 2, (2, 3): 2, (3, 4): 1, (2, 2): 1}
    """
    <YOUR CODE HERE>


def merge(numbers: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    Двигаясь слева направо, заменяет все вхождения заданной пары чисел в массиве на заданный индекс.
    Гарантируется, что заданный индекс не встречается в массиве чисел.

    Параметры:
    ----------
    numbers : List[int]
        Список целых чисел.
    pair : Tuple[int, int]
        Пара целых чисел, которую необходимо найти и заменить.
    idx : int
        Значение, на которое заменяется найденная пара.

    Возвращает:
    -----------
    List[int]
        Новый список, где каждая найденная пара заменена на значение idx.

    Примеры:
    -------
    >>> merge([1, 2, 3, 2, 3, 4], (2, 3), 9)
    [1, 9, 9, 4]

    >>> merge([1, 2, 3, 4, 5, 6], (4, 5), 0)
    [1, 2, 3, 0, 6]

    >>> merge([1, 2, 2, 3, 4], (2, 3), 99)
    [1, 2, 99, 4]

    >>> merge([0, 0, 0, 1], (0, 0), 2)
    [2, 0, 1]
    """
    <YOUR CODE HERE>


class BpeTokenizer(ByteTokenizer):
    """
    Класс для токенизации текста с использованием байтового представления и BPE (Byte Pair Encoding) алгоритма.

    Этот класс является наследником ByteTokenizer и расширяет его функционал, реализуя механизм
    склеивания наиболее часто встречающихся пар байт в новые токены, что позволяет эффективно
    сжимать словарь и улучшать обработку текстов.

    Атрибуты:
    ----------
    merges : dict
        Словарь, в котором хранятся пары токенов и их новые индексы после склеивания (BPE).

    Методы:
    -------
    init_vocab() -> None
        Переинициализирует словарь, добавляя таблицу склеиваний BPE.
    train(texts: List[str], max_vocab: int) -> None
        Тренирует BPE-токенизатор, находя наиболее частотные пары токенов и склеивая их,
        пока не будет достигнут заданный размер словаря.
    encode(text: str) -> List[int]
        Преобразует строку в список байтов с применением BPE для наиболее частотных пар токенов.

    Пример использования:
    ---------------------
    >>> tokenizer = BpeTokenizer()
    >>> tokenizer.train(["Мама мыла раму"], max_vocab=300)
    >>> encoded = tokenizer.encode("Мама мыла раму")
    >>> print(encoded)
    [208, 156, 261, 262, 260, 209, 139, 208, 187, 262, 209, 128, 261, 209, 131]

    >>> decoded = tokenizer.decode(encoded)
    >>> print(decoded)
    'Мама мыла раму'

    >>> vocab_size = tokenizer.get_vocab_size()
    >>> print(vocab_size)
    263
    """
    def __init__(self):
        """
        Инициализирует BpeTokenizer, добавляя словарь для хранения склеиваний пар токенов (merges).
        """
        self.merges = {}
        super().__init__()

    def init_vocab(self) -> None:
        """
        Инициализирует словарь для токенизации и обнуляет таблицу склеиваний пар токенов.

        Вызывает родительский метод для создания исходного словаря с байтами и специальными токенами.
        """
        super().init_vocab()
        self.merges = {}

    def train(self, texts: List[str], max_vocab: int) -> None:
        """
        Тренирует BPE-токенизатор на предоставленных текстах, последовательно склеивая
        наиболее частотные пары токенов до достижения заданного размера словаря.

        Параметры:
        ----------
        texts : List[str]
            Список текстов для тренировки токенизатора.
        max_vocab : int
            Максимальный размер словаря, после достижения которого процесс тренировки остановится.

        Возвращает:
        -----------
        None
        """
        self.init_vocab()

        if max_vocab <= len(self.vocab):
            return
        progress_bar = tqdm(range(max_vocab - len(self.vocab)))

        # Формируем исходный список номеров токенов для каждого текста (изначально это байты в кодировке utf-8)
        list_of_ids = [list(text.encode('utf-8')) for text in texts]

        for _ in progress_bar:
            # Находим наиболее частотную пару токенов для склеивания в один токен
            cnt = count_pairs(<YOUR CODE HERE>)
            pair = <YOUR CODE HERE>
            freq = cnt[pair]
            progress_bar.set_description(f'pair={pair}, freq={freq}')

            if freq == 1:
                break

            # Пополняем таблицу склеиваний и заводим новый токен в словаре
            new_idx = len(self.vocab)
            self.merges[pair] = new_idx
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Обновляем токенизацию для наших тренировочных текстов с учетом нового токена
            for i, ids in enumerate(list_of_ids):
                list_of_ids[i] = merge(<YOUR CODE HERE>)

    def encode(self, text: str) -> List[int]:
        """
        Преобразует строку в последовательность идентификаторов с применением BPE.

        Параметры:
        ----------
        text : str
            Входная строка для токенизации.

        Возвращает:
        -----------
        List[int]
            Список идентификаторов с учётом частотных пар токенов, объединённых алгоритмом BPE.
        """
        # Формируем исходный список номеров токенов для данного текста (изначально это байты в кодировке utf-8)
        ids = list(text.encode('utf-8'))

        # Последовательно применяем таблицу склеиваний в том порядке, в котором добавлялись токены в словарь
        while len(ids) > 1:
            cnt = count_pairs(<YOUR CODE HERE>)
            pair = <YOUR CODE HERE>
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(<YOUR CODE HERE>)
        return ids
