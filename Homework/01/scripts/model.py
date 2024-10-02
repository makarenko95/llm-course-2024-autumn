import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor


class Model(nn.Module):
    """
    Класс Model представляет собой нейронную сеть на основе LSTM для обработки последовательностей, таких как текст.
    Она состоит из слоев эмбеддингов, LSTM и линейного слоя для получения логитов, соответствующих размерам словаря.

    Аргументы:
        vocab_size (int): Размер словаря (количество уникальных слов).
        emb_size (int, необязательный): Размерность эмбеддингов. По умолчанию 128.
        num_layers (int, необязательный): Количество слоев в LSTM. По умолчанию 1.
        hidden_size (int, необязательный): Размерность скрытого состояния LSTM. По умолчанию 256.
        dropout (float, необязательный): Вероятность отключения нейронов (dropout) между слоями LSTM. По умолчанию 0.0.

    Методы:
        forward(x, hx=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
            Проводит прямое распространение через сеть.

    Пример использования:
        >>> import torch
        >>> from torch import nn
        >>> model = Model(vocab_size=10000, emb_size=128, num_layers=2, hidden_size=256, dropout=0.5)
        >>> input_data = torch.randint(0, 10000, (32, 50))  # Входные данные: батч из 32 последовательностей длиной 50
        >>> output, (h_n, c_n) = model(input_data)  # Прямое распространение
        >>> print(output.shape)  # Результат: torch.Size([32, 50, 10000])
    """
    def __init__(
            self,
            vocab_size: int,
            emb_size: int = 128,
            num_layers: int = 1,
            hidden_size: int = 256,
            dropout: float = 0.0
    ):
        super().__init__()
        self.embeddings = nn.Embedding(<YOUR CODE HERE>)
        self.lstm = nn.LSTM(<YOUR CODE HERE>)
        self.logits = nn.Linear(<YOUR CODE HERE>)

    def forward(
            self,
            x: Tensor,
            hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Проводит прямое распространение через сеть.

        Аргументы:
            x (Tensor): Входные данные (индексы слов) размером (batch_size, seq_len).
            hx (Optional[Tuple[Tensor, Tensor]]): Начальные скрытые состояния (h_n, c_n) для LSTM. По умолчанию None.

        Возвращает:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - Логиты (предсказания для каждого слова в последовательности) размером (batch_size, seq_len, vocab_size).
                - Пара скрытых состояний (h_n, c_n), где h_n и c_n — это последние скрытые и клеточные состояния LSTM.
        """
        <YOUR CODE HERE>
