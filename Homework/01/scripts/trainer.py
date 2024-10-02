import torch
import torch.nn as nn
from typing import List, Optional, Callable, Union
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scripts.model import Model


class Trainer:
    """
    Класс для обучения и оценки модели.

    Параметры:
        model (Model): Модель, которую необходимо обучить.
        train_dataset (Union[Dataset, List[Tensor]]): Датасет для обучения.
        eval_dataset (Union[Dataset, List[Tensor]]): Датасет для оценки.
        n_epochs (int, по умолчанию 3): Количество эпох для обучения.
        lr (float, по умолчанию 1e-5): Скорость обучения.
        train_batch_size (int, по умолчанию 1): Размер батча для обучения.
        eval_batch_size (int, по умолчанию 1): Размер батча для оценки.
        eval_steps (Optional[int], по умолчанию None): Шаги между оценками.
        collator (Optional[Callable[[List[List[int]]], Tensor]], по умолчанию None): Функция для подготовки батча.
        ignore_index (int, по умолчанию -100): Индекс для игнорирования в функции потерь.

    Атрибуты:
        model (Model): Модель, которая обучается.
        loss_func (nn.CrossEntropyLoss): Функция потерь.
        optimizer (torch.optim.Adam): Оптимизатор для обучения.
        train_loader (DataLoader): Загрузчик данных для обучения.
        eval_loader (DataLoader): Загрузчик данных для оценки.
        n_epochs (int): Количество эпох.
        eval_steps (Optional[int]): Шаги между оценками.

    Методы:
        calc_loss(logits: Tensor, y: Tensor) -> Tensor:
            Вычисляет потери по логитам и целевым меткам.

        train() -> None:
            Запускает процесс обучения модели.

        evaluate() -> float:
            Оценивает модель на наборе данных для оценки и возвращает среднее значение потерь.

    Пример использования:
    --------------
    >>> model = Model()  # Ваша модель
    >>> train_data = ...  # Ваш тренировочный датасет
    >>> eval_data = ...  # Ваш датасет для оценки
    >>> trainer = Trainer(model, train_data, eval_data, n_epochs=5, lr=3e-5, train_batch_size=32, eval_batch_size=64)
    >>> trainer.train()  # Запуск процесса обучения
    >>> eval_loss = trainer.evaluate()  # Оценка модели
    >>> print(f"Потери на оценке: {eval_loss}")
    """
    def __init__(
            self,
            model: Model,
            train_dataset: Union[Dataset, List[Tensor]],
            eval_dataset: Union[Dataset, List[Tensor]],
            n_epochs: int = 3,
            lr: float = 1e-5,
            train_batch_size: int = 1,
            eval_batch_size: int = 1,
            eval_steps: Optional[int] = None,
            collator: Optional[Callable[[List[List[int]]], Tensor]] = None,
            ignore_index: int = -100
    ):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collator
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collator
        )
        self.n_epochs = n_epochs
        self.eval_steps = eval_steps

    def calc_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Вычисляет потери (loss) на основе предсказанных логитов и целевых меток.

        Параметры:
            logits (Tensor): Предсказанные моделью значения (логиты).
            y (Tensor): Истинные целевые метки.

        Возвращает:
            Tensor: Значение потерь.
        """
        return <YOUR CODE HERE>

    def train(self) -> None:
        """
        Запускает процесс обучения модели. После каждой эпохи выводит значение потерь.
        Если задан eval_steps, проводит оценку через каждые eval_steps итераций.
        """
        progress_bar = tqdm(total=self.n_epochs * len(self.train_loader))
        iterations = 0
        for _ in range(self.n_epochs):
            for ids in self.train_loader:
                iterations += 1
                self.model.train()
                # Готовим входы (текущие токены) и выходы (следующие токены)
                x = <YOUR CODE HERE>
                y = <YOUR CODE HERE>
                # Получаем логиты и считаем лосс
                logits, _ = self.model(x)
                loss = self.calc_loss(logits, y)
                progress_bar.update()
                progress_bar.set_description(f'epoch={iterations / len(self.train_loader)}, loss={loss.item()}')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.eval_steps is not None and iterations % self.eval_steps == 0:
                    print(f'epoch={iterations / len(self.train_loader)}, eval_loss={self.evaluate()}')

    def evaluate(self) -> float:
        """
        Оценивает модель на наборе данных для оценки, вычисляя среднее значение потерь.

        Возвращает:
            float: Среднее значение потерь на наборе данных для оценки.
        """
        self.model.eval()
        total_loss = 0.0
        for ids in self.eval_loader:
            # Готовим входы (текущие номера токенов) и выходы (следующие номера токенов)
            x = <YOUR CODE HERE>
            y = <YOUR CODE HERE>
            with (torch.no_grad()):
                # Получаем логиты и считаем лосс
                logits, _ = self.model(x)
                loss = self.calc_loss(logits, y)
                total_loss += loss.item() / len(self.eval_loader)
        return total_loss
