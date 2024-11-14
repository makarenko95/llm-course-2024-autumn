import torch
from torch import Tensor
from torch.nn import Module, Linear


class LoraLayer(Module):
    """
    Класс LoraLayer представляет слой Low-Rank Adaptation (LoRA),
    который добавляет адаптивные параметры для уменьшения размерности в линейных слоях нейронных сетей.
    Этот слой состоит из двух линейных подслоев, A и B, которые уменьшают размерность
    входных данных с `in_features` до `r` и затем увеличивают ее обратно до `out_features`.

    Атрибуты:
        A (Linear): Линейный слой, который уменьшает размерность входных данных до `r`.
        B (Linear): Линейный слой, который восстанавливает размерность данных до `out_features`
                    после уменьшения. Инициализируется с нулевыми весами для начала обучения
                    с нуля.
    """
    def __init__(self, in_features: int, out_features: int, r: int):
        """
        Инициализирует LoraLayer с заданными параметрами размерностей.

        Параметры:
            in_features (int): Размерность входных данных.
            out_features (int): Размерность выходных данных.
            r (int): Ранг адаптивных слоев, то есть промежуточная размерность.
        """
        super().__init__()
        self.A = Linear(<ВАШ КОД>, bias=False)
        self.B = Linear(<ВАШ КОД>, bias=False)
        self.B.weight.data = <ВАШ КОД>

    def forward(self, x: Tensor):
        """
        Прямое распространение данных через LoraLayer.
        Сначала входные данные проходят через слой A для уменьшения размерности, затем через слой B
        для восстановления до требуемой размерности.

        Параметры:
            x (Tensor): Входной тензор с размерностью `in_features`.

        Возвращает:
            Tensor: Тензор с размерностью `out_features` после прохождения через слои A и B.
        """
        return <ВАШ КОД>

    def load(self, a_weights: Tensor, b_weights: Tensor) -> None:
        """
        Загружает предварительно обученные веса для слоев A и B.

        Параметры:
            a_weights (Tensor): Тензор весов для слоя A.
            b_weights (Tensor): Тензор весов для слоя B.
        """
        self.A.weight.data = a_weights
        self.B.weight.data = b_weights


def merge(linear_layer: Linear, lora_layer: LoraLayer) -> None:
    """
    Объединяет параметры LoraLayer с параметрами заданного линейного слоя.
    Рассчитывает дельта-веса из матриц весов слоев A и B в LoraLayer
    и добавляет их к весам основного линейного слоя.

    Параметры:
        linear_layer (Linear): Основной линейный слой, к которому добавляются адаптивные параметры.
        lora_layer (LoraLayer): LoraLayer, веса которого объединяются с основным линейным слоем.
    """
    delta = <ВАШ КОД>
    linear_layer.weight.data += delta
