# Домашнее задание 3: "Transformers and beyond"

Для успешного выполнения задания, необходимо:

1. Реализовать ALiBi
2. Реализовать Grouped Query Attention

Заготовки функций находятся в файлах [`alibi.py`](./alibi.py) и [`gqa.py`](./gqa.py) соответственно.
Задание будет считаться выполненным, если будут пройдены все Unit-тесты. Тесты расположены в папке `tests` и запускаются командой:

```bash
python -m pytest -s Homework/03/tests/test_gqa.py
python -m pytest -s Homework/03/tests/test_alibi.py
```