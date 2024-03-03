import numpy as np


class Function:
    class activate:
        """Функции активации"""
        @staticmethod
        def sigmoid(x: int | float):
            return 1 / (1 + np.exp(-x))

    class calculation:
        """Функции рассчёта"""
        @staticmethod
        def weighted_sum(inputs: list[int | float], weights: list[int | float]) -> float | int:
            """
            Взвешенная сумма входов

            :param inputs: массив входных данных
            :param weights: массив весов
            :return: взвешенная сумма входов
            """
            assert (len(inputs) == len(
                weights)), "Количество входных значений, должно соответствоовать количеству весов"
            output = 0
            for i in range(len(inputs)):
                output += (inputs[i] * weights[i])
            return output
