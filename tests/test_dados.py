"""Testes simples das funções de dados.

Rodar na raiz do projeto: python -m unittest discover tests
"""
import unittest

import pandas as pd

from src.config import ALVO, FEATURES
from src.dados import criar_alvo, preparar_base, remover_duplicados, renomear_colunas
from src.modelagem import sem_vazamento, separar_treino_teste


class TestDados(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = preparar_base()

    def test_renomear_colunas(self):
        df = pd.DataFrame({"fixed acidity": [7.4], "quality": [5], "Id": [0]})
        self.assertEqual(list(renomear_colunas(df).columns), ["acidez_fixa", "qualidade"])

    def test_alvo(self):
        df = pd.DataFrame({"qualidade": [5, 6, 7, 8]})
        self.assertEqual(criar_alvo(df)[ALVO].tolist(), [0, 0, 1, 1])

    def test_remover_duplicados(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        self.assertEqual(len(remover_duplicados(df)), 2)

    def test_base_limpa(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)
        self.assertEqual(self.df.duplicated().sum(), 0)
        for col in FEATURES:
            self.assertIn(col, self.df.columns)

    def test_divisao_treino_teste(self):
        x_treino, x_teste, y_treino, y_teste = separar_treino_teste(self.df)
        # stratify: proporção de vinhos bons parecida nos dois conjuntos
        self.assertAlmostEqual(y_treino.mean(), y_teste.mean(), delta=0.01)
        self.assertAlmostEqual(len(x_teste) / len(self.df), 0.2, delta=0.01)
        self.assertTrue(sem_vazamento(x_treino, x_teste))


if __name__ == "__main__":
    unittest.main()
