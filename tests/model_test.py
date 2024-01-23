import os
import pandas as pd
import unittest

from pathlib import Path
os.chdir(str(Path('../')))


class TestModel(unittest.TestCase):
    def setUp(self):
        from src.model import Model
        dataframe = pd.read_csv('../data/apple_stock.csv')

        self.X = dataframe[['Open', 'Volume']]
        self.y = dataframe['High'].shift(1)

        return

    def test_treino(self):
        modelo = Model()
        modelo.treino(self.X, self.y)

        self.assertIsNotNone(modelo.X_test, 'o treino não gerou a variável X_test')
        self.assertIsNotNone(modelo.X_train, 'o treino não gerou a variável X_train')
        self.assertIsNotNone(modelo.y_test, 'o treino não gerou a variável y_test')
        self.assertIsNotNone(modelo.y_train, 'o treino não gerou a variável y_train')

    def test_predicao(self):
        modelo = Model()
        modelo.treino(self.X, self.y)
        y_pred = modelo.predicao(modelo.X_test)

        self.assertIsNotNone(y_pred, 'não foi gerada nenhuma previsão')

    def test_salva_modelo(self):
        modelo = Model()
        modelo.treino(self.X, self.y)
        y_pred = modelo.predicao(modelo.X_test)
        nome_modelo = 'teste_geracao_de_modelo'
        modelo.salva_modelo(nome_modelo)

        arquivo_existe = os.path.exists('../models/{nome_modelo}.pkl')
        self.assertTrue(arquivo_existe, 'o arquivo do modelo não foi gerado')

# run the test
if __name__ == '__main__':
    unittest.main()