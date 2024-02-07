import os
import pandas as pd
import pytest

from pathlib import Path
os.chdir(str(Path('../')))
#from src.model import Model

class TestModel():
    def test_treino(self, treino):
        modelo = treino

        print(type(modelo.X_test))
        assert modelo.X_test.empty == False, 'o treino não gerou a variável X_test'
        assert modelo.X_train.empty == False, 'o treino não gerou a variável X_train'
        assert modelo.y_test.empty == False, 'o treino não gerou a variável y_test'
        assert modelo.y_train.empty == False, 'o treino não gerou a variável y_train'

    def test_predicao(self, predicao):
        y_pred = predicao

        assert len(y_pred) > 0, 'não foi gerada nenhuma previsão'

    def test_salva_modelo(self, treino):
        modelo = treino
        nome_modelo = 'teste_geracao_de_modelo'
        modelo.salva_modelo(nome_modelo)

        diretorio_atual = os.getcwd()
        arquivos = os.listdir(diretorio_atual)
        print('=============================================================================================================')
        print(arquivos)

        exist = os.path.exists(fr'models/{nome_modelo}.pkl')
        assert exist == True, 'o arquivo do modelo não foi gerado'

