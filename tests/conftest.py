import pytest
import pandas as pd

import os
from pathlib import Path
os.chdir(str(Path('./')))
from src.model import Model


@pytest.fixture(scope="class")
def setUp():
    diretorio_atual = os.getcwd()
    arquivos = os.listdir(diretorio_atual)
    print(arquivos)
    dataframe = pd.read_csv('data/apple_stock.csv')

    X = dataframe[['Open', 'Volume']]
    y = dataframe['High'].shift(1)

    return X, y

@pytest.fixture(scope="class")
def treino(setUp):
    # diretorio_atual = os.getcwd()
    # arquivos = os.listdir(diretorio_atual)
    # print('=============================================================================================================')
    # print(arquivos)

    #from src.model import Model
    
    modelo = Model()
    X,y = setUp
    modelo.treino(X,y)

    return modelo

@pytest.fixture(scope="class")
def predicao(treino):
    modelo = treino

    return modelo.predicao(modelo.X_test)


# @pytest.fixture(scope="class")
# def salva_modelo(treino):
#     nome_modelo = 'teste_geracao_de_modelo'
#     modelo = treino()
#     modelo.salva_modelo(nome_modelo)

#     return os.path.exists('../models/{nome_modelo}.pkl')