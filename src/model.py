from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Model():
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        return

    def treino(self, X: pd.DataFrame, y: pd.DataFrame, tamanho_teste = 0.33) -> None:
        X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[1:], test_size=tamanho_teste, random_state=42, shuffle = False)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        model = LinearRegression()
        model.fit(X_train, y_train)

        self.model = model

        return
    
    def avaliacao_treino(self, y_test: pd.DataFrame, y_pred: pd.DataFrame) -> None:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("R-squared:", r2)

        plt.plot(y_test.to_list(), label = 'Real')
        plt.plot(y_pred, label = 'Predito')
        plt.legend()

        plt.show()

        return
    
    def salva_modelo(self, nome: str) -> None:
        if self.model == None:
            print('Nenhum  modelo foi treinado com esse objeto')
        else:
            with open(f'models/{nome}.pkl','wb') as f:
                pickle.dump(self.model,f)

        return
    
    def carrega_modelo(self, nome: str) -> None:
        with open(f'{nome}.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        return

    def predicao(self, X_test: pd.DataFrame) -> pd.DataFrame:
        y_pred = self.model.predict(X_test)

        return y_pred


