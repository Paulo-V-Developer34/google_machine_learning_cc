from lib.build_model import run_experiment
from src.dataAnalysis import ReadDataset

#@title Code - Load dependencies

#general
import io

# # data
# import numpy as np
import pandas as pd

# # machine learning
# import keras

# data visualization
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import seaborn as sns

def main():
    #Analisando os dados

    #pd.read_csv
    chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

    # Updates dataframe to use specific columns.
    training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

    while True:
        print("\nBem vindo à atividade de modelo de regressão linear")
        
        #@title análise
        #análise de dados
        responseDataAnalysis = str(input("\nVocê quer anilizar os dados antes de treinar o modelo? digite \"s\" para analisar ou qualquer coisa para não analisar"))
        if responseDataAnalysis.lower() == "s":
            print("\nAnalisando dados")
            print
            ReadDataset(training_df)

        #@title treino
        #treinar o modelo com os dados do taxi
        input("\nVamos treinar o modelo, digite qualquer coisa para continuar")
        hyperparameter = float(input("\nDigite o hyperparametro (recomenda-se utilizar 0.001): "))
        batch = int(input("\nDigite o batch (recomenda-se utilizar 50): "))
        epoch = int(input("\nDigite a época (recomenda-se utilizar 20): "))

        label = 'FARE'

        howMuchFeatures = 0
        while howMuchFeatures <= 0 or howMuchFeatures > 2:
            howMuchFeatures = int(input("\nVocê quer executar o modelo passando 1 ou 2 parâmetros como \"features\" para ele? digite a quantidade"))
            if howMuchFeatures <= 0 or howMuchFeatures > 2:
                print("\nDigite um número válido")

        if howMuchFeatures == 1:
            features = ['TRIP_MILES']


            print("\nTreinando modelo com 1 parâmetro")
            model_1 = run_experiment(training_df, features, label, hyperparameter, epoch, batch)
        else:
            modifiedData_df = training_df.copy()
            modifiedData_df.loc[:, 'TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

            features = ['TRIP_MILES','TRIP_MINUTES']

            print("\nTreinando o modelo com 2 parâmetros")
            model_2 = run_experiment(training_df, features, label, hyperparameter, epoch, batch)


        

        #@title análise modelo
        #análise do desempenho das predições do modelo
        print("\nVamos verificar o desempenho do modelo: ")

        #reiniciar programa
        restart = str(input("\nQuer recomeçar o programa? digite \"n\" se não ou qualquer character caso queira continuar"))
        if restart.lower() == "n":
            break
        
#COMEÇAR O PROGRAMA
if __name__ == "__main__":
    main()
