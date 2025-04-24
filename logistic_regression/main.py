#@title Code - Load dependencies

#general
import io

# meus códigos
from lib.data_analysis import ReadDataset
from lib.ploting import plot_train
from lib.evaluate_model import compare_train_test

# # data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # machine learning
import keras
import ml_edu.results

# o seguinte código faz 2 coisas
# 1 - aleatorializa o embaralhamento dos dados para o modelo treinar entre outras coisas
# 2 - utiliza essa aleatoriedade como padrão em várias libs python como numpy e pandas
# OBS - quanto maior o número mais aleatório fica
keras.utils.set_random_seed(42)

# data visualization
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import seaborn as sns

def main():
    #Pegando os dados

    rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

    rice_dataset = rice_dataset_raw[[
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
    ]]

    #normalizando os dados
    features_mean = rice_dataset.mean(numeric_only=True)
    features_std = rice_dataset.std(numeric_only=True)
    numerical_features = rice_dataset.select_dtypes('number').columns
    
    normalized_dataset = (
        rice_dataset[numerical_features] - features_mean
    ) / features_std

    #copiar a coluna "classe"
    normalized_dataset['Class'] = rice_dataset['Class']

    #criando os dados para treino, teste e validação
    #classificando o arroz "Cammeo" como tipo "1"
    normalized_dataset['Class_Bool'] = (
        normalized_dataset['Class'] == 'Cammeo'
    ).astype(int)

    normalized_dataset.sample(10)
    number_samples = len(normalized_dataset) #vai pegar o número de itens
    index_80th = round(number_samples * 0.8)
    index_90th = index_80th + round(number_samples * 0.1)

    #shuffled significa "embaralhado", NÃO se confunda com "shuffled"
    shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100) #o "frac" é a porcentagem da quantidade dos dados que serão pegos #"randowm_state" é como se fosse a "chave" que será usada para embaralhar os dados, se o número for o mesmo a ordem será a mesma

    #atribuindo os dados às variáveis de treino
    train_data = shuffled_dataset.iloc[0:index_80th]
    validate_data = shuffled_dataset.iloc[index_80th:index_90th]
    test_data = shuffled_dataset.iloc[index_90th:]

    #Para evitar que o modelo acabe pegando o "label"(rótulo ou resultado) enquanto treina podemos colocá-lo em uma tabela separada
    label_columns = ['Class','Class_Bool']

    train_features = train_data.drop(columns=label_columns)
    train_labels = train_data['Class_Bool'].to_numpy()
    validate_features = validate_data.drop(columns=label_columns)
    validate_labels = validate_data['Class_Bool'].to_numpy()
    test_features = test_data.drop(columns=label_columns)
    test_labels = test_data['Class_Bool'].to_numpy()

    # #Devemos separar quais são as colunas que iremos usar como "features" para o modelo
    # input_features = [
    #     'Eccentricity',
    #     'Major_Axis_Length',
    #     'Area',
    # ]
    
    restart = "s"
    model_1_finish = False
    model_2_finish = False
    while restart.lower() != 'n':
        print("\nBem vindo à atividade de modelo de regressão linear")
        
        #@title análise
        #análise de dados
        responseDataAnalysis = str(input("\nVocê quer anilizar os dados antes de treinar o modelo? digite \"s\" para analisar ou qualquer coisa para não analisar"))
        if responseDataAnalysis.lower() == "s":
            print("\nAnalisando dados")
            ReadDataset(rice_dataset, normalized_dataset, test_data)

        #@title treino
        #treinar o modelo com os dados do taxi
        input("\nVamos treinar o modelo, digite qualquer coisa para continuar ")
        hyperparameter = float(input("\nDigite o hyperparametro (recomenda-se utilizar 0.001): "))
        batch = int(input("\nDigite o batch (recomenda-se utilizar 100): "))
        epoch = int(input("\nDigite a época (recomenda-se utilizar 60): "))
        threshold = float(input("\nDigite o \"threshold\" (recomenda-se utilizar 0.35): "))

        howMuchFeatures = "k"
        modifiedData_df = 0
        while howMuchFeatures.lower() != "m" and howMuchFeatures.lower() != "p":
            howMuchFeatures = str(input("\nVocê quer executar o modelo passando poucos ou muitos parâmetros como \"features\" para ele? digite \"m\" para muitos ou \"p\" para poucos: "))
            if howMuchFeatures.lower() != "m" and howMuchFeatures.lower() != "p":
                print("\nDigite uma letra válida")

        if howMuchFeatures == "p":
            input_features = [
                'Eccentricity',
                'Major_Axis_Length',
                'Area',
            ]

            print("\nTreinando modelo com poucos parâmetros")
            model_1 = plot_train(
                input_features=input_features,
                train_features=train_features,
                train_labels=train_labels,
                batch=batch,
                epochs=epoch,
                threshold=threshold,
                hyperparameter=hyperparameter
            )
            model_1_finish = True
        else:
            input_features = [
                'Eccentricity',
                'Major_Axis_Length',
                'Minor_Axis_Length',
                'Area',
                'Convex_Area',
                'Perimeter',
                'Extent',
            ]

            print("\nTreinando modelo com vários parâmetros")
            model_2 = plot_train(
                input_features=input_features,
                train_features=train_features,
                train_labels=train_labels,
                batch=batch,
                epochs=epoch,
                threshold=threshold,
                hyperparameter=hyperparameter
            )
            model_2_finish = True

        #@title análise modelo
        #análise do desempenho das predições do modelo
        print("\nVamos verificar o desempenho do modelo, para isso faremos predições com ele")
        input("\nDigite qualquer coisa para continuar ")

        #@title Code - Make predictions

        if(howMuchFeatures == "p"):
            test_metrics = model_1.evaluate(test_features, test_labels)
            compare_train_test(model_1, test_metrics)
        else:
            test_metrics = model_2.evaluate(test_features, test_labels)
            compare_train_test(model_2, test_metrics)

        #comparando os modelos
        if model_1_finish:
            if model_2_finish:
                compare_response = str(input("\nVocê treinou os dois modelos! quer compará-los? digite \"s\" para comparar ou qualquer coisa para proseguir: "))
                if compare_response.lower() == 's':
                    #fazendo a comparação
                    ml_edu.results.compare_experiment([model_1,model_2],['accuracy','auc'],test_features,test_labels)
                    plt.show()
            else:
                print("\nInfelizmente não é possível comparar os dois modelos, pois você treinou apenas o modelo 1, treine o outro para permitir a comparação!")
        else:
            if model_2_finish:
                print("\nInfelizmente não é possível comparar os dois modelos, pois você treinou apenas o modelo 2, treine o outro para permitir a comparação!")

        #reiniciar programa
        restart = str(input("\nQuer recomeçar o programa? digite \"n\" se não ou qualquer character caso queira continuar"))
        
#COMEÇAR O PROGRAMA
if __name__ == "__main__":
    main()
