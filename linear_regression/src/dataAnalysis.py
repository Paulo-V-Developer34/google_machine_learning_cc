import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ReadDataset(df):
    #@title Code - Read dataset
    print('\nRead dataset completed successfully.')
    print('\nTotal number of rows: {0}\n\n'.format(len(df.index)))
    print(df.head(200))
    print('\n')
    print(df.describe(include='all'))

    #analisando a relação entre cada uma das colunas #a relação é dada em porcentagem em decimal
    print('\n')
    print(df.corr(numeric_only = True))

    #analisando a relação em gráfico com @method pairplot
    sns.pairplot(df, x_vars=["FARE","TRIP_MILES","TRIP_SECONDS"], y_vars=["FARE","TRIP_MILES","TRIP_SECONDS"])
    #o plt.show() serve para poder mostrar o gráfico ao usuário
    plt.show() 
