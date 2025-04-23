import pandas as pd
import plotly.express as px

#configurando o pandas
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

def ReadDataset(df: pd.DataFrame, normalized_df: pd.DataFrame, test_df: pd.DataFrame):
    #analizando o dataset
    print(
        f'The shortest grain is {df.Major_Axis_Length.min():.1f}px long,'
        f' while the longest is {df.Major_Axis_Length.max():.1f}px.'
    )
    print(
        f'The smallest rice grain has an area of {df.Area.min()}px, while'
        f' the largest has an area of {df.Area.max()}px.'
    )
    print(
        'The largest rice grain, with a perimeter of'
        f' {df.Perimeter.max():.1f}px, is'
        f' ~{(df.Perimeter.max() - df.Perimeter.mean())/df.Perimeter.std():.1f} standard'
        f' deviations ({df.Perimeter.std():.1f}) from the mean'
        f' ({df.Perimeter.mean():.1f}px).'
    )
    print(
        f'This is calculated as: ({df.Perimeter.max():.1f} -'
        f' {df.Perimeter.mean():.1f})/{df.Perimeter.std():.1f} ='
        f' {(df.Perimeter.max() - df.Perimeter.mean())/df.Perimeter.std():.1f}'
    )

    print("\n")

    #criando gráficos com $method px.scatter
    for x_axis_data, y_axis_data in [
        ('Area', 'Eccentricity'),
        ('Convex_Area', 'Perimeter'),
        ('Major_Axis_Length', 'Minor_Axis_Length'),
        ('Perimeter', 'Extent'),
        ('Eccentricity', 'Major_Axis_Length'),
    ]:
        px.scatter(df, x=x_axis_data, y=y_axis_data, color='Class').show()

    answer_3d_Grafics = str(input("Você quer criar um gráfico 3d com 3 elementos (Area, Eccentricity e Major_Axis_Length)?\n(OBS) a possibilidade de especificar os elementos ainda não foi adicionada\nResponda com 's' ou 'n'"))
    if answer_3d_Grafics == 's':
        px.scatter_3d(
            df, 
            x='Area', 
            y='Eccentricity', 
            z='Major_Axis_Length',
            color='Class'
        ).show()

    print("\nDevemos normalizar os dados para ajudar o modelo a treinar e a fazer melhores previsões, vamos analisar os dados normalizados com o *z-score*")
    print("\n" + normalized_df.head())

    print("\nDevemos separar alguns dados aleatórios para treinar e testar o modelo, observe alguns dados para o teste: \n")
    print(test_df.head())

    #Finalizado!
    print("Análise finalizada!")

