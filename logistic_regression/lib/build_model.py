import keras
import pandas as pd
import ml_edu.experiment
import ml_edu.results
import numpy as np

def create_model( #usando type hint
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric]
) -> keras.Model:
    
    #definindo os inputs
    #notas
    # 1 - isso é um "list comprehension", pois colocamos um laço "for" dentro de uma lista
    model_inputs = [
        keras.Input(name=feature, shape=(1,))
        for feature in settings.input_features
    ]

    #definindo as camadas (3) --> input, hide layer(1), output
    concatenated_inputs = keras.layers.Concatenate()(model_inputs) #aqui estamos juntando todos os inputs em um único, esse é um conceito mais utilizado em redes neurais
    model_output = keras.layers.Dense(
        units=1, name='dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)

    #criando o modelo
    model = keras.Model(inputs=model_inputs, outputs=model_output) #esse é um dos motivos pela qual eu não gosto do python, eu havia escrito "output" ao invés de "outputs" e ele não disse nada

    #configurando o modelo
    model.compile(
        optimizer = keras.optimizers.RMSprop(
            settings.learning_rate
        ),
        loss = keras.losses.BinaryCrossentropy(),
        metrics = metrics
    )

    #retornando o modelo
    return model

def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:

    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )
    