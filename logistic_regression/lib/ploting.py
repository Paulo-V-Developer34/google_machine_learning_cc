import keras
import ml_edu.experiment
import ml_edu.results
from lib.build_model import create_model, train_model

def plot_train(input_features, train_features, train_labels, hyperparameter, epochs, batch, threshold):
    #vamos definir as configurações do modelo

    settings = ml_edu.experiment.ExperimentSettings(
        learning_rate=hyperparameter,
        number_epochs=epochs,
        batch_size=batch,
        classification_threshold=threshold,
        input_features=input_features,
    )

    #pegando as métricas de avaliação do modelo
    metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy',
        threshold=settings.classification_threshold,
    ),
    keras.metrics.Precision(
        name='precision',
        thresholds=settings.classification_threshold,
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

    model = create_model(settings, metrics)

    experiment = train_model(
        'baseline', model, train_features, train_labels, settings
    )

    #mostrar o progresso do modelo para o usuário
    ml_edu.results.plot_experiment_metrics(experiment, ['accuracy','precision','recall'])
    ml_edu.results.plot_experiment_metrics(experiment, ['auc'])

    return experiment
