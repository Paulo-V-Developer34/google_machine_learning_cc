import ml_edu.experiment

def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
    print('Comparando as métricas entre o treino e o teste')
    for metric, test_value in test_metrics.items():
        print("-------------")
        print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Test {metric}: {test_value}')

    