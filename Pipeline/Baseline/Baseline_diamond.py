# Model Baseline

from flaml import AutoML


def flaml_baseline(training_data, training_labels, log_dir):
    automl = AutoML()

    automl_settings = {
        "time_budget": 1,
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": f"{log_dir}_flaml.log"
    }

    automl.fit(X_train=training_data, y_train=training_labels, **automl_settings)

    # print(automl.predict_proba(training_data).shape)
    print(automl.model)
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparameter config:', automl.best_config)
    print('Best accuracy on validation data: {0:.4g}'.format(1 - automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
