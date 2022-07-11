# Nucleus File


# Model class
class TitanicModel:
    def __init__(self, version, model_name, data_dir, saved_weights_dir):
        self.data_dir = data_dir
        version_model_name = fr'{version}_{model_name}'

        self.model_saved_weights_dir = fr'{saved_weights_dir}\{version_model_name}'
        self.log_dir = fr'Model-Graphs&Logs\Model-Data_{model_name}\Logs\{version_model_name}'
        self.metric_dir = fr'Model-Graphs&Logs\Model-Data_{model_name}\Metric-Graphs\{version_model_name}'

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor_pipeline = None

        self.model = None
        self.history = None

    # Data Preprocessing
    def preprocess(self):
        pass


# Executor
if __name__ == '__main__':
    model_instance = TitanicModel(version='first_gen', model_name='titanic',
                                  data_dir=fr'D:\Data-Warehouse\Titanic-Data',
                                  saved_weights_dir=fr'D:\Saved-Models\Titanic-Models')
    model_instance.preprocess()
