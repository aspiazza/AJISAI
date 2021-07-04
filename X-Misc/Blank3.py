class ModelSummaryClass(keras.callbacks.Callback):
    def model_summary(self):
        with open(f'{self.log_dir}_summary.txt', 'a') as log_file:
            sys.stdout = log_file
            self.model.summary()
            log_file.close()


self.model_summary = ModelSummaryClass()