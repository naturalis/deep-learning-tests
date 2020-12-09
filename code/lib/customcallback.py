import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):

    current_epoch = 0
    dataset = None
    timer = None

    def set_dataset(self,dataset):
        self.dataset = dataset

    def set_timer(self,timer):
        self.timer = timer

    def update_datset(self):
        if self.dataset:
            self.dataset.set_epochs_trained(self.get_current_epoch())
            if self.timer:
                self.dataset.set_training_time(self.timer.get_time_passed())
            self.dataset.save_dataset()

    # def on_train_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Starting training; got log keys: {}".format(keys))

    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        # keys = list(logs.keys())
        # self.current_epoch = epoch
        # self.update_datset()
        pass

    def on_epoch_end(self, epoch, logs=None):
        # keys = list(logs.keys())
        self.current_epoch = epoch
        self.update_datset()

    def get_current_epoch(self):
        return self.current_epoch

    # def on_test_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start testing; got log keys: {}".format(keys))

    # def on_test_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop testing; got log keys: {}".format(keys))

    # def on_predict_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start predicting; got log keys: {}".format(keys))

    # def on_predict_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop predicting; got log keys: {}".format(keys))

    # def on_train_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_test_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_test_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
