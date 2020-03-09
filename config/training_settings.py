class TrainingSettings:
    def get_settings(self):
        return {
            "validation_split": 0.2,
            "conv_base": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),
            "batch_size": 64,
            "epochs": 255,
            "loss": "categorical_crossentropy",
            "optimizer": tf.keras.optimizers.RMSprop(lr=1e-4)
        }