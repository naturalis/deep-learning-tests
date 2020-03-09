settings={
    "validation_split": 0.2,
    "conv_base": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),
    "batch_size": 64,
    "epochs": 200,
    "loss": "categorical_crossentropy",
    "optimizer": tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    "callbacks" : [ 
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=5, mode="auto", restore_best_weights=True),
        # tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-8 ),
        tf.keras.callbacks.ModelCheckpoint(trainer.get_model_save_path(), monitor="val_acc", save_best_only=True)
    ],
    "image_augmentation" : {
        "rotation_range": 90,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2, 
        "vertical_flip": False
    }
}