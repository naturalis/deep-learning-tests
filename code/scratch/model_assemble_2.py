    def assemble_model_2(self):

        # Create the base model from the pre-trained model --> MobileNetV2
        self.base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       weights='imagenet')

        self.base_model.trainable = False
        self.base_model.summary()

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # prediction_layer = tf.keras.layers.Dense(1)
        prediction_layer = tf.keras.layers.Dense(len(self.class_list), activation='softmax')

        self.model = tf.keras.Sequential([
          self.base_model,
          global_average_layer,
          prediction_layer
        ])

        # self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=self.predictions)


        base_learning_rate = 0.0001
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['acc'])


        print("\n\n====================== model 2 ===============================")

        self.model.summary()

        # self.model_settings["epochs"] = 10

        self.train_model()


        # self.base_model.trainable = True

        # # Let's take a look to see how many layers are in the base model
        # print("Number of layers in the base model: ", len(self.base_model.layers))

        # # Fine-tune from this layer onwards
        # fine_tune_at = 249

        # # Freeze all the layers before the `fine_tune_at` layer
        # for layer in self.base_model.layers[:fine_tune_at]:
        #     layer.trainable =  False

        # self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #               optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
        #               metrics=['acc'])

        # self.model.summary()

        # self.model_settings["epochs"] = 100

        # self.train_model()


