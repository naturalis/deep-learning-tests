class DataSet():

    def __init__(self):
        super().__init__()


    def create_dataset(self):


"base_model": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),
    print(self.base_model.name)
    complete model:
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)
        exit(0)



"callbacks" :
    self.model_settings["callbacks"]
        [
            [ 
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, verbose=1),
                # tf.keras.callbacks.TensorBoard(trainer.get_tensorboard_log_path()),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-8, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(trainer.get_model_path(), monitor="val_acc", save_best_only=True, save_freq="epoch", verbose=1)
            ]
        ],


        str(Obj)
        <tensorflow.python.keras.callbacks.ReduceLROnPlateau object at 0x7f6ef1dfd4e0>
            monitor
            factor
            patience
            min_lr


        <tensorflow.python.keras.losses.CategoricalCrossentropy object at 0x7f1aa6f57f98>
            no params


        <tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop object at 0x7f1aa6f4d0b8>
        "optimizer_learning_rate": [
            learning_rate
        ],



        if 'IMAGE_AUGMENTATION' in os_environ:
            self.presets.update( {'image_augmentation' : json.loads(os_environ.get("IMAGE_AUGMENTATION")) } )    
        else:
            self.presets.update( {'image_augmentation' : {
                "rotation_range": 90,
                "shear_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2, 
                "vertical_flip": False
            } } )

        self.presets.update( { "validation_split" : float(os_environ.get("VALIDATION_SPLIT")) if "VALIDATION_SPLIT" in os_environ else 0.2 } )
        self.presets.update( { "initial_learning_rate" : float(os_environ.get("INITIAL_LR")) if "INITIAL_LR" in os_environ else 1e-4 } )
        self.presets.update( { "batch_size" : int(os_environ.get("BATCH_SIZE")) if "BATCH_SIZE" in os_environ else 64 } )
        self.presets.update( { "epochs" : json.loads(os_environ.get("EPOCHS")) if "EPOCHS" in os_environ else [ 200 ]   } )
        self.presets.update( { "freeze_layers" : json.loads(os_environ.get("FREEZE_LAYERS")) if "FREEZE_LAYERS" in os_environ else [ "none" ] } )










    def set_timestamp(self):
        d = datetime.now()
        self.timestamp = "{0}{1:02d}{2:02d}-{3:02d}{4:02d}{5:02d}".format(d.year,d.month,d.day,d.hour,d.minute,d.second)

    def set_debug(self,state):
        self.debug = state

    def set_model_name(self,model_name=None):
        if model_name is None:
            self.model_name = self.timestamp
        else:
            self.model_name = model_name
        self.logger.info("model name: {}".format(self.model_name))

    def set_project_folders(self, project_root, image_path=None):
        self.project_root = project_root

        if not os.path.isdir(self.project_root):
            raise FileNotFoundError("project root doesn't exist: {}".format(self.project_root))

        self.logger.info("project root: {}".format(self.project_root))

        self.class_list_file_json = os.path.join(self.project_root, "lists", "classes.json")
        self.class_list_file_csv = os.path.join(self.project_root, "lists", "classes.csv")
        self.image_list_file_csv = os.path.join(self.project_root, "lists", "images.csv")

        if image_path is not None:
            self.image_path = image_path
        else:
            self.image_path = os.path.join(self.project_root, "images")

        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
            self.logger.info("created image folder {}".format(self.image_path))

        self.class_list_file = os.path.join(self.project_root, "lists", "classes.csv")
        self.downloaded_images_file = os.path.join(self.project_root, "lists", "downloaded_images.csv")

    def set_class_image_minimum(self, class_image_minimum):
        class_image_minimum = int(class_image_minimum)
        if class_image_minimum > 2:
            self.class_image_minimum = class_image_minimum
        else:
            raise ValueError("class minimum must be greater than 2 ({})".format(class_image_minimum))

    def set_class_image_maximum(self,class_image_maximum):
        self.class_image_maximum = int(class_image_maximum)

    def read_class_list(self, class_col=0):
        if not os.path.isfile(self.class_list_file_model):
            raise FileNotFoundError("class list file not found: {}".format(self.class_list_file_model))
        self.class_list_file_class_col = class_col
        self.class_list = _csv_to_dataframe(self.class_list_file_model, [self.class_list_file_class_col])

        # making a list of just the classes that make the image minimum
        with open(self.class_list_file_model, 'r', encoding='utf-8-sig') as file:
            c = csv.reader(file)
            for row in c:
                if int(row[1])>=self.class_image_minimum:
                    self.classes_to_use.append(row[0])

    def class_list_apply_image_minimum(self):
        before = len(self.class_list)
        self.class_list = self.class_list[self.class_list[self.class_list_file_class_col].isin(self.classes_to_use)]
        after = len(self.class_list)
        self.logger.info("dropped {} out of {} classes due to image minimum of {}".format(
            before - after, before, self.class_image_minimum))

    def get_class_list(self):
        return self.class_list

    def copy_class_list_file(self):
        copyfile(self.class_list_file,self.class_list_file_model)

    def set_model_folder(self):
        self.model_folder = os.path.join(self.project_root, "models", self.model_name)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
            self.logger.info("created model folder {}".format(self.model_folder))

        self.class_list_file_model = os.path.join(self.model_folder, "classes.csv")
        self.downloaded_images_file_model = os.path.join(self.model_folder, "downloaded_images.csv")

    def set_model_settings(self, model_settings):
        self.model_settings = model_settings
        for setting in self.model_settings:
            self.logger.info("setting - {}: {}".format(setting, str(self.model_settings[setting])))

    # TODO: implement Test split
    def read_image_list_file(self, class_col=0, image_col=1):
        if not os.path.exists(self.downloaded_images_file_model):
            copyfile(self.downloaded_images_file, self.downloaded_images_file_model)
            self.logger.info("copied downloaded images file {}".format(self.downloaded_images_file_model))

        self.image_list_class_col = class_col
        self.image_list_image_col = image_col

        self.logger.info("reading images from: {}".format(self.downloaded_images_file_model))

        df = _csv_to_dataframe(filepath=self.downloaded_images_file_model,
                               usecols=[self.image_list_class_col, self.image_list_image_col])
        # if Test split
        #   df = df.sample(frac=1)
        #   msk = np.random.rand(len(df)) < 0.8
        #   self.traindf = df[msk]
        #   self.testdf = df[~msk]
        # # print(len(df), len(self.traindf), len(self.testdf))
        df[2] = self.image_path.rstrip("/") + "/" + df[2].astype(str)
        df.columns = [self.COL_CLASS, self.COL_IMAGE]
        self.traindf = df

        self.logger.info("read {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))


    def image_list_apply_class_list(self):
        before = len(self.traindf)
        self.traindf = self.traindf[self.traindf[self.COL_CLASS].isin(self.classes_to_use)]
        after = len(self.traindf)
        self.logger.info("dropped {} out of {} images due to image minimum of {}".format(
            before - after, before, self.class_image_minimum))

        self.logger.info("retained {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))


    def get_model_path(self):
        self.model_path = os.path.join(self.model_folder, "model.hdf5")
        return self.model_path

    def get_architecture_path(self):
        self.architecture_path = os.path.join(self.model_folder, "architecture.json")
        return self.architecture_path

    def get_analysis_path(self):
        self.analysis_path = os.path.join(self.model_folder, "analysis.json")
        return self.analysis_path

    def get_classes_path(self):
        self.classes_save_path = os.path.join(self.model_folder, "classes.json")
        return self.classes_save_path

    def load_model(self):
        self.model = tf.keras.models.load_model(self.get_model_path(),compile=False)
        self.logger.info("loaded model {}".format(self.get_model_path()))

        with open(os.path.join(self.get_classes_path())) as f:
            self.classes = json.load(f)

        self.logger.info("loaded {} classes from {}".format(len(self.classes),self.get_classes_path()))

    def set_presets(self,os_environ):
        if 'IMAGE_AUGMENTATION' in os_environ:
            self.presets.update( {'image_augmentation' : json.loads(os_environ.get("IMAGE_AUGMENTATION")) } )    
        else:
            self.presets.update( {'image_augmentation' : {
                "rotation_range": 90,
                "shear_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2, 
                "vertical_flip": False
            } } )

        self.presets.update( { "validation_split" : float(os_environ.get("VALIDATION_SPLIT")) if "VALIDATION_SPLIT" in os_environ else 0.2 } )
        self.presets.update( { "initial_learning_rate" : float(os_environ.get("INITIAL_LR")) if "INITIAL_LR" in os_environ else 1e-4 } )
        self.presets.update( { "batch_size" : int(os_environ.get("BATCH_SIZE")) if "BATCH_SIZE" in os_environ else 64 } )
        self.presets.update( { "epochs" : json.loads(os_environ.get("EPOCHS")) if "EPOCHS" in os_environ else [ 200 ]   } )
        self.presets.update( { "freeze_layers" : json.loads(os_environ.get("FREEZE_LAYERS")) if "FREEZE_LAYERS" in os_environ else [ "none" ] } )
        self.presets.update( { "metrics" : json.loads(os_environ.get("METRICS")) if "METRICS" in os_environ else [ "acc" ] } )
        # epochs [ 10, 200 ]
        # freeze_layers [ "base_model", "none" ] # 249

    def get_preset(self, preset):
        if preset in self.presets:
            return self.presets[preset]
        else:
            raise ValueError("preset {} doesn't exist: {}".format(preset))

