import os, re, json
from hashlib import md5
# import tensorflow as tf
from lib import logclass
from lib import baseclass

class DataSet(baseclass.BaseClass):

    model_trainer = None
    model_summary = None
    model_note = None
    model_state = "constructing"
    model_retrain_note = None
    data_set = {}
    data_set_file = None
    env_vars = []

    def __init__(self):
        super().__init__()

    def set_model_trainer(self,model_trainer):
        self.model_trainer = model_trainer
        self.set_dataset_path(self.model_trainer.model_folder)
        self._set_model_summary()

    def set_dataset_path(self,model_path):
        self.data_set_file = os.path.join(model_path, "dataset.json")

    def set_note(self,note):
        if not note is None:
            self.model_note = note

    def set_environ(self,os_environ):
        for item in [
            'IMAGE_AUGMENTATION','REDUCE_LR_PARAMS','VALIDATION_SPLIT', \
            'LEARNING_RATE','BATCH_SIZE','EPOCHS','FREEZE_LAYERS','METRICS', \
            'CHECKPOINT_MONITOR', 'EARLY_STOPPING_MONITOR','CLASS_IMAGE_MINIMUM', \
            'CLASS_IMAGE_MAXIMUM'
        ]:
            if item in os_environ:
                self.env_vars.append({ item :  os_environ.get(item) })

    def update_model_state(self,model_state):
        self.model_state = model_state
        self.data_set["state"] = self.model_state

    def ask_note(self,message="enter model note"):
        while not self.model_note:
            self.model_note = input("{}: ".format(message))

    def ask_retrain_note(self,message="enter retrain note"):
        while not self.model_retrain_note:
            self.model_retrain_note = input("{}: ".format(message))
        self.data_set["model_retrain_note"] = self.model_retrain_note

    def make_dataset(self):
        self._make_dataset()
        # self._make_file_list()

    def open_dataset(self):
        with open(self.data_set_file) as f:
            self.data_set = json.load(f)
        f.close()
        self.logger.info("opened data set: {}".format(self.data_set_file))

    def get_dataset(self):
        return self.data_set

    def _make_dataset(self):
        self.data_set["project_name"] = self.model_trainer.project_name
        self.data_set["project_root"] = self.model_trainer.project_root
        self.data_set["model_name"] = self.model_trainer.model_name
        self.data_set["created"] = str(self.model_trainer.timestamp)
        self.data_set["state"] = self.model_state
        self.data_set["training_time"] = "n/a"
        self.data_set["model_note"] = self.model_note

        self.data_set["base_model"] = str(self.model_trainer.base_model.name)
        self.data_set["model_summary_hash"] = md5(self.model_summary.encode('utf-8')).hexdigest()

        self.data_set["class_image_minimum"] =self.model_trainer.class_image_minimum
        self.data_set["class_image_maximum"] =self.model_trainer.class_image_maximum
        self.data_set["class_count"] = len(self.model_trainer.class_list)
        self.data_set["class_count_before_maximum"] = len(self.model_trainer.complete_class_list)
        self.data_set["class_list_hash"] = md5(str(self.model_trainer.class_list).encode('utf-8')).hexdigest()

        self.data_set["training_settings"] = { 
            "validation_split" : self.model_trainer.get_preset("validation_split"),
            "image_augmentation" : str(self.model_trainer.get_preset("image_augmentation")),
            "batch_size" : self.model_trainer.get_preset("batch_size")
        }

        regex = re.compile('(^<|[\s](.*)$)')
        
        calls = []
        for phase, callbacks in enumerate(self.model_trainer.model_settings["callbacks"]):
            call = []
            for callback in callbacks:
                if str(callback).find("ReduceLROnPlateau") > -1:
                    call.append(
                        "{} (monitor: {}; factor: {}; patience: {}; min_lr: {})".format(
                            regex.sub('',str(callback)),
                            str(callback.monitor),
                            str(callback.factor),
                            str(callback.patience),
                            str(callback.min_lr)
                        )
                    )
                else:
                    call.append(regex.sub('',str(callback)))
            calls.append(call)
    
        opt = []
        for phase, optimizer in enumerate(self.model_trainer.model_settings["optimizer"]):
            lr = self.model_trainer.get_preset("learning_rate")
            opt.append("{} (lr: {})".format(regex.sub('',str(optimizer)), str(lr[phase])))

        self.data_set["training_phases"] = { 
            "epochs" : self.model_trainer.get_preset("epochs"),
            "freeze_layers" : self.model_trainer.get_preset("freeze_layers"),
            "optimizer" : opt,
            "callbacks" : calls
        }

        self.data_set["env_vars"] = self.env_vars


        # to be saved

        # self.model_summary,
        # image_table = self.model_trainer.traindf.sort_values(by=[self.model_trainer.COL_CLASS,self.model_trainer.COL_IMAGE]).values.tolist()
        # image_table = list(map(lambda x: [x[0], os.path.basename(x[1]) ], image_table))
        # print(image_table)

        # self.model_trainer.class_list
        # self.model_trainer.original_class_list


    def save_dataset(self):
        f = open(self.data_set_file, "w")
        f.write(json.dumps(self.data_set))
        f.close()
        self.logger.info("saved data set: {}".format(self.data_set_file))

    def _set_model_summary(self):
        stringlist = []
        self.model_trainer.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)

    def set_training_time(self,time_passed):
        self.data_set["training_time"]=time_passed


