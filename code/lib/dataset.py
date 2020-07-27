import os, re
from hashlib import md5
import tensorflow as tf

class DataSet():

    model_trainer = None
    model_summary = None
    model_summary_hash = None
    model_note = None
    data_set = []

    def set_note(self,note):
        if not note is None:
            self.model_note = note

    def ask_note(self):
        while not self.model_note:
            self.model_note = input("enter model note: ")

    def make_dataset(self,model_trainer):
        self.model_trainer = model_trainer
        self._set_model_summary()

        self.data_set["model_name"] = self.model_trainer.model_name
        self.data_set["timestamp"] = str(self.model_trainer.timestamp)
        self.data_set["model_note"] = self.model_note

        self.data_set["model"] = { 
            "base_model" : str(self.model_trainer.base_model.name),
            "summary" : self.model_summary,
            "summary_hash" : self.model_summary_hash
        }
        
        self.data_set["training_settings"] = { 
            "validation_split" : str(self.model_trainer.get_preset("validation_split")),
            "image_augmentation" : str(self.model_trainer.get_preset("image_augmentation")),
            "batch_size" : str(self.model_trainer.get_preset("batch_size")),
            "class_image_minimum" : str(self.model_trainer.class_image_minimum),
            "class_image_maximum" : str(self.model_trainer.class_image_maximum),
        }

        regex = re.compile('(^<|[\s](.*)$)')
        
        call = []
        for phase, callbacks in enumerate(self.model_trainer.model_settings["callbacks"]):
            for callback in callbacks:
                if str(callback).find("ReduceLROnPlateau") > -1:
                    call.append(
                        "{} (monitor: {}; factor: {}; patience: {}; min_lr: {}".format(
                            regex.sub('',str(callback)),
                            str(callback.monitor),
                            str(callback.factor),
                            str(callback.patience),
                            str(callback.min_lr)
                        )
                    )
                else:
                    c.append(regex.sub('',str(callback)))
    
        opt = []
        for phase, optimizer in enumerate(self.model_trainer.model_settings["optimizer"]):
            lr = self.model_trainer.get_preset("learning_rate")
            opt.append("{} (lr: {})".format(str(optimizer), str(lr[phase])))


        self.data_set["training_phases"] = { 
            "epochs" : str(self.model_trainer.get_preset("epochs")),
            "freeze_layers" : str(self.model_trainer.get_preset("freeze_layers")),
            "optimizer" : opt,
            "callbacks" : call
        }

        print(self.data_set)
        

        # print(" ==> " + str(self.model_trainer.project_root))
        # print(" ==> " + str(self.model_trainer.class_list_file_json))
        # print(" ==> " + str(self.model_trainer.class_list_file_csv))
        # print(" ==> " + str(self.model_trainer.image_list_file_csv))
        # print(" ==> " + str(self.model_trainer.image_path))
        # print(" ==> " + str(self.model_trainer.class_list_file))
        # print(" ==> " + str(self.model_trainer.downloaded_images_file))

        # image_table = self.model_trainer.traindf.sort_values(by=[self.model_trainer.COL_CLASS,self.model_trainer.COL_IMAGE]).values.tolist()
        # image_table = list(map(lambda x: [x[0], os.path.basename(x[1]) ], image_table))
        # print(image_table)

        # print(" ==> " + str(self.model_trainer.class_list))



    def _set_model_summary(self):
        stringlist = []
        self.model_trainer.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)
        self.model_summary_hash = md5(self.model_summary.encode('utf-8')).hexdigest()

