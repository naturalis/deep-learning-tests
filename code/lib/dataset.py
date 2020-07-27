import os
from hashlib import md5
import tensorflow as tf

class DataSet():

    model_trainer = None
    model_summary = None
    model_summary_hash = None
    model_note = None

    def set_note(self,note):
        if not note is None:
            self.model_note = note

    def ask_note(self):
        while not self.model_note:
            self.model_note = input("enter model note: ")

    def make_dataset(self,model_trainer):
        self.model_trainer = model_trainer
        self._set_model_summary()

        print(" ==> " + self.model_trainer.model_name)
        print(" ==> " + self.model_trainer.base_model.name)
        # print(" ==> " + self.model_summary)
        print(" ==> " + self.model_summary_hash)
        print(" ==> " + str(self.model_trainer.get_preset("image_augmentation")))
        print(" ==> " + str(self.model_trainer.get_preset("validation_split")))
        print(" ==> " + str(self.model_trainer.get_preset("initial_learning_rate")))
        print(" ==> " + str(self.model_trainer.get_preset("batch_size")))
        print(" ==> " + str(self.model_trainer.get_preset("epochs")))
        print(" ==> " + str(self.model_trainer.get_preset("freeze_layers")))
        print(" ==> " + str(self.model_trainer.class_image_minimum)) # do not convert
        print(" ==> " + str(self.model_trainer.class_image_maximum)) # do not convert
        print(" ==> " + str(self.model_trainer.project_root))
        print(" ==> " + str(self.model_trainer.class_list_file_json))
        print(" ==> " + str(self.model_trainer.class_list_file_csv))
        print(" ==> " + str(self.model_trainer.image_list_file_csv))
        print(" ==> " + str(self.model_trainer.image_path))
        print(" ==> " + str(self.model_trainer.class_list_file))
        print(" ==> " + str(self.model_trainer.downloaded_images_file))
        print(" ==> " + self.model_note)
        print(" ==> " + str(self.model_trainer.timestamp))

        image_table = self.model_trainer.traindf.sort_values(by=[self.model_trainer.COL_CLASS,self.model_trainer.COL_IMAGE]).values.tolist()
        image_table = list(map(lambda x: [x[0], os.path.basename(x[1]) ], image_table))
        print(image_table)

        print(" ==> " + str(self.model_trainer.class_list))

        for phase, callbacks in enumerate(self.model_trainer.model_settings["callbacks"]):
            for callback in callbacks:
                tmp = str(callback)
                print(str(phase) + " ==> " + str(tmp))
                if tmp.find("ReduceLROnPlateau") > -1:
                    print("   ReduceLROnPlateau ==> " + str(callback.monitor))
                    print("   ReduceLROnPlateau ==> " + str(callback.factor))
                    print("   ReduceLROnPlateau ==> " + str(callback.patience))
                    print("   ReduceLROnPlateau ==> " + str(callback.min_lr))
    
        for phase, optimizers in enumerate(self.model_trainer.model_settings["optimizer"]):
            print(optimizers)
            print(isinstance(optimizers, list))


        
        # for phase, optimizers in enumerate(self.model_trainer.model_settings["optimizer"]):
        #     if isinstance(self.model_trainer.model_settings["optimizer"], list):
        #         for optimizer in optimizers:
        #             print(str(phase) + " ==> " + str(optimizer))
        #     else:
        #         print(" ==> " + self.model_trainer.model_settings["optimizer"])

    def _set_model_summary(self):
        stringlist = []
        self.model_trainer.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)
        self.model_summary_hash = md5(self.model_summary.encode('utf-8')).hexdigest()


#         <tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop object at 0x7f1aa6f4d0b8>
#         "optimizer_learning_rate": [
#             learning_rate
#         ],


