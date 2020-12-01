from lib import baseclass, utils, dataset

if __name__ == "__main__":

    dataset = dataset.DataSet()
    dataset.set_environ(os.environ)
    dataset.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)    
    dataset.set_project(os.environ)

    print(dataset.image_path)
    # bad_files = utils.verify_images(dataset.image_path)