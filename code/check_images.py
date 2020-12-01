import os, argparse
from lib import baseclass, utils, dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--path",type=str)
    args = parser.parse_args()

    dataset = dataset.DataSet()
    dataset.set_environ(os.environ)
    dataset.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)    
    dataset.set_project(os.environ)

    dataset.logger.info("checking for corrupt files")

    if args.path:
        dataset.logger.info("checking for corrupt files in {}".format(dataset.image_path))
        bad_files = utils.verify_images(dataset.image_path)
    else:
        dataset.logger.info("checking for corrupt files in {}".format(args.path))
        bad_files = utils.verify_images(args.path)

    for file in bad_files:
        dataset.logger.info("found corrupt file: {} ({})".format(file["filename"],file["error"]))