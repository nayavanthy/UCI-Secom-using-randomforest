# External imports
from os.path import join

# Internal imports
from tools.train import Train
from utils.file_io import save_artifact

def main(args):
    """
    Description: This function drives the flow of the application.

    :param args: Argparse object containing required arguments.
    """
    # Checking for mode
    if args.mode == "train":
        train_obj = Train(args.data_path)
        processesing_info = train_obj.run()

    # Save processing info if save mode is on
    if args.save_mode == "ON":
        for key, value in processesing_info.items():
            save_artifact(value, join(args.save_path, key))

    return True
