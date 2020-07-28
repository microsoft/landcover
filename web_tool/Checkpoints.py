import os
import json

from .Utils import get_random_string

CHECKPOINT_DIRECTORY = "tmp/checkpoints/"


class Checkpoints():

    @staticmethod
    def create_new_checkpoint_directory(dataset_name, model_name, checkpoint_name):
        
        checkpoint_long_name = "%s-%s-%s" % (dataset_name, model_name, checkpoint_name)
        checkpoint_dir = os.path.join(CHECKPOINT_DIRECTORY, checkpoint_long_name)

        if os.path.exists(checkpoint_dir):
            raise ValueError("Checkpoint '%s' already exists" % (checkpoint_name))
        else:
            os.makedirs(checkpoint_dir, exist_ok=False)
            return checkpoint_dir

    @staticmethod
    def list_checkpoints():

        checkpoints = []
        for checkpoint_long_name in sorted(os.listdir(CHECKPOINT_DIRECTORY)):
            checkpoint_dir = os.path.join(CHECKPOINT_DIRECTORY, checkpoint_long_name)
            if len(os.listdir(checkpoint_dir)) > 0:

                dataset_name, model_name, checkpoint_name = checkpoint_long_name.split("-")

                with open(os.path.join(checkpoint_dir, "classes.json"), "r") as f:
                    classes = json.loads(f.read().strip())

                checkpoints.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "name": checkpoint_name,
                    "directory": checkpoint_dir,
                    "classes": classes
                })

        return checkpoints


