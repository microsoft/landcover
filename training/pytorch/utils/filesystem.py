import os


def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

