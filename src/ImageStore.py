import os
import random


class ImageStore:

    def __init__(self, file_path):
        self.file_path = file_path

    def fetch_random_image(self):
        files = os.listdir(self.file_path)
        index = random.randrange(0, len(files))
        return files[index]
