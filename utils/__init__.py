import yaml
import os
from PIL import Image


def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yaml_read(path):
    return yaml.safe_load(txtread(path=path))


def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)