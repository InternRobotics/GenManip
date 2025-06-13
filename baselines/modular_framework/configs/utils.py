import importlib
from configs import Config

def load_config(config_path):
    module_name, class_name = config_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    config_class = getattr(module, class_name)
    for key, value in config_class.__dict__.items():
        if not key.startswith('__'):
            setattr(Config, key, value)