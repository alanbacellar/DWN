import os
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
modules = [f for f in os.listdir(dir_path) if f.endswith('.py') and f != '__init__.py']

for module in modules:
    module_name = module[:-3]
    submodule = importlib.import_module('.' + module_name, __name__)
    for item in dir(submodule):
        if not item.startswith("_"): 
            globals()[item] = getattr(submodule, item)
