import os
import importlib
import pkgutil

# Get the directory of the current package
package_dir = os.path.dirname(__file__)

# Loop through all modules in the package directory
for module_info in pkgutil.iter_modules([package_dir]):
    module_name = module_info.name
    module = importlib.import_module(f".{module_name}", package=__name__)

    # Import each attribute directly from the module
    for attribute_name in getattr(module, "__all__", []):
        globals()[attribute_name] = getattr(module, attribute_name)
