import importlib
import pkgutil

# urllib3 ships optional emscripten support that is not available in normal CPython
# environments. Walk the package on disk instead of importing every submodule so
# PyInstaller does not trigger warnings for modules that are not needed here.
package_name = 'urllib3'
package = importlib.import_module(package_name)
package_path = getattr(package, '__path__', None)

hiddenimports = []
if package_path:
    for _, submodule_name, _ in pkgutil.walk_packages(package_path, prefix=f'{package_name}.'):
        if submodule_name == 'urllib3.contrib.emscripten':
            continue
        hiddenimports.append(submodule_name)
