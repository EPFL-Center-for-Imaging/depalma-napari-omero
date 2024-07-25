"""
Packages the depalma-napari-omero project into a single exectuable with PyApp.
"""
import os
import shutil
import subprocess
from pathlib import Path

from depalma_napari_omero import __version__, __name__

script_directory = Path(__file__).parent

# Set environment variables
os.environ['PYAPP_PROJECT_NAME'] = __name__
os.environ['PYAPP_PROJECT_VERSION'] = __version__
os.environ['PYAPP_PYTHON_VERSION'] = "3.9"
os.environ['PYAPP_EXEC_SCRIPT'] = str((script_directory.parent / 'src' / __name__ / '__main__.py').resolve())
if os.name == 'nt':
    print("Building for Windows.")
    extension = '.exe'  # Windows
    platform = 'w64'
    os.environ['PYAPP_PROJECT_DEPENDENCY_FILE'] = str((script_directory / 'requirements.w64.txt').resolve())
else:
    print("Building for Linux / MacOS.")
    extension = ''  # Linux and MacOS
    platform = 'u64'
    os.environ['PYAPP_PROJECT_DEPENDENCY_FILE'] = str((script_directory / 'requirements.txt').resolve())

# Print them
print(f"{os.environ['PYAPP_PROJECT_NAME']=}")
print(f"{os.environ['PYAPP_PROJECT_VERSION']=}")
print(f"{os.environ['PYAPP_PYTHON_VERSION']=}")
print(f"{os.environ['PYAPP_EXEC_SCRIPT']=}")
print(f"{os.environ['PYAPP_PROJECT_DEPENDENCY_FILE']=}")

# Change directory and run cargo build
os.chdir(str(script_directory / 'pyapp-latest'))
subprocess.run(['cargo', 'build', '--release'], capture_output=True, text=True)
os.chdir(str(script_directory))

source_path = str((script_directory / 'pyapp-latest' / 'target' / 'release' / f'pyapp{extension}').resolve())
destination_path = str((script_directory.parent / 'release' / f'{__name__}_{platform}_{__version__}{extension}').resolve())

if os.path.exists(source_path):
    shutil.copy(source_path, destination_path)
    print(f"Copied {source_path} to {destination_path}")
else:
    print(f"{source_path} does not exist")