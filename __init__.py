bl_info = {
    "name": "Kirby",
    "author": "Thomas Wolfgang Peschlow",
    "version": (0, 0, 0),
    "blender": (4, 1, 0),
    "description": "Semi-neural mesh style transfer",
    "category": "Mesh"
}

import bpy
import sys
import subprocess
from .operators import transfer, process, load
from .panels import main_panel
from . import properties

external_packages = ["torch", "scipy"]

def register():
    for pkg in external_packages:
        ensure_package(pkg)
    transfer.register()
    process.register()
    load.register()
    main_panel.register()
    properties.register()

def unregister():
    transfer.unregister()
    process.unregister()
    load.unregister()
    main_panel.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()

def ensure_package(name:str):
    try:
        __import__(name)
    except ImportError:
        exec = sys.executable
        try:
            subprocess.check_call([exec, "-m", "pip", "ensurepip", "--upgrade"])
            subprocess.check_call([exec, "-m", "pip", "install", "--upgrade", name])
        except Exception as e:
            print(f"Failed to install {name}.")