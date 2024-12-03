bl_info = {
    "name": "Kirby",
    "author": "Thomas Wolfgang Peschlow",
    "version": (0, 0, 0),
    "blender": (4, 1, 0),
    "description": "Semi-neural mesh style transfer",
    "category": "Mesh"
}

import bpy
from .operators import transfer, process
from .panels import main_panel
from . import properties

def register():
    transfer.register()
    process.register()
    main_panel.register()
    properties.register()

def unregister():
    transfer.unregister()
    process.unregister()
    main_panel.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()
