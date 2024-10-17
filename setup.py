import os
import subprocess

## Download comfy ui + extensions
os.makedirs("external", exist_ok=True)
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
subprocess.run("git clone https://github.com/comfyanonymous/ComfyUI", cwd="./external/", shell=True)
subprocess.run("git clone https://github.com/Fannovel16/comfyui_controlnet_aux/", cwd="./external/ComfyUI/custom_nodes/", shell=True)

## TODO automatic conda stuff
