import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
import torch
import logging
import requests
from PIL import Image
import base64
from ultralytics import YOLO
from tqdm import tqdm
import multiprocessing as mp
import time
from skimage.restoration import estimate_sigma
from . import comfyui_interface as comfy
from clip_interrogator import Config, Interrogator
from .pipeline import *
import os


import comfy.model_management as mm

def apply_args(args):
    global set_vram_to
    global vram_state
    if args.highvram is not None:
        set_vram_to = mm.VRAMState.HIGH_VRAM
        vram_state = mm.VRAMState.HIGH_VRAM
        logging.info("Setting VRAM to high")



class ComfyUIPipeline:
    def __init__(self, ckpt_name, controlnet_name, size, pipe_classes = DEFAULT_PIPE_CLS, box_scaling = 1.1, use_control_net = False, verbose = False, yolo_result_out = None, clip_enabled = False, min_box_size = 64):
        self.yolo = YOLOPipeline(pipe_classes=pipe_classes, result_out=yolo_result_out)
        self.dilate_kernel = np.ones((5, 5), np.uint8)
        self.min_box_size = min_box_size
        self.box_scaling = box_scaling
        self.use_control_net = use_control_net
        self.size = size
        self.clip_enabled = clip_enabled
    
        self.verbose = verbose
        if(verbose):
            os.makedirs("./verbose/", exist_ok=True)

        config = Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k")#"ViT-L-14/openai")
        total_vram = torch.cuda.get_device_properties(0).total_memory
        logging.info(f"Available VRAM {total_vram} bytes")
        if total_vram / 1e9 < 12:
            config.apply_low_vram_defaults()

        if clip_enabled:
            self.clip_interrogator = Interrogator(config)
        self.ci = comfy.ComfyUIInterface(ckpt_name, controlnet_name, size, use_pose=self.use_control_net)

    def __str__(self):
        return f"ComfyUIPipeline"

    def checkAvailable(self):
        return True

    def apply_settings(self, use_sdxl = False, use_refiner = False):
        #self.use_sdxl = use_sdxl
        #self.refiner = use_refiner

        #ckpt = "sd_xl_base_1.0.safetensors" if use_sdxl else "v2-1_768-ema-pruned.safetensors"
        #size = (1024, 1024) if use_sdxl else (768, 768)
        #self.ci = comfy.ComfyUIInterface(ckpt, size, use_pose=self.use_control_net)
        pass



    def loadImg(self, path):
        logging.debug(f"Loading image {path}")
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    def cropImageAndMask(self, img, mask, box, size):
        sizeM = max(size[0], size[1])
        w = int(np.max(box[2:] - box[:2]) * self.box_scaling * 0.5)
        c = (box[:2] + box[2:]) // 2

        img = img[c[1]-w:c[1]+w, c[0]-w:c[0]+w]
        mask = mask[c[1]-w:c[1]+w, c[0]-w:c[0]+w]

        img = cv2.resize(img, (sizeM, sizeM))#, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (sizeM, sizeM))#, interpolation=cv2.INTER_CUBIC)

        p = np.array([sizeM - size[0], sizeM - size[1]]) // 2
        h = w
        if size[0] < size[1]:
            img = img[:,p[0]:-p[0]]
            mask = mask[:,p[0]:-p[0]]
            w = int(h * size[0] / size[1])
        elif size[1] < size[0]:
            img = img[p[1]:-p[1],:]
            mask = mask[p[1]:-p[1],:]
            h = int(w * size[1] / size[0])

        return img, mask, (c[0]-w, c[0]+w, c[1]-h, c[1]+h) 
    

    ####### parallel ########
    def prepare_img(self, box, img, pad_size, caption, mask_box, bar = None):
        boxSize = np.minimum(box[0][3] - box[0][1], box[0][2] - box[0][0])
        if boxSize < self.min_box_size:
            return None

        #mask_scaled = cv2.dilate(box[1], self.dilate_kernel, iterations=3)
        mask_scaled = box[1]
        mask_scaled = (mask_scaled * 255).astype(np.uint8)

        mask = np.zeros(img.shape, dtype=np.uint8)
        if mask_box:
            b = box[0]
            mask[b[1]:b[3], b[0]:b[2]] = 255
        else:
            mask[pad_size:-pad_size,pad_size:-pad_size,:] = mask_scaled[:,:,None]

        #size = (1024, 1024) if self.use_sdxl else (768, 768)
        size = self.size
        img_sample, mask_sample, box_sample = self.cropImageAndMask(img, mask, box[0], size)

        label_cls = self.yolo.pipe_classes.get_by_name(box[2])
        if not self.clip_enabled:
            prompt = label_cls.prompt
        elif caption is None:
            if not bar is None:
                bar.set_description("CLIP Interrogate")

            caption = self.clip_interrogator.interrogate_fast(Image.fromarray(img_sample), max_flavors=3)

            prompt = label_cls.prompt + ", " + ",".join(caption.split(","))
            #prompt = caption.split(",")[0] + ", " + label_cls.prompt# + ", " + ",".join(caption.split(",")[1:])
        else:
            prompt = caption

        negative_prompt = label_cls.neg_prompt

        if self.verbose:
            logging.debug(f"Prompt: '{prompt}'\nNegative prompt: '{negative_prompt}'")
            plt.imsave(f"./verbose/{np.random.randint(0, 1000)}.jpg", img_sample)
            plt.imsave(f"./verbose/{np.random.randint(0, 1000)}.jpg", mask_sample)

        return box, size, prompt, negative_prompt, img_sample, mask_sample, box_sample

    def inpaint_parallel(self, prepared_data, steps, denoising_strength):
        box, size, prompt, negative_prompt, img_sample, mask_sample, box_sample = prepared_data

        img_paint = self.ci.run_inpainting(img_sample / 255.0, mask_sample[..., 0] / 255.0, prompt, negative_prompt, steps, denoising_strength)
        img_paint = (img_paint * 255).astype(np.uint8)

        if self.verbose:
            plt.imsave(f"./verbose/{np.random.randint(0, 1000)}.jpg", img_paint)  

        return box, box_sample, img_paint, mask_sample

    def merge_img(self, inpaint_result, img):
        box_sample, img_paint, mask_sample = inpaint_result
        size = (box_sample[1] - box_sample[0], box_sample[3] - box_sample[2])
        m = cv2.resize(mask_sample, size) / 255.0
        img[box_sample[2]:box_sample[3], box_sample[0]:box_sample[1]] = cv2.resize(img_paint, size) * m + img[box_sample[2]:box_sample[3], box_sample[0]:box_sample[1]]  * (1 - m)

        return img

    def inpaint_img_parallel(self, img_path, steps = 20, denoising_strength = 0.6, mask_box = False, caption = None):
        steps = int(steps)
        denoising_strength = float(denoising_strength)

        orig_img = self.loadImg(img_path)
        img = orig_img
        
        logging.debug("Extracting boxes")
        boxes = self.yolo.segmentImage(img, verbose=self.verbose)

        if self.verbose:
            img_verb = img.copy()
            for b in boxes:
                img_verb = cv2.rectangle(img_verb, b[0][:2], b[0][2:], color=(255,0,0), thickness=8)
            plt.imsave(f"./verbose/{np.random.randint(0, 1000)}.jpg", img_verb)
        
        pad_size = np.max(img.shape[:2])
        img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], 'edge')

        #pool = mp.Pool(8)

        logging.debug("Preparing image")
        tasksA = []
        for b in tqdm(boxes, desc="Preparing image", position=tqdm._get_free_pos(), leave=False):
            b = (b[0] + pad_size, b[1], b[2], b[3])
            tasksA.append(self.prepare_img(b, img, pad_size, caption, mask_box))

        tasksB = []
        """for task in tqdm(tasksA):
            prepared_data = task.get()
            if not prepared_data is None:
                tasksB.append(prepared_data)"""

        logging.debug("Inpainting")
        for prepared_data in tqdm(tasksA, desc="Inpaint image", position=tqdm._get_free_pos(), leave=False):
            if not prepared_data is None:
                tasksB.append(self.inpaint_parallel(prepared_data, steps, denoising_strength))

        logging.debug("Sorting objects")
        sorted(tasksB, key=lambda x: x[0][3])

        logging.debug("Merging images")
        crops = []
        for _, box_sample, img_paint, mask_sample in tqdm(reversed(tasksB), desc="Merge images", position=tqdm._get_free_pos(), leave=False):
            img = self.merge_img((box_sample, img_paint, mask_sample), img)

            crops.append(img_paint)

        #pool.close()
        #pool.join()

        img = img[pad_size:-pad_size, pad_size:-pad_size, :]


        return img, crops, orig_img
    
    ####
    def inpaint_img(self, img_path, steps = 20, denoising_strength = 0.6, mask_box = False, caption = None):
        return self.inpaint_img_parallel(img_path, steps, denoising_strength, mask_box, caption)
