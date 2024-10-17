import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
import torch
import requests
from PIL import Image
import base64
from ultralytics import YOLO
from tqdm import tqdm
import multiprocessing as mp
import time
from skimage.restoration import estimate_sigma
import logging


class ClassPromptLabel:
    def __init__(self, cls_name: str, prompt = "", neg_prompt = ""):
        self.cls_name = cls_name
        self.prompt = prompt
        self.neg_prompt = neg_prompt

class PipelineClasses:
    def __init__(self, init_labels):
        self.classes = []
        self.cls_names = []

        for l in init_labels:
            self.add(l)

    def add(self, label: ClassPromptLabel):
        self.classes.append(label)
        self.cls_names.append(label.cls_name)

    def contains(self, cls_name: str):
        return cls_name in self.cls_names
    
    def get_by_name(self, cls_name: str):
        return self.classes[self.cls_names.index(cls_name)]


######################## LABELS ####################
LABEL_PERSON = ClassPromptLabel("person",
                                #"beautiful face, photograph, realistic, cinematic, photorealistic",
                                #"ugly, deformed limbs, artistic, ugly face, comic, drawing, unrealistic")
                                "RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
                                "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,")
LABEL_BICYCLE = ClassPromptLabel("bicycle",
                                 "photograph, realistic, sharp, cinematic, photorealistic",
                                 "artistic, comic, drawing, blurry, unrealistic")
LABEL_CAR = ClassPromptLabel("car",
                             "photograph, realistic, sharp, cinematic, photorealistic",
                             "ugly, deformed, artistic, comic, drawing, blurry, unrealistic")
LABEL_TRUCK = ClassPromptLabel("truck",
                             LABEL_CAR.prompt,
                             LABEL_CAR.neg_prompt)
LABEL_BUS = ClassPromptLabel("bus",
                             LABEL_CAR.prompt,
                             LABEL_CAR.neg_prompt)

LABEL_GENERAL = ClassPromptLabel("general",
                                 "photograph, realistic, sharp, cinematic, photorealistic",
                                 "ugly, deformed, artistic, comic, drawing, blurry, unrealistic")

DEFAULT_PIPE_CLS = PipelineClasses([LABEL_PERSON])
############################################

class YOLOPipeline:
    def __init__(self, pipe_classes = DEFAULT_PIPE_CLS, model="external/yolo/yolov8m-seg.pt", result_out = None):
        with torch.no_grad():
            self.model = YOLO(model)
        self.names = self.model.names
        self.pipe_classes = pipe_classes
        self.result_out = result_out

        self.print_available_classes()


    def print_available_classes(self):
        no_columns = 8
        items = list(self.names.items())
        max_key_len = max(len(str(k)) for k in self.names.keys())
        max_val_len = max(len(v) for v in self.names.values())

        print("Available Classes with YoloV8:")
        for i, (k, v) in enumerate(items):
            print(f"{k:>{max_key_len}}: {v:<{max_val_len}}", end='  ')
            if (i + 1) % no_columns == 0 or i == len(items) - 1:
                print()  # Newline after every `cols` items or at the end

    @torch.no_grad()
    def segmentImage(self, img, minProb = 0.0, verbose=False):
        result = self.model.predict(img[..., ::-1], verbose=False, device="cuda")

        boxes = []
        for r in result:
            for i in range(len(r.boxes)):
                b = r.boxes[i]
                cls_id = int(b.cls.item())
                cls_name = self.names[cls_id]

                if not self.pipe_classes.contains(cls_name):
                    continue

                p = b.conf.item()
                if p < minProb:
                    continue

                box = b.xyxy.cpu().numpy().flatten().astype(np.int32)
                mask = r.masks[i].data[0].cpu().numpy()
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                
                footprint = np.sum(np.clip(mask[:,:], 0, 1))

                boxes.append((box, mask, cls_name, footprint))

            if verbose:
                r.save(filename="./verbose/yolo.png")
            if self.result_out is not None:
                r.save(filename=self.result_out)

        torch.cuda.empty_cache()

        return boxes


class NoiseManager:
    def estimate_variance(x):
        return estimate_sigma(x, channel_axis=2, average_sigmas=False)


class Automatic1111Pipeline:
    def __init__(self, url = "http://127.0.0.1:7860", pipe_classes = DEFAULT_PIPE_CLS, box_scaling = 1.5):
        self.url = url

        self.yolo = YOLOPipeline(pipe_classes=pipe_classes)
        self.dilate_kernel = np.ones((5, 5), np.uint8)
        self.min_box_size = 64
        self.box_scaling = box_scaling

    def __str__(self):
        return f"Automatic1111Pipeline"

    def checkAvailable(self):
        try:
            response = requests.get(url=f"{self.url}/info")
        except:
            logging.error("SDWeb API not found!!!")
            return False

        self.info = response.json()

        self.models = requests.get(url=f"{self.url}/sdapi/v1/sd-models").json()
        self.vaes = requests.get(url=f"{self.url}/sdapi/v1/sd-vae").json()

        return True
    
    def apply_settings(self, use_sdxl = False, use_refiner = False):
        #Find xl checkpoint
        sd_ckpt = ""
        sd_vae = "Automatic"

        for c in self.models:
            sd_ckpt = c["title"]
            if use_sdxl and "xl_base" in sd_ckpt:
                break
            elif not use_sdxl and "2-1_768" in sd_ckpt:
                break

        self.refiner = ""
        if use_sdxl and use_refiner:
            for c in self.models:
                if "xl_refiner" in c["title"]:
                    self.refiner = c["title"]
                    break
            logging.info(f"Using refiner model {self.refiner}")

        for c in self.vaes:
            v = c["model_name"]
            if use_sdxl and "sdxl" in v:
                sd_vae = v
                break

        logging.info(f"Setting model to {sd_ckpt} and VAE to {sd_vae}")

        payload = {
            "sd_model_checkpoint": sd_ckpt,
            "sd_vae": sd_vae
        }

        response = requests.post(url=f"{self.url}/sdapi/v1/options", json=payload)

        self.use_sdxl = use_sdxl


    def loadImg(self, path):
        logging.debug(f"Loading image {path}")
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def numpyImgToBytes(self, img):
        img = Image.fromarray(img)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    def interrogateCLIP(self, img_bytes):
        #print("CLIP interrogate")

        payload = {
            "image": img_bytes,
            "model": "clip"#"deepdanbooru"
        }

        response = requests.post(url=f"{self.url}/sdapi/v1/interrogate", json=payload)
        
        r = response.json()
        return r["caption"]
    
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
        
    def async_inpaint(self, payload):
        response = requests.post(url=f"{self.url}/sdapi/v1/img2img", json=payload)

        return response
    
    def check_progress(self, bar, text):
        response = requests.get(f"{self.url}/sdapi/v1/progress")

        data = response.json()
        pg = float(data["progress"])

        if not bar is None:
            bar.set_description(f"{text} {int(100 * pg)}%")
        
        return pg

    def inpaint(self, box, img, pad_size, caption, steps, denoising_strength, mask_box, bar = None, seed = -1):
        boxSize = np.minimum(box[0][3] - box[0][1], box[0][2] - box[0][0])
        if boxSize < self.min_box_size:
            return img, None

        mask_scaled = cv2.dilate(box[1], self.dilate_kernel, iterations=3)
        mask_scaled = (mask_scaled * 255).astype(np.uint8)

        mask = np.zeros(img.shape, dtype=np.uint8)
        if mask_box:
            b = box[0]
            mask[b[1]:b[3], b[0]:b[2]] = 255
        else:
            mask[pad_size:-pad_size,pad_size:-pad_size,:] = mask_scaled[:,:,None]

        size = (1024, 1024) if self.use_sdxl else (768, 768)
        img_sample, mask_sample, box_sample = self.cropImageAndMask(img, mask, box[0], size)

        img_bytes = self.numpyImgToBytes(img_sample)
        mask_bytes = self.numpyImgToBytes(mask_sample)

        label_cls = self.yolo.pipe_classes.get_by_name(box[2])
        prompt = label_cls.prompt + ", "
        if caption is None:
            if not bar is None:
                bar.set_description("CLIP Interrogate")
            caption = self.interrogateCLIP(img_bytes)
            #if not bar is None:
            #    bar.set_postfix_str(f"\nCaption: {caption}")

            prompt += caption.split(",")[0]
        else:
            prompt = caption

        payload = {
            "init_images" : [img_bytes],
            "mask": mask_bytes,
            "prompt": prompt,
            "negative_prompt": label_cls.neg_prompt,
            "steps": steps,
            "inpainting_mask_invert": 0,
            #"resize_mode": 1,
            #"inpaint_full_res": True,
            #"inpaint_full_res_padding": 128,
            "inpainting_fill": 1,
            "denoising_strength": denoising_strength,
            "include_init_images": True,
            "restore_faces": False,
            "width": size[0],
            "height": size[1],
            "sampler_index": "DPM++ SDE Karras",
            "seed": seed
        }
        if self.use_sdxl and self.refiner != "":
            payload["refiner_checkpoint"] = self.refiner
            payload["refiner_switch_at"] = 0.8

        if not bar is None:
            bar.set_description(f"Running SD")

        with mp.Pool(1) as pool:
            task = pool.apply_async(self.async_inpaint, (payload, ))
            
            while not task.ready():
                self.check_progress(bar, "Running SD")
                time.sleep(0.1)

            response = task.get()

        #response = requests.post(url=f"{self.url}/sdapi/v1/img2img", json=payload)

        if not bar is None:
            bar.set_description(f"Decoding result")

        r = response.json()
        img_paint = np.array(Image.open(io.BytesIO(base64.b64decode(r["images"][0]))))

        size = (box_sample[1] - box_sample[0], box_sample[3] - box_sample[2])
        img[box_sample[2]:box_sample[3], box_sample[0]:box_sample[1]] = cv2.resize(img_paint, size)#, interpolation=cv2.INTER_CUBIC)

        return img, img_paint
        
 

    def inpaint_img(self, img_path, steps = 20, denoising_strength = 0.6, mask_box = False, caption = None):
        orig_img = self.loadImg(img_path)
        img = orig_img

        #img_bytes = self.numpyImgToBytes(img)
        #caption = self.interrogateCLIP(img_bytes)
        #print(f"Img Caption: {caption}")

        logging.debug("Extracting boxes")
        boxes = self.yolo.segmentImage(img)
        
        pad_size = np.max(img.shape[:2])
        img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], 'edge')

        logging.debug("Inpainting")
        crops = []
        bar = tqdm(boxes)
        for b in bar:
            b = (b[0] + pad_size, b[1], b[2], b[3])
            img, crop = self.inpaint(b, img, pad_size, caption, steps, denoising_strength, mask_box, bar)
            if not crop is None:
                crops.append(crop)

        img = img[pad_size:-pad_size, pad_size:-pad_size, :]

        return img, crops, orig_img
        

    def inpaint_video(self, video_path, steps = 20, denoising_strength = 0.75, mask_box = False, caption = None):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened(): 
            logging.error("Error opening video stream or file")
            return
        
        seed = 13
        idx = 0
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logging.debug(f"Inpainting frame {idx + 1} / {framecount}")

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = self.yolo.segmentImage(img)

                pad_size = np.max(img.shape[:2])
                img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], 'edge')

                bar = tqdm(boxes)
                for b in bar:
                    b = (b[0] + pad_size, b[1], b[2])
                    img, crop = self.inpaint(b, img, pad_size, caption, steps, denoising_strength, mask_box, bar, seed)

                img = img[pad_size:-pad_size, pad_size:-pad_size, :]

                plt.imsave(f"./videos/out/frame_{idx}.jpg", img)
                idx += 1

        cap.release()

    ####### parallel ########
    def prepare_img(self, box, img, pad_size, caption, mask_box, bar = None):
        boxSize = np.minimum(box[0][3] - box[0][1], box[0][2] - box[0][0])
        if boxSize < self.min_box_size:
            return None

        mask_scaled = cv2.dilate(box[1], self.dilate_kernel, iterations=3)
        mask_scaled = (mask_scaled * 255).astype(np.uint8)

        mask = np.zeros(img.shape, dtype=np.uint8)
        if mask_box:
            b = box[0]
            mask[b[1]:b[3], b[0]:b[2]] = 255
        else:
            mask[pad_size:-pad_size,pad_size:-pad_size,:] = mask_scaled[:,:,None]

        size = (1024, 1024) if self.use_sdxl else (768, 768)
        img_sample, mask_sample, box_sample = self.cropImageAndMask(img, mask, box[0], size)

        img_bytes = self.numpyImgToBytes(img_sample)
        mask_bytes = self.numpyImgToBytes(mask_sample)

        label_cls = self.yolo.pipe_classes.get_by_name(box[2])
        prompt = label_cls.prompt + ", "
        if caption is None:
            if not bar is None:
                bar.set_description("CLIP Interrogate")
            caption = self.interrogateCLIP(img_bytes)

            prompt += caption.split(",")[0]
        else:
            prompt = caption

        negative_prompt = label_cls.neg_prompt

        return box, size, prompt, negative_prompt, img_bytes, mask_bytes, box_sample, mask_sample

    def inpaint_parallel(self, prepared_data, steps, denoising_strength, seed = -1):
        box, size, prompt, negative_prompt, img_bytes, mask_bytes, box_sample, mask_sample = prepared_data

        payload = {
            "init_images" : [img_bytes],
            "mask": mask_bytes,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "inpainting_mask_invert": 0,
            #"resize_mode": 1,
            #"inpaint_full_res": True,
            #"inpaint_full_res_padding": 128,
            "inpainting_fill": 1,
            "denoising_strength": denoising_strength,
            "include_init_images": True,
            "restore_faces": False,
            "width": size[0],
            "height": size[1],
            "sampler_index": "DPM++ SDE Karras",
            "seed": seed
        }
        if self.use_sdxl and self.refiner != "":
            payload["refiner_checkpoint"] = self.refiner
            payload["refiner_switch_at"] = 0.8

        response = requests.post(url=f"{self.url}/sdapi/v1/img2img", json=payload)

        r = response.json()
        img_paint = np.array(Image.open(io.BytesIO(base64.b64decode(r["images"][0]))))

        return box, box_sample, img_paint, mask_sample

    def merge_img(self, inpaint_result, img):
        box_sample, img_paint, mask_sample = inpaint_result
        size = (box_sample[1] - box_sample[0], box_sample[3] - box_sample[2])
        m = cv2.resize(mask_sample, size) / 255.0
        img[box_sample[2]:box_sample[3], box_sample[0]:box_sample[1]] = cv2.resize(img_paint, size) * m + img[box_sample[2]:box_sample[3], box_sample[0]:box_sample[1]]  * (1 - m)

        return img

    def inpaint_img_parallel(self, img_path, steps = 20, denoising_strength = 0.6, mask_box = False, caption = None):
        orig_img = self.loadImg(img_path)
        img = orig_img
        
        logging.debug("Extracting boxes")
        boxes = self.yolo.segmentImage(img)
        
        pad_size = np.max(img.shape[:2])
        img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], 'edge')

        pool = mp.Pool(8)

        logging.debug("Preparing image")
        tasksA = []
        for b in boxes:
            b = (b[0] + pad_size, b[1], b[2], b[3])
            tasksA.append(pool.apply_async(self.prepare_img, (b, img, pad_size, caption, mask_box)))

        tasksB = []
        for task in tqdm(tasksA, desc="Preparing images"):
            prepared_data = task.get()
            if not prepared_data is None:
                tasksB.append(pool.apply_async(self.inpaint_parallel, (prepared_data, steps, denoising_strength)))

        logging.debug("Inpainting")
        tasksA.clear()
        for task in tqdm(tasksB, desc="Inpainting"):
            result = task.get()
            tasksA.append(result)

        logging.debug("Sorting objects")
        sorted(tasksA, key=lambda x: x[0][3])

        logging.debug("Merging images")
        crops = []
        for _, box_sample, img_paint, mask_sample in tqdm(reversed(tasksA), desc="Merging images"):
            img = self.merge_img((box_sample, img_paint, mask_sample), img)

            crops.append(img_paint)

        pool.close()
        pool.join()

        img = img[pad_size:-pad_size, pad_size:-pad_size, :]
        
        ## add variance to masks (Needs SVGF variance estimation)
        """
        var = NoiseManager.estimate_variance(orig_img)
        var = np.mean(np.stack(var))
        var_mask = np.zeros_like(img[..., 0]).astype(np.float32)
        for b in boxes:
            var_mask += b[1]
        
        img_var = np.clip(var_mask, 0, 1) * np.random.randn(*img.shape[:2]) * var * 0.02
        img = np.clip(img + img * img_var[..., None], 0, 255).astype(np.uint8)"""


        return img, crops, orig_img
    
    ###################### Full Image Inpaint ########################
    def inpaint_img_full(self, img_path, steps = 20, denoising_strength = 0.6, caption = None):
        orig_img = self.loadImg(img_path)   
        img = orig_img

        logging.debug("Preparing image")   
        scale = 1
        base_size = (1024 * scale, 1024 * scale) if self.use_sdxl else (768 * scale, 768 * scale)
        size = base_size

        h, w = img.shape[:2]
        ms = max(w, h)
        pad = (0, 0)
        if w > h:
            pad = (w - h) // 2
            img = np.pad(img, [[pad, pad], (0, 0), (0, 0)])
        elif h > w:
            pad = (h - w) // 2
            img = np.pad(img, [(0, 0), [pad, pad], (0, 0)])

        img = cv2.resize(img, size)

        img_bytes = self.numpyImgToBytes(img)
        if caption is None:
            caption = self.interrogateCLIP(img_bytes)

        mask = (np.ones_like(img) * 255).astype(np.uint8)
        mask_bytes = self.numpyImgToBytes(mask)

        negative_prompt = LABEL_GENERAL.neg_prompt
        prompt = LABEL_GENERAL.prompt + ", " + caption

        logging.debug("Inpainting image")
        payload = {
            "init_images" : [img_bytes],
            "mask": mask_bytes,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "inpainting_mask_invert": 0,
            #"resize_mode": 1,
            #"inpaint_full_res": True,
            #"inpaint_full_res_padding": 128,
            "inpainting_fill": 1,
            "denoising_strength": denoising_strength,
            "include_init_images": True,
            "restore_faces": False,
            "width": size[0],
            "height": size[1],
            "sampler_index": "DPM++ SDE Karras",
            "seed": -1
        }
        if self.use_sdxl and self.refiner != "":
            payload["refiner_checkpoint"] = self.refiner
            payload["refiner_switch_at"] = 0.8

        response = requests.post(url=f"{self.url}/sdapi/v1/img2img", json=payload)

        r = response.json()
        img = np.array(Image.open(io.BytesIO(base64.b64decode(r["images"][0]))))

        img = cv2.resize(img, (ms, ms))

        if w > h:
            img = img[pad:-pad, ...]
        elif h > w:
            img = img[:, pad:-pad, :]

        return img, orig_img
    


########### Simple methods #############
class BlurPipeline:
    def __init__(self, pipe_classes = DEFAULT_PIPE_CLS):
        self.yolo = YOLOPipeline(pipe_classes=pipe_classes)

    def __str__(self):
        return f"BlurPipeline"

    def checkAvailable(self):
        return True
    
    def apply_settings(self, use_sdxl, use_refiner):
        pass

    def loadImg(self, path):
        logging.debug(f"Loading image {path}")
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def inpaint(self, img_path):
        orig_img = self.loadImg(img_path)
        img = orig_img
        
        logging.debug("Extracting boxes")
        boxes = self.yolo.segmentImage(img)

        mask = np.zeros_like(img).astype(np.float32)
        for b in boxes:
            mask += b[1][..., None]

        mask = np.clip(mask, 0, 1)

        logging.debug("Inpainting")
        s = (img.shape[1] // 8, img.shape[0] // 8)
        img = cv2.resize(img, s)
        img = cv2.blur(img, (7, 7))
        img = cv2.resize(img, (orig_img.shape[1], orig_img.shape[0]))

        img = orig_img * (1 - mask) + img * mask

        return img, [], orig_img
    
    def inpaint_img_parallel(self, img_path, t0, t1, t2, cap):
        return self.inpaint(img_path)

    def inpaint_img(self, img_path, t0, t1, t2, cap):
        return self.inpaint(img_path)
    

class PixelPipeline(BlurPipeline):
    def __init__(self, pipe_classes = DEFAULT_PIPE_CLS):
        self.yolo = YOLOPipeline(pipe_classes=pipe_classes)

    def __str__(self):
        return f"PixelPipeline"

    def inpaint(self, img_path):
        orig_img = self.loadImg(img_path)
        img = orig_img
        
        logging.debug("Extracting boxes")
        boxes = self.yolo.segmentImage(img)

        mask = np.zeros_like(img).astype(np.float32)
        for b in boxes:
            mask += b[1][..., None]

        mask = np.clip(mask, 0, 1)

        logging.debug("Inpainting")
        s = (img.shape[1] // 32, img.shape[0] // 32)
        img = cv2.resize(img, s)
        img = cv2.resize(img, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img = orig_img * (1 - mask) + img * mask

        return img, [], orig_img
    

class MaskingPipeline(BlurPipeline):
    def __init__(self, pipe_classes = DEFAULT_PIPE_CLS):
        self.yolo = YOLOPipeline(pipe_classes=pipe_classes)

    def __str__(self):
        return f"MaskingPipeline"

    def inpaint(self, img_path):
        orig_img = self.loadImg(img_path)
        img = orig_img
        
        logging.debug("Extracting boxes")
        boxes = self.yolo.segmentImage(img)

        mask = np.zeros_like(img).astype(np.float32)
        for b in boxes:
            mask += b[1][..., None]

        mask = np.clip(mask, 0, 1)

        logging.debug("Inpainting")
        img = orig_img * (1 - mask) + mask * 120

        return img, [], orig_img
