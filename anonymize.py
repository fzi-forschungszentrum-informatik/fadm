import utils.pipeline as ap
import utils.io_functions as fadm_io
import logging
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2


DIFFUSION_VERSIONS = {
    "21":["v2-1_768-ema-pruned.safetensors",
          "control_v11p_sd21_openposev2.safetensors",
          (768, 768),
          "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors?download=truec",
          "https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_openposev2.safetensors?download=true"],
    "20ip":["v2_512-inpainting-ema.safetensors",
          "control_v11p_sd21_openposev2.safetensors",
          (768, 768),
          "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors?download=true",
          "https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_openposev2.safetensors?download=true"],
    "rv6ip":["rv6ip.safetensors",
          "control_openpose_v1.safetensors",
          (768, 768),
          "https://civitai.com/api/download/models/245627",
          "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/diffusion_pytorch_model.safetensors?download=true"],
}

# maps input from argparse method to the pipeline class
PIPELINE_MAPPING = {
    'auto1111': ap.Automatic1111Pipeline,
    'blur': ap.BlurPipeline,
    'pixel': ap.PixelPipeline,
    'mask': ap.MaskingPipeline,
    'default': None  # assuming cp.ComfyPipeline is the default
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def checkVersion(version):
    fadm_io.download_file(version[3], f"./external/ComfyUI/models/checkpoints/{version[0]}")
    fadm_io.download_file(version[4], f"./external/ComfyUI/models/controlnet/{version[1]}")
    
    return os.path.exists(f"./external/ComfyUI/models/checkpoints/{version[0]}") and os.path.exists(f"./external/ComfyUI/models/controlnet/{version[1]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Anonymizer for people. Project ANYMOS!!!")

    parser.add_argument("--input", default="./input", help="Input folder for images")
    parser.add_argument("--output", default="./output", help="Output folder for images")
    parser.add_argument("--ds", default=0.6, help="Denoising strength for SD model.")
    parser.add_argument("--s", default="25", help="Steps for SD model")
    parser.add_argument("--version", default="21", help="Diffusion Model version [21, 20ip, rv6ip]", choices=DIFFUSION_VERSIONS.keys())
    parser.add_argument("--mb", action="store_true", help="Use box as mask")
    parser.add_argument("--out_sbs", action="store_true", help="Output side-by-side")
    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model")
    parser.add_argument("--ref", action="store_true", help="Use SDXL refiner model")
    parser.add_argument("--parallel", action="store_true", help="Use parallel execution")
    parser.add_argument("--file", default="", help="Specific file")
    parser.add_argument("--prompt", default="", help="Overwrite caption created by CLIP interrogate stage")
    parser.add_argument("--video", default="", help="Use video file")
    parser.add_argument("--full", action="store_true", help="Single step full image anonymization")
    parser.add_argument("--interval", default="0,0", help="Interval of images in input folder")
    parser.add_argument("--pipeline_method", help="Pipeline method to use", choices=PIPELINE_MAPPING.keys())
    parser.add_argument("--no_pose", action="store_true", help="Disable ControlNet pose estimation")
    parser.add_argument("--enable_clip", action="store_true", help="enable clip interrogator")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument("--cuda_device", default="0", help="CUDA device id to use")
    parser.add_argument("--highvram", action="store_true", help="High VRAM mode")
    parser.add_argument("--disable_min_scale", action="store_true", help="Disables skipping anonymization on boxes smaller than 64px")

    args = parser.parse_args()
    return args

def runImg(pipe, f, args):
    cap = args.prompt if args.prompt != "" else None

    if args.parallel:
        img, crops, orig_img = pipe.inpaint_img_parallel(f, args.s, args.ds, args.mb, cap)
    elif args.full:
        img, orig_img = pipe.inpaint_img_full(f, args.s, args.ds, cap)
    else:
        img, crops, orig_img = pipe.inpaint_img(f, args.s, args.ds, args.mb, cap)

    if args.out_sbs:
        img = np.concatenate([orig_img, img], axis=1)

    return img

def runVid(pipe, f, args):
    cap = args.prompt if args.prompt != "" else None
    pipe.inpaint_video(f, args.s, args.ds, args.mb, cap)

if __name__=="__main__":
    logger = logging.getLogger()
    args = parse_args()
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    logger.info("Loading Pipeline...")
    #pipeline_class = ap.PipelineClasses([ap.LABEL_PERSON, ap.LABEL_CAR, ap.LABEL_TRUCK, ap.LABEL_BUS])
    
    pipe = PIPELINE_MAPPING.get(args.pipeline_method, PIPELINE_MAPPING['default'])
    if pipe:
        pipe = pipe()
        if isinstance(pipe, ap.Automatic1111Pipeline):
            pipe.url= "http://sim-melian.fzi.de:7860"
        
    if not pipe:
        from utils import comfy_pipeline as cp

        version = DIFFUSION_VERSIONS[args.version]
        if not checkVersion(version):
            sys.exit(0)
        
        cp.apply_args(args)
        mbs = 0 if args.disable_min_scale else 64
        pipe = cp.ComfyUIPipeline(ckpt_name=version[0], controlnet_name=version[1], size=version[2], use_control_net=not args.no_pose, verbose=args.verbose, clip_enabled = args.enable_clip, min_box_size=mbs)


    print(f"Loaded {pipe}")
    if not pipe.checkAvailable():
        sys.exit(0)

    pipe.apply_settings(args.sdxl, args.ref)


    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if args.video != "":
        runVid(pipe, args.video, args)
    elif args.file != "":
        img = runImg(pipe, args.file, args)
        
        filename = os.path.basename(args.file)
        cv2.imwrite(f"{args.output}/{filename}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        data = fadm_io.find_all_images_in_directory(args.input)
        #image_file_to_output_path = fadm_io.mirror_data_folder_structure(args.output, data)

        interval = args.interval.split(",")
        if interval[1] != "0":
            data = data[int(interval[0]):int(interval[1])]

        for f in tqdm(data, desc="Anonymizing Images", position=0, leave=True):
            img = runImg(pipe, f, args)
            #fadm_io.save_img(img, image_file_to_output_path[f], args.output)
            
            filename = os.path.basename(f)
            cv2.imwrite(f"{args.output}/{filename}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        

    


