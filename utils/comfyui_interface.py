import os
import random
import sys
import torch
import logging
import cv2
import numpy as np
from typing import Sequence, Mapping, Any, Union

COMFY_UI_PATH = "./external/ComfyUI"

def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    path = os.path.abspath(COMFY_UI_PATH)
    if os.path.isdir(path):
        sys.path.append(path)
        logging.info(f"'{path}' added to sys.path")
    else:
        logging.error(f"'{path}' is not a directory")
        sys.exit(1)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


add_comfyui_directory_to_sys_path()

from nodes import EmptyLatentImage, InpaintModelConditioning, CheckpointLoaderSimple, CLIPTextEncode, VAEEncode, VAEEncodeForInpaint, VAEDecode, SetLatentNoiseMask, KSampler, VAELoader, ControlNetApplyAdvanced, ControlNetLoader, NODE_CLASS_MAPPINGS

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()



from custom_nodes.comfyui_controlnet_aux.utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import json

class CustomOpenPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand = (["enable", "disable"], {"default": "enable"}),
            detect_body = (["enable", "disable"], {"default": "enable"}),
            detect_face = (["enable", "disable"], {"default": "enable"})
        )
        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def __init__(self):
        from controlnet_aux.open_pose import OpenposeDetector
        self.model = OpenposeDetector.from_pretrained().to(model_management.get_torch_device())        

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, **kwargs):

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"

        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = self.model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img
        
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)

        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }



class ComfyUIInterface:
    def __init__(self, ckpt_name = "v2-1_768-ema-pruned.safetensors", control_net_name="control_v11p_sd21_openposev2.safetensors", size = (768, 768), use_pose = False) -> None:
        import_custom_nodes()

        self.cliptextencode = CLIPTextEncode()
        self.vaeencode = VAEEncode()
        self.vaeencode_inpaint = VAEEncodeForInpaint()
        self.inpaint_model_cond = InpaintModelConditioning()
        self.vaedecode = VAEDecode()
        self.setlatentnoisemask = SetLatentNoiseMask()
        self.ksampleradvanced = KSampler()
        self.use_pose = use_pose

        if use_pose:
            self.control_net_apply = ControlNetApplyAdvanced()
            #self.openposepreprocessor = NODE_CLASS_MAPPINGS["OpenposePreprocessor"]()
            self.openposepreprocessor = CustomOpenPose_Preprocessor()

            controlnetloader = ControlNetLoader()
            self.controlnet = get_value_at_index(controlnetloader.load_controlnet(
                control_net_name=control_net_name
            ), 0)

        self.load_ckpt(ckpt_name, size)

    @torch.no_grad()
    def load_ckpt(self, name="v2-1_768-ema-pruned.safetensors", size = (768, 768)):
        checkpointloadersimple = CheckpointLoaderSimple()
        self.ckpt_loader = checkpointloadersimple.load_checkpoint(
            ckpt_name=name
        )
        logging.info(f"Using SD-Model {name}")

        #self.vaeloader = VAELoader().load_vae("sd_2_1.safetensors")

        self.img_size = size

    def get_model(self):
        return get_value_at_index(self.ckpt_loader, 0)
    def get_clip(self):
        return get_value_at_index(self.ckpt_loader, 1)
    def get_vae(self):
        return get_value_at_index(self.ckpt_loader, 2)
        #return get_value_at_index(self.vaeloader, 0)
    
    def encode_clip(self, text):
        return self.cliptextencode.encode(text=text, clip=self.get_clip())


    ############# TASKS ##############
    @torch.no_grad()
    def run_inpainting(self, img, mask, pos_prompt: str, neg_prompt: str, steps = 20, denoising_strength = 0.6):
        img = torch.tensor(img)[None, ...].to(torch.bfloat16)
        mask = torch.tensor(mask)[None, ...].to(torch.bfloat16)


        vae = self.get_vae()

                
        clip_pos = get_value_at_index(self.encode_clip(pos_prompt), 0)
        clip_neg = get_value_at_index(self.encode_clip(neg_prompt), 0)
        
        if denoising_strength == 1.0:
            #img_enc = get_value_at_index(EmptyLatentImage().generate(
            #    width=img.shape[2], height=img.shape[1], batch_size=1
            #), 0)
            img_enc = get_value_at_index(self.vaeencode_inpaint.encode(
                grow_mask_by=0,
                pixels=img,
                mask=mask,
                vae=vae
            ), 0)
        else:
            #img_enc = get_value_at_index(self.vaeencode.encode(pixels=img, vae=vae), 0)

            #img_enc = get_value_at_index(self.setlatentnoisemask.set_mask(samples=img_enc, mask=mask), 0)
            cond = self.inpaint_model_cond.encode(
                positive=clip_pos,
                negative=clip_neg,
                pixels=img,
                vae=vae,
                mask=mask
            )
            clip_pos = get_value_at_index(cond, 0)
            clip_neg = get_value_at_index(cond, 1)
            img_enc = get_value_at_index(cond, 2)
    
        if self.use_pose:
            pose = self.openposepreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                image=img.to(torch.float16),
            )
            cn_apply = self.control_net_apply.apply_controlnet(
                strength=1,
                start_percent=0,
                end_percent=1,
                positive=clip_pos,
                negative=clip_neg,
                control_net=self.controlnet,
                image=get_value_at_index(pose, 0),
            )
            clip_pos = get_value_at_index(cn_apply, 0)
            clip_neg = get_value_at_index(cn_apply, 1)


        ksampler = self.ksampleradvanced.sample(
                #add_noise="enable",
                seed=random.randint(1, 2**64),
                steps=int(steps * denoising_strength),
                cfg=7,
                sampler_name="euler",#"dpmpp_sde_gpu",
                scheduler="normal",#"karras",
                #start_at_step=int(steps * (1.0 - denoising_strength)),
                #end_at_step=10000,
                #return_with_leftover_noise="disable",
                denoise=denoising_strength,
                model=self.get_model(),
                positive=clip_pos,
                negative=clip_neg,
                latent_image=img_enc,
            )
        
        result = get_value_at_index(ksampler, 0)
        img_out = get_value_at_index(self.vaedecode.decode(samples=result, vae=vae), 0).cpu().numpy()

        return img_out[0]
    

if __name__ == "__main__":
    interface = ComfyUIInterface()

    img = cv2.imread("../input/3950077043_3b395b221e_o.jpg")[..., ::-1] / 255.0
    img = cv2.resize(img, (768, 768))

    mask = np.ones(img.shape[:2])

    img = interface.run_inpainting(img, mask, "A beautiful cat standing in front of a sea, cinematic, beautiful, blue sky, daylight", "ugly, deformed",
                                   denoising_strength=1.0)
    
    cv2.imwrite("../output/test.jpg", (img[..., ::-1] * 255).astype(np.uint8))
        
