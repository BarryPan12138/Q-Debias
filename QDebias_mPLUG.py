from PIL import Image
import torch
import json, random
from tqdm import tqdm
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
from iqa_utils import load_config, softmax
from imagecorruptions import corrupt, get_corruption_names
import argparse

from transformers import AutoTokenizer, AutoProcessor, AutoModel
from decord import VideoReader, cpu    # pip install decord
model_path = "../mPLUG-Owl/mPLUG-Owl3/iic//mPLUG-Owl3-7B-240728"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.makedirs("IQA_outputs/Q-Debias/mix/", exist_ok=True)
# device = "cuda:2" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2").eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = model.init_processor(tokenizer)

config = load_config("./config.yaml")

parser = argparse.ArgumentParser(description="Script with YAML and Argparse")

parser.add_argument("--image_paths", nargs="+", default=config["image_paths"], help="List of image paths")
parser.add_argument("--jsons", nargs="+", default=config["jsons"], help="List of JSON files")
parser.add_argument("--corruption_names", nargs="+", default=config["corruption_name"], help="Type of corruption")
parser.add_argument("--severitys", nargs="+", default=config["severity"], help="Severity level")

args = parser.parse_args()

image_paths = args.image_paths
jsons = args.jsons

test_len = 100
dst_names = args.corruption_names
severitys = args.severitys
dst_num = len(dst_names)

def softmax_lst(lst):
    exp_values = np.exp(lst) 
    return exp_values / np.sum(exp_values)

with torch.set_grad_enabled(False):
    # for severity in severitys:
    for image_path, input_json in zip(image_paths, jsons):
        with open(input_json) as f:
            all_data = json.load(f) 
        
        gts = [float(di["gt_score"]) for di in all_data]
        prs = []

        for llddata in tqdm(all_data):
            name = llddata["img_path"]

            raw_image = Image.open(image_path + name).convert("RGB")

            yes_probs = []
            good_probs = []

            for i in range(dst_num):

                narry_image = np.asarray(raw_image)
                corrupted = corrupt(narry_image, corruption_name=dst_names[i], severity=severitys[i])
                dst_image = Image.fromarray(corrupted)

                messages = [
                {"role": "user", "content": "<|image|><|image|> Do these two images describe the same object? Yes or no"},
                {"role": "assistant", "content": ""}
                ]

                inputs = processor(messages, images=[dst_image, raw_image], videos=None)

                inputs.to(model.device)
                output_logits = model(**inputs).logits[0, -1]
                lyes, lno = output_logits[9454].item(), output_logits[2753].item()
                yes_prob = softmax(lyes, lno)
                yes_probs.append(yes_prob)
            
                messages = [
                    {"role": "user", "content": "<|image|><|image|> The visual quality of the first image is poor. How about the visual quality of the second image. Good or poor?"},
                    {"role": "assistant", "content": "The quality of the image is "}
                ]

                inputs = processor(messages, images=[dst_image, raw_image], videos=None)

                inputs.to(model.device)
                output_logits = model(**inputs).logits[0, -1]
                lgood, lpoor = output_logits[15216].item(), output_logits[84103].item()
                lgood_prob = softmax(lgood, lpoor)
                good_probs.append(lgood_prob)

            yess = softmax_lst(yes_probs)
            q_pred = np.dot(yess, good_probs)
            prs.append(q_pred)


        
        # with open(f"IQA_outputs/Q-Debias/mix/{(input_json.split('/')[-1]).split('.')[0]}_{corruption_name}.json", "a") as wf:
        #     json.dump(all_data, wf)

        srcc, _ = spearmanr(prs, gts)
        plcc, _ = pearsonr(prs, gts)
        print(f'dataset: {input_json}')  
        print("SRCC:", srcc)
        print("PLCC:", plcc)

        


