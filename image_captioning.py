import os
import ast
from tqdm import tqdm
from io import BytesIO

import json
from itertools import islice

import requests
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image

disable_torch_init()

MODEL = "4bit/llava-v1.5-13b-3GB"
model_name = get_model_name_from_path(MODEL)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True)

def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def process_image(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)


CONV_MODE = "llava_v0"

def create_prompt(prompt: str):
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv

def ask_image(image: Image, prompt: str):
    image_tensor = process_image(image)
    prompt, conv = create_prompt(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


dress_types = ['shirt', 'dress', 'toptee']
k = 15

def batch_iterator(iterator, batch_size, start=0):
    iterator = iter(iterator)
    iterator = islice(iterator, start, None)  # Skip the first 'start' items
    for first in iterator:
        yield list(islice([first] + list(iterator), batch_size))

def get_fiq_it():
  image_path_format = 'drive/My Drive/images/{}.jpg'
  type2itlist = dict()
  for dress_type in dress_types:
    with open(f'drive/My Drive/splits/split.{dress_type}.val.json') as f:
      image_names = json.load(f)
      print(len(image_names))

    it_list = []
    for image_name in tqdm(image_names, desc=f"Processing {dress_type} images"):
      it_list.append({"image_id": image_name, "caption": "", "image_path": image_path_format.format(image_name)})

    type2itlist[dress_type] = it_list
  return type2itlist

type2itlist = get_fiq_it()
all_it_list = []

start_index = 1500

for dress_type in dress_types:
    it_list = type2itlist[dress_type]
    prompt = f"Describe the {dress_type} in {k} words"
    for batch in batch_iterator(it_list, 500, start=start_index):
        for it in tqdm(batch):
            try:
                image = load_image(it["image_path"])
                it["caption"] = ask_image(image, prompt)
            except FileNotFoundError:
                continue
        all_it_list.extend(batch)
        try:
          with open(f"drive/My Drive/all/{dress_type}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(batch, ensure_ascii=False))
        except:
            print("Error")