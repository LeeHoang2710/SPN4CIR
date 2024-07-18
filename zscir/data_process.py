import argparse
import json
import os
import random
from typing import List

import clip
import torch
from tqdm import tqdm
from prompts import prompts_reference, prompts_both, prompts_target
from retrieval import ImageDataset, extract_image_features


def get_fiq_it():
    image_path_fmt = '/root_path/fashionIQ_dataset/images/{}.png'
    type2itlist = dict()
    for dress_type in ['dress', 'shirt', 'toptee']:
        with open(
                f'/root_path/fashionIQ_dataset/image_splits/split.{dress_type}.train.json') as f:
            image_names = json.load(f)
            print(len(image_names))
        it_list = []
        for image_name in tqdm(image_names):
            it_list.append({"image_id": image_name, "caption": "", "image_path": image_path_fmt.format(image_name)})
        # with open(f"fashioniq_{dress_type}_it.json", 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(it_list, ensure_ascii=False))
        type2itlist[dress_type] = it_list
    return type2itlist


def get_cirr_it():
    image_path_file = "/root_path/cirr_dataset/cirr/image_splits/split.rc2.train.json"
    image_path_fmt = '/root_path/cirr_dataset/{}'
    it_list = []
    with open(image_path_file) as f:
        name2path = json.loads(f.read())
        for image_name in name2path:
            it_list.append(
                {"image_id": image_name, "caption": "", "image_path": image_path_fmt.format(name2path[image_name])})
    return it_list


def get_cc_it(cc_id):
    data_ls = []
    path = f"/root_path/pretrain_data/translated_4M/cc3m-mm-data-all/part_{cc_id}.data"
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == '':
                continue
            data = json.loads(line.strip())
            data_ls.append({"image_id": data['url'], "image_path": data['image'], "caption": data['caption']['en']})
    print(len(data_ls))
    return data_ls


def get_coco_it():
    json_file = '/root_path/CCLM/data/finetune/mscoco/en.train.json'
    with open(json_file) as f:
        it_list = json.loads(f.read())
    image2captions = dict()
    for coco_it in tqdm(it_list):
        image = coco_it['image']
        if os.path.exists(image) and os.path.getsize(image) > 0:
            if image not in image2captions:
                image2captions[image] = [coco_it['caption']]
            else:
                image2captions[image].append(coco_it['caption'])
        else:
            print("error")
    coco_it_list = []
    for image in image2captions:
        coco_it_list.append({"image_path": image, "caption": random.choice(image2captions[image])})
    with open("/root_path/coco_dataset/coco_it.json", 'w') as f:
        f.write(json.dumps(coco_it_list, ensure_ascii=False))


def process_coco():
    json_file = '/root_path/coco_dataset/coco_it.json'
    if not os.path.exists(json_file):
        get_coco_it()
    with open(json_file) as f:
        coco_it_list = json.loads(f.read())
    triplets = get_triplets(coco_it_list)
    with open(f"coco_dataset/cap.train.{args.method}.json", 'w') as f:
        f.write(json.dumps(triplets, ensure_ascii=False))
    print(len(triplets))


def process_fiq():
    dress_types = ['shirt', 'dress', 'toptee']
    json_file_format = '/root_path/cirdata/fashioniq_{}_it.json'
    for dress_type in dress_types:
        json_file = json_file_format.format(dress_type)
        if not os.path.exists(json_file):
            print(f'{dress_type} caption should be generated by image captioning model')
            continue
        with open(json_file) as f:
            fiq_it_list = json.loads(f.read())
        triplets = get_triplets(fiq_it_list)
        with open(f"fashionIQ_dataset/zs/cap.train.{dress_type}.{args.method}.json", 'w') as f:
            f.write(json.dumps(triplets, ensure_ascii=False))
        print(len(triplets))


def get_triplet(prompt, it1, it2):
    image1 = it1['image_path']
    image2 = it2['image_path']
    caption1 = it1['caption']
    caption2 = it2['caption']
    if isinstance(prompt, str):
        text = prompt.format(caption1, caption2)
    elif isinstance(prompt, List):
        text = []
        for p in prompt:
            text.append(p.format(caption1, caption2))
    triplet = {"reference": image1, "target": image2, "text": text}
    return triplet


def get_triplets(it_list):
    triplets = []
    N = len(it_list)
    random.shuffle(it_list)
    if args.method == 'baseline':
        for i in tqdm(range(0, N, 2)):
            if i + 1 >= N:
                continue
            triplet = get_triplet(prompts_both[0], it_list[i], it_list[i + 1])
            triplets.append(triplet)
    elif args.method == 'multip':
        pass
    elif args.method == 'mostsim':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        query_image_dataset = ImageDataset(it_list, clip_preprocess)
        query_image_features, query_image_paths = extract_image_features(clip_model, query_image_dataset, device)
        target_image_features, target_image_paths = query_image_features, query_image_paths
        distances = query_image_features @ target_image_features.T  # bs*dim @ dim*num = bs*num
        sorted_indices = torch.argsort(distances, dim=-1, descending=True).cpu()[:, :args.topk + 1]
        for i, sorted_indice in enumerate(sorted_indices):
            sorted_indice = sorted_indice[sorted_indice != i][:args.topk]  # remove reference
            for j in sorted_indice:
                triplet = get_triplet(prompts_both[0], it_list[i], it_list[j])
                triplets.append(triplet)
    return triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['coco', 'fiq'])
    parser.add_argument("--seed", default=42)
    parser.add_argument("--method", default='baseline', choices=['baseline', 'multip', 'mostsim'])
    parser.add_argument("--topk", default=2)
    args = parser.parse_args()
    if args.dataset == 'coco':
        process_coco()
    elif args.dataset == 'fiq':
        process_fiq()
