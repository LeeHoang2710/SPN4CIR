import json

with open("dress.json", "r") as file:
    data = json.load(file)

image_caption_dict = {item["image_id"]: item["caption"] for item in data}

with open("dress_test.json", "r") as file:
    original = json.load(file)

for item in original:
    target_id = item["target"]
    if target_id in image_caption_dict:
        item["generated"] = image_caption_dict[target_id]

with open("dress_test_updated.json", "w") as file:
    json.dump(original, file)

# with open("new.json", "w") as file:
#     json.dump(image_caption_dict, file)