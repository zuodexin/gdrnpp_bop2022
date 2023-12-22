import mmcv
import sys
from tqdm import tqdm
import argparse
import json


def save_json(path, content, sort=False):
    """Saves the provided content to a JSON file.

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    """
    with open(path, "w") as f:
        if isinstance(content, dict):
            f.write("{\n")
            if sort:
                content_sorted = sorted(content.items(), key=lambda x: x[0])
            else:
                content_sorted = content.items()
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}")

        elif isinstance(content, list):
            f.write("[\n")
            for elem_id, elem in enumerate(content):
                f.write("  {}".format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]")

        else:
            json.dump(content, f, sort_keys=True)


def main(args):
    path = args.coco_bop_input
    ds = mmcv.load(path)

    outs = {}
    for d in tqdm(ds):
        scene_id = d["scene_id"]
        image_id = d["image_id"]
        scene_im_id = f"{scene_id}/{image_id}"

        obj_id = d["category_id"]
        score = d["score"]

        bbox = d["bbox"]
        time = d["time"]

        cur_dict = {
            "bbox_est": bbox,
            "obj_id": obj_id,
            "score": score,
            "time": time,
        }

        if scene_im_id in outs.keys():
            outs[scene_im_id].append(cur_dict)
        else:
            outs[scene_im_id] = [cur_dict]
    save_json(
        args.output,
        outs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_bop_input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    main(args)
