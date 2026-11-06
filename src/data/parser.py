"""
Parser module for Stanford Cars MATLAB metadata and annotations.
Extracts hierarchical metadata (Make, Model, Year, Body Type) and bounding boxes.
"""

import json
import os
from typing import Any

import scipy.io as sio
from PIL import Image

from ..utils.box_utils import box_to_yolo, matlab_to_pixel_box

# Known multi-word vehicle manufacturer names in Stanford Cars
KNOWN_MULTIWORD_MAKES = [
    "AM General",
    "Aston Martin",
    "Land Rover",
]

# Canonical body types present in Stanford Cars nomenclature
KNOWN_BODY_TYPES = [
    "SuperCab",
    "Regular Cab",
    "Extended Cab",
    "Crew Cab",
    "Quad Cab",
    "Club Cab",
    "Convertible",
    "Coupe",
    "Sedan",
    "Hatchback",
    "SUV",
    "Minivan",
    "Van",
    "Wagon",
]

# Body type overrides for trim/performance designations where body style is implicit
BODY_TYPE_OVERRIDES: dict[str, str] = {
    "Acura TL Type-S 2008": "Sedan",
    "Acura Integra Type R 2001": "Coupe",
    "Buick Regal GS 2012": "Sedan",
    "Chevrolet Corvette ZR1 2012": "Coupe",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007": "Coupe",
    "Chevrolet HHR SS 2010": "SUV",
    "Chevrolet Cobalt SS 2010": "Coupe",
    "Chevrolet TrailBlazer SS 2009": "SUV",
    "Chrysler 300 SRT-8 2010": "Sedan",
    "Dodge Challenger SRT8 2011": "Coupe",
    "Dodge Charger SRT-8 2009": "Sedan",
    "FIAT 500 Abarth 2012": "Hatchback",
    "Jaguar XK XKR 2012": "Coupe",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012": "Coupe",
}


def parse_cars_meta(
    mat_path: str, output_json_path: str | None = None
) -> dict[str, Any]:
    """
    Parse cars_meta.mat into structured metadata containing make, model, year, body_type
    for all 196 classes.

    Args:
        mat_path: Path to cars_meta.mat.
        output_json_path: Optional path to save classes.json.

    Returns:
        Dictionary containing classes list, maps (make_to_id, year_to_id, etc.).
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"cars_meta.mat not found at {mat_path}")

    mat = sio.loadmat(mat_path)
    raw_class_names = [str(c[0]) for c in mat["class_names"][0]]

    # First pass: extract makes, years, body_types, models to create sorted vocabularies
    parsed_items = []
    unique_makes = set()
    unique_years = set()
    unique_body_types = set()
    unique_models = set()

    for idx, raw_name in enumerate(raw_class_names):
        tokens = raw_name.split(" ")
        year = int(tokens[-1])
        unique_years.add(year)

        # Extract make
        make = None
        rem = ""
        for m in KNOWN_MULTIWORD_MAKES:
            if raw_name.startswith(m + " "):
                make = m
                rem = raw_name[len(m) + 1 : -(len(str(year)) + 1)].strip()
                break
        if not make:
            make = tokens[0]
            rem = " ".join(tokens[1:-1]).strip()
        unique_makes.add(make)

        # Extract body type
        body_type = BODY_TYPE_OVERRIDES.get(raw_name)
        if not body_type:
            for bt in KNOWN_BODY_TYPES:
                if bt in rem:
                    body_type = bt
                    break
        if not body_type:
            body_type = "Unknown"
        unique_body_types.add(body_type)

        # Model is rem (e.g. '3 Series Sedan' or 'Corvette ZR1')
        model = rem.strip()
        unique_models.add(model)

        # Clean model without body type
        clean_model = model
        if body_type in model:
            clean_model = model.replace(body_type, "").strip()
        clean_model = " ".join(clean_model.split())
        if not clean_model:
            clean_model = model

        parsed_items.append(
            {
                "class_id": idx,
                "class_id_0": idx,
                "class_id_1": idx + 1,
                "raw_name": raw_name,
                "make": make,
                "model": model,
                "clean_model": clean_model,
                "year": year,
                "body_type": body_type,
            }
        )

    # Sort vocabularies deterministically
    sorted_makes = sorted(unique_makes)
    sorted_years = sorted(unique_years)
    sorted_body_types = sorted(unique_body_types)
    sorted_models = sorted(unique_models)

    make_to_id = {m: i for i, m in enumerate(sorted_makes)}
    year_to_id = {y: i for i, y in enumerate(sorted_years)}
    body_type_to_id = {b: i for i, b in enumerate(sorted_body_types)}
    model_to_id = {m: i for i, m in enumerate(sorted_models)}

    # Attach IDs to parsed items
    for item in parsed_items:
        item["make_id"] = make_to_id[item["make"]]
        item["year_id"] = year_to_id[item["year"]]
        item["body_type_id"] = body_type_to_id[item["body_type"]]
        item["model_id"] = model_to_id[item["model"]]

    meta_result = {
        "num_classes": len(parsed_items),
        "classes": parsed_items,
        "makes": sorted_makes,
        "years": sorted_years,
        "body_types": sorted_body_types,
        "models": sorted_models,
        "make_to_id": make_to_id,
        "year_to_id": {str(y): i for y, i in year_to_id.items()},
        "body_type_to_id": body_type_to_id,
        "model_to_id": model_to_id,
    }

    if output_json_path:
        out_dir = os.path.dirname(output_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_result, f, indent=2)

    return meta_result


def parse_annotations(
    annos_mat_path: str,
    img_dir: str,
    classes_meta: dict[str, Any] | None = None,
    is_test: bool = False,
) -> list[dict[str, Any]]:
    """
    Parse cars_train_annos.mat or cars_test_annos.mat into structured records.
    Converts 1-based MATLAB bounding boxes to 0-based pixel boxes and YOLO coordinates.

    Args:
        annos_mat_path: Path to annotations MAT file.
        img_dir: Directory containing the corresponding image files.
        classes_meta: Parsed metadata dictionary from parse_cars_meta.
        is_test: If True, indicates test annotations where class labels are omitted.

    Returns:
        List of annotation dictionaries.
    """
    if not os.path.exists(annos_mat_path):
        raise FileNotFoundError(f"Annotations MAT file not found: {annos_mat_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    mat = sio.loadmat(annos_mat_path)
    raw_annos = mat["annotations"][0]

    class_lookup = {}
    if classes_meta and "classes" in classes_meta:
        for c in classes_meta["classes"]:
            class_lookup[c["class_id_1"]] = c

    records = []
    for i, anno in enumerate(raw_annos):
        fname = str(anno["fname"][0])
        fpath = os.path.join(img_dir, fname)

        bx1 = int(anno["bbox_x1"].item())
        by1 = int(anno["bbox_y1"].item())
        bx2 = int(anno["bbox_x2"].item())
        by2 = int(anno["bbox_y2"].item())

        # Read actual image dimensions
        with Image.open(fpath) as img:
            img_w, img_h = img.size
            mode = img.mode

        # 0-based inclusive coordinates [x1, y1, x2, y2]
        x1_inc, y1_inc, x2_inc, y2_inc = matlab_to_pixel_box(
            bx1, by1, bx2, by2, img_w, img_h, inclusive=True
        )

        # 0-based half-open coordinates for PIL crop [left, top, right, bottom]
        crop_box = matlab_to_pixel_box(
            bx1, by1, bx2, by2, img_w, img_h, inclusive=False
        )

        # YOLO normalized coordinates [xc, yc, w, h] in [0, 1]
        yolo_box = box_to_yolo(
            x1_inc, y1_inc, x2_inc, y2_inc, img_w, img_h, inclusive=True
        )

        record: dict[str, Any] = {
            "index": i,
            "filename": fname,
            "file_path": os.path.abspath(fpath),
            "relative_path": os.path.relpath(fpath, os.getcwd()),
            "img_width": img_w,
            "img_height": img_h,
            "is_grayscale": (mode != "RGB"),
            "bbox_matlab": [bx1, by1, bx2, by2],
            "bbox_pixel_inc": [x1_inc, y1_inc, x2_inc, y2_inc],
            "bbox_crop": list(crop_box),
            "bbox_yolo": list(yolo_box),
            "box_width": x2_inc - x1_inc + 1,
            "box_height": y2_inc - y1_inc + 1,
        }

        if not is_test and "class" in anno.dtype.names:
            class_id_1 = int(anno["class"].item())
            class_id_0 = class_id_1 - 1
            record["class_id"] = class_id_0
            record["class_id_0"] = class_id_0
            record["class_id_1"] = class_id_1

            if class_lookup and class_id_1 in class_lookup:
                cinfo = class_lookup[class_id_1]
                record["class_name"] = cinfo["raw_name"]
                record["make"] = cinfo["make"]
                record["model"] = cinfo["model"]
                record["clean_model"] = cinfo["clean_model"]
                record["year"] = cinfo["year"]
                record["body_type"] = cinfo["body_type"]
                record["make_id"] = cinfo["make_id"]
                record["year_id"] = cinfo["year_id"]
                record["body_type_id"] = cinfo["body_type_id"]
                record["model_id"] = cinfo["model_id"]

        records.append(record)

    return records
