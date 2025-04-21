#!/usr/bin/env python3

import logging
import os
import sys
import tempfile

import torch
import cv2
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Model and file settings
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
MODEL_DIR = "models/llama-joycaption-alpha-two-hf-llava"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Download model if missing
def ensure_model():
    if not os.path.isdir(MODEL_DIR):
        logging.info("Model missing, downloading %s...", MODEL_NAME)
        os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
        snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_DIR)
        logging.info("Download complete.")

# Initialize
ensure_model()
logging.info("Loading processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
logging.info("Model ready.")


def describe_image(path: str, return_caption=False):
    """
    Caption a single image and optionally return the text.
    """
    if not os.path.isfile(path):
        logging.error("Image not found: %s", path)
        return None

    img = Image.open(path).convert("RGB")
    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": (
            "Write a long descriptive caption for this image in a formal tone. "
            "Include details about lighting and camera angle. "
            "Do not mention any visible text."
        )},
    ]
    prompt = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt").to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            use_cache=True
        )[0]

    caption_ids = outputs[inputs["input_ids"].shape[1]:]
    caption = processor.tokenizer.decode(caption_ids, skip_special_tokens=True).strip()

    out_txt = os.path.splitext(path)[0] + ".txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(caption)
    logging.info("Caption saved: %s", out_txt)

    return caption if return_caption else None


def process_folder(folder: str):
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    if not files:
        logging.warning("No images found in %s", folder)
        return
    logging.info("Processing %d images in %s", len(files), folder)
    for img in files:
        describe_image(img)


def process_video(path: str, interval_sec: float = 1.0):
    if not os.path.isfile(path):
        logging.error("Video not found: %s", path)
        return

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = int(fps * interval_sec)

    captions = []
    idx = 0
    tmp = tempfile.mkdtemp(prefix="frames_")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = os.path.join(tmp, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            cap_txt = describe_image(fname, return_caption=True)
            if cap_txt:
                captions.append(cap_txt)
        idx += 1
    cap.release()
    logging.info("Captured %d frames", len(captions))

    summary = summarize(captions)
    out_sum = os.path.splitext(path)[0] + "_narrative.txt"
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write(summary)
    logging.info("Narrative saved: %s", out_sum)


def summarize(captions: list[str]) -> str:
    numbered = "\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])
    user_content = (
        "Combine these captions into a single, smoothly flowing narrative in a formal tone:\n" + numbered
    )
    convo = [
        {"role": "system", "content": "You write coherent narratives."},
        {"role": "user", "content": user_content},
    ]
    prompt = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    # Use a real RGB dummy image
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    inputs = processor(
        text=[prompt],
        images=[dummy_image],
        return_tensors="pt"
    ).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )[0]

    narrative_ids = output_ids[inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(narrative_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python video2text_pipeline.py <file_or_folder>")
        sys.exit(1)

    src = sys.argv[1]
    ext = os.path.splitext(src)[1].lower()
    if os.path.isdir(src):
        process_folder(src)
    elif ext in VIDEO_EXTS:
        process_video(src)
    else:
        describe_image(src)
