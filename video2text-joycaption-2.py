#!/usr/bin/env python3

import logging
import os
import sys
import tempfile
import torch
import cv2
import requests
import argparse
import re
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

args = None

def load_model():
    if not os.path.isdir(MODEL_DIR):
        logging.info("Model missing, downloading %s...", MODEL_NAME)
        os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
        snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_DIR)
        logging.info("Download complete.")
    logging.info("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    logging.info("Model ready.")
    return processor, model


processor, model = load_model()

def generate_caption(image: Image.Image, prev_description: str = None) -> str:
    description_line = f" The word '{args.trigger}' must be included accurately in your description."
    if args and args.trigger_description:
        description_line += f" The word '{args.trigger}' refers to {args.trigger_description}."

    intro = (
        "You are a helpful image captioner that helps reconstruct video scenes."
    )

    user_prompt = (
        "This is a frame from a video. Describe it in a formal and highly detailed way. "
        "Focus on describing people, objects, animals, and background elements. "
        "Include spatial relationships and relative positions (e.g., left, right, near, far, center). "
        "Include posture, orientation, or movement if possible. "
        "Do not mention any visible text. Do not speculate about emotions or narrative. "
        "Only describe what is visible in the frame."
    )

    if prev_description:
        user_prompt = (
            f"The previous frame was described as follows: '{prev_description}'.\n"
            f"Now, {user_prompt} Focus on what has changed since the last frame."
        )

    if args and args.trigger:
        user_prompt += description_line

    convo = [
        {"role": "system", "content": intro},
        {"role": "user", "content": user_prompt},
    ]

    prompt = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            use_cache=True
        )[0]

    caption_ids = outputs[inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(caption_ids, skip_special_tokens=True).strip().replace("\n", " ")


def describe_image(path: str, return_caption=False):
    if not os.path.isfile(path):
        logging.error("Image not found: %s", path)
        return None

    img = Image.open(path).convert("RGB")
    caption = generate_caption(img)
    out_txt = os.path.splitext(path)[0] + ".txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(caption)
    logging.info("Caption saved: %s", out_txt)
    print(f"[Frame Caption] {caption}\n")
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


def summarize(captions: list[str]) -> str:
    combined = "\n".join(captions)
    summary_prompt = (
        "You are a video scene summarizer. Below is a chronological list of frame-level visual descriptions from a short video. "
        "Your task is to write a flowing narrative that captures what is happening in the video over time. "
        "Focus on visual changes, movement, positions of people and objects, and continuity. "
        "Do not list steps or number frames. Do not mention any image or caption. "
        "Do not format the result as markdown. Do not say 'image description'. "
        "Avoid speculating about intentions or emotions. Do not include any metadata or formatting. "
        "Write as if describing the scene to someone who cannot see it."
    )
    if args and args.trigger:
        summary_prompt += f" The phrase '{args.trigger}' must be included."
        if args.trigger_description:
            summary_prompt += f" The phrase '{args.trigger}' refers to {args.trigger_description}."

    prompt = summary_prompt + "\n\n" + combined

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": args.model,
        "prompt": prompt,
        "stream": False
    })

    if not response.ok:
        logging.error("LLM response error: %s", response.status_code)
        return "<No response from model>"

    result_raw = response.json().get("response", "").strip()
    print(f"\n[Raw LLM Output]\n{result_raw}\n")

    # Strip thinking tags if present
    result = re.sub(r"<(thinking|think)>.*?</\\1>", "", result_raw, flags=re.IGNORECASE | re.DOTALL).strip().replace("\n", " ")

    if args.trigger:
        result = f"{args.trigger}. {result}"

    print(f"\n[Final Narrative]\n{result}\n")
    return result


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

    prev_caption = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = os.path.join(tmp, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            img = Image.open(fname).convert("RGB")
            caption = generate_caption(img, prev_caption)
            captions.append(caption)
            prev_caption = caption
            with open(os.path.splitext(fname)[0] + ".txt", "w", encoding="utf-8") as f:
                f.write(caption)
            logging.info("Caption saved: %s", fname)
            print(f"[Frame Caption] {caption}\n")
        idx += 1
    cap.release()
    logging.info("Captured %d frames", len(captions))

    summary = summarize(captions)
    out_sum = os.path.splitext(path)[0] + ".txt"
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write(summary)
    logging.info("Narrative saved: %s", out_sum)


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image, video file or folder")
    parser.add_argument("--trigger", help="Phrase that must be included in the narrative", default=None)
    parser.add_argument("--trigger-description", help="Explanation of what the trigger means", default=None)
    parser.add_argument("--model", help="Ollama model to use", default="huihui_ai/qwq-abliterated:32b")
    args = parser.parse_args()

    src = args.input
    ext = os.path.splitext(src)[1].lower()
    if os.path.isdir(src):
        process_folder(src)
    elif ext in VIDEO_EXTS:
        process_video(src)
    else:
        describe_image(src)


if __name__ == "__main__":
    main()
