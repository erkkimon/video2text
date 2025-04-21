# Video2Text Pipeline

This tool processes videos and images to generate detailed frame-level captions and cohesive scene narratives using a local visual-language model and an optional local Ollama-based LLM.

---

## ✨ Features

- Frame-by-frame captioning with high-detail spatial descriptions.
- Context-aware descriptions with comparison to previous frame.
- Generates a flowing narrative from individual frame captions.
- Automatically injects a user-defined **trigger** term into the narrative.
- Supports trigger **description** to give semantic context to both VLM and LLM.
- Supports **Ollama** for local LLM generation (e.g., Gemma, QWQ).
- Trigger word is prepended to the summary automatically.
- VLM and LLM instructed to use the term “frame” (not “image” or “photo”).
- Optional: removes `<think>` or `<thinking>` tags from output.
- Outputs `.txt` files for each frame and final `.txt` narrative for video.

---

## 🛠 Setup

1. Clone the repo:

    ```bash
    git clone https://github.com/erkkimon/video2text
    cd video2text
    ```

2. Create and activate a virtual environment:

    - Linux/macOS:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - Windows:
      ```cmd
      python -m venv venv
      venv\Scripts\activate
      ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) Install and configure Ollama:

    - [Install Ollama](https://ollama.com/)
    - Pull your preferred model:
      ```bash
      ollama pull huihui_ai/gemma3-abliterated:4b
      ```

---

## 🚀 Usage

### Process a video:
```bash
python video2text-joycaption-2.py \
  --trigger "fifth_wheel" \
  --trigger-description "a trailer designed for living which can be pulled by truck" \
  --model "huihui_ai/gemma3-abliterated:4b" \
  input/fifh_wheel.mp4
```

### Caption a single image:
```bash
python video2text-joycaption-2.py path/to/image.jpg
```

### Caption all images in a folder:
```bash
python video2text-joycaption-2.py path/to/image_folder/
```

---

## 🔍 Parameters

- `--trigger`  
  Required word/phrase that must be included in the final narrative.

- `--trigger-description`  
  Optional human-readable description of the trigger, e.g.,  
  `"a trailer designed for living which can be pulled by truck"`.

- `--model`  
  Ollama model to use for summarization (default is `huihui_ai/qwq-abliterated:32b`).

---

## 📁 Output

- A `.txt` file is created for every extracted video frame or individual image.
- A final `.txt` file is generated for video inputs containing the summary narrative.
- The narrative begins with the trigger followed by a period and the scene description.
- Example:
  ```
  fifth_wheel. A large black pickup truck tows a fifth-wheel trailer down a muddy forest path...
  ```

---

## 🧪 Future Ideas

- 📦 Add bounding box detection and include dimensions and positions in the captions.
- 🧠 Improve frame comparison logic to describe object movement and change over time.
- 🎞 Enhance temporal consistency and introduce actor tracking (e.g. "the same truck...").
- ⚙️ Add parallel processing of frame captioning for performance.

---

Let us know how you'd like to customize it further!
