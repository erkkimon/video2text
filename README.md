# Video2Text Pipeline

This tool processes videos and images to generate descriptive captions and cohesive narratives using the LLaVA-based `fancyfeast/llama-joycaption-alpha-two-hf-llava` model.

## Features
- Generate formal, detailed captions for images.
- Extract frames from video (one per second) and caption each.
- Combine frame-level captions into a single narrative.

## Setup

1. Clone this repository and enter the directory:
```bash
git clone https://github.com/erkkimon/video2text
cd video2text
```

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the environment:
- **Linux/macOS:**
```bash
source venv/bin/activate
```
- **Windows (CMD):**
```cmd
venv\Scripts\activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

To generate a narrative for a video:
```bash
python video2text-joycaption-2.py input/big-buck-bunny-33s.mp4
```

To caption a single image:
```bash
python video2text-joycaption-2.py path/to/image.jpg
```

To caption all images in a folder:
```bash
python video2text-joycaption-2.py path/to/image_folder/
```

## Output
- Each image frame gets a `.txt` file with its caption.
- Video inputs also generate a single `_narrative.txt` summarizing all frame captions.

---

Let us know how you'd like to customize it further!