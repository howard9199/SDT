import gradio as gr
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

from models.model import SDT_Generator
from utils.util import coords_render

# default values follow the config definitions
NUM_IMGS = 15
MAX_LEN = 120
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model checkpoint if provided
CKPT_PATH = os.environ.get('SDT_CKPT', 'model_zoo/pretrained_model.pth')
model = SDT_Generator().to(DEVICE)
if os.path.exists(CKPT_PATH):
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded weights from {CKPT_PATH}")
else:
    print(f"Warning: checkpoint {CKPT_PATH} not found, using random weights")
model.eval()


def text_to_image(text, size=(64, 64)):
    """Render text to a grayscale PIL image."""
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    # use default font
    font = ImageFont.load_default()
    w, h = draw.textsize(text, font=font)
    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), text, fill=0, font=font)
    return img


def predict(style_img: Image.Image, character: str):
    style_img = style_img.convert('L').resize((64, 64))
    style = np.array(style_img, dtype=np.float32) / 255.0
    style = torch.from_numpy(style).unsqueeze(0).unsqueeze(0)
    style = style.unsqueeze(1).repeat(1, NUM_IMGS, 1, 1, 1).to(DEVICE)

    char_img = text_to_image(character).resize((64, 64))
    char = torch.from_numpy(np.array(char_img, dtype=np.float32) / 255.0)
    char = char.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_seq = model.inference(style, char, MAX_LEN)
    sos = torch.tensor([[0, 0, 1, 0, 0]], device=pred_seq.device)
    coords = torch.cat([sos, pred_seq[0]], dim=0).cpu().numpy()
    result = coords_render(coords, split=True, width=256, height=256, thickness=8, board=1)
    return result


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type='pil', label='Style Image'), gr.Textbox(label='Character')],
    outputs=gr.Image(type='pil', label='Generated'),
    title='SDT Handwriting Generation'
)

if __name__ == '__main__':
    demo.launch()

