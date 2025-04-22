import json
from pathlib import Path

import gradio as gr

story_name = "story"
DATA_DIR = Path(__file__).parent / "data"
STORY_DIR = DATA_DIR / story_name
IMG_DIR = STORY_DIR / "img"


def load_story_data():
    with open(STORY_DIR / f"{story_name}.json", "r", encoding="utf-8") as file:
        return json.load(file)


story = load_story_data()
pages = story["pages"]

with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown(f"# {title if (title := story['title']) else 'Untitled'}")
    page_radio = gr.Radio(
        label="Page",
        choices=[(k.title(), k) for k in pages.keys()],
        interactive=True,
    )
    page_text = gr.Textbox(
        label="Text",
        lines=5,
        interactive=False,
        placeholder="Select a page to see the text.",
    )
    page_img = gr.Image(
        label="Image",
        type="filepath",
        interactive=False,
        show_label=True,
        height=600,
    )

    @page_radio.change(inputs=page_radio, outputs=[page_text, page_img])
    def show_page_content(choice):
        page = pages[choice]
        text = page["text"]
        img_name = page["img"]
        img_path = IMG_DIR / img_name
        return text, img_path


if __name__ == "__main__":
    demo.launch()
