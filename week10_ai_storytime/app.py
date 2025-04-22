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

selected_image = None
selected_text = None

with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    # TODO: allow user upload of data (json with text and img paths; images in img folder)
    gr.Markdown(f"# {title if (title := story['title']) else 'Untitled'}")
    page_radio = gr.Radio(
        label="Page",
        choices=[(k.title(), k) for k in pages.keys()],
        interactive=True,
    )
    with gr.Row(equal_height=True):
        page_text = gr.Textbox(
            label="Text",
            lines=20,
            interactive=False,
            placeholder="Select a page to see the text.",
        )
        page_img = gr.Image(
            label="Image",
            type="filepath",
            interactive=False,
            show_label=True,
            height=500,
        )

    @page_radio.change(inputs=page_radio, outputs=[page_text, page_img])
    def show_page_content(choice):
        global selected_text, selected_image
        page = pages[choice]
        text = page["text"]
        img_name = page["img"]
        img_path = IMG_DIR / img_name
        selected_text = text

        selected_image = img_path
        return text, img_path


if __name__ == "__main__":
    demo.launch()
