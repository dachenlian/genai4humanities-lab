import json
from pathlib import Path

import gradio as gr

from ai_storytime import prompts as p

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

with gr.Blocks(
    theme=gr.themes.Citrus()  # type: ignore
) as demo:
    # TODO: allow user upload of data (json with text and img paths; images in img folder)
    gr.Markdown(f"# {title if (title := story['title']) else 'Untitled'}")
    page_radio = gr.Radio(
        label="Page",
        choices=[(k.title(), k) for k in pages.keys()],
        interactive=True,
    )
    with gr.Row():
        chatbot = gr.Chatbot(type="messages", label="和我聊聊故事吧！")
    with gr.Row():
        msg = gr.Textbox()
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])

        def respond(message: str, chat_history: list[gr.ChatMessage]):
            print(selected_image, selected_text)
            if not selected_image:
                raise gr.Error(
                    "請先選擇一頁以獲取圖片和文本！", duration=5)
            if not chat_history and selected_image and selected_text:
                initial_prompt = p.prepare_answer_illustration_question_prompt(
                    img_path=selected_image,
                    book_text=selected_text,
                    user_question=message,
                )
                chat_history.append(gr.ChatMessage(role="user", content=message))
                assistant_response = "Test"
                chat_history.append(gr.ChatMessage(role="assistant", content=assistant_response))
            else:
                chat_history.append(gr.ChatMessage(role="user", content=message))
                assistant_response = "Test"
                chat_history.append(gr.ChatMessage(role="assistant", content=assistant_response))

            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)



    with gr.Row(equal_height=True):
        page_text = gr.Markdown(
            label="Text",
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
        text = page['text'].split("\n")
        text = [f"# {line}" for line in text]
        text = "\n".join(text)

        img_name = page["img"]
        img_path = IMG_DIR / img_name
        selected_text = text

        selected_image = img_path
        return text, img_path


if __name__ == "__main__":
    demo.launch()
