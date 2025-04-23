from pathlib import Path

import gradio as gr
from google import genai
from PIL import Image
import numpy as np
import torch
from transformers import pipeline

# from ai_storytime.asr import asr
from ai_storytime.tts import tts
from ai_storytime.models import Story
from ai_storytime.utils import get_gemini_api_key

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
story_name = "story"
MODEL_ID = "gemini-2.0-flash"
DATA_DIR = Path(__file__).parent / "data"
STORY_DIR = DATA_DIR / story_name
IMG_DIR = STORY_DIR / "img"
DEFAULT_API_KEY = get_gemini_api_key()
# DEFAULT_API_KEY = ""
ALL_TEXT_IMG: list[Image.Image | str] = []

client = None
CHAT = None


def load_story_data() -> Story:
    global ALL_TEXT_IMG
    with open(STORY_DIR / f"{story_name}.json", "r", encoding="utf-8") as file:
        story = Story.model_validate_json(file.read())

    ALL_TEXT_IMG = story.gather_text_and_images(IMG_DIR)
    return story


story = load_story_data()
pages = story.pages

selected_image = None
selected_text = None
selected_choice = None

with gr.Blocks(
    theme=gr.themes.Citrus()  # type: ignore
) as demo:
    local_storage = gr.BrowserState([DEFAULT_API_KEY])
    with gr.Sidebar():
        api_key = gr.Textbox(
            label="Gemini API Key",
            placeholder="請輸入您的 Gemini API 金鑰",
            type="password",
            show_label=True,
            lines=1,
            interactive=True,
            elem_id="api_key",
        )

    @demo.load(inputs=[local_storage], outputs=[api_key])
    def load_from_local_storage(saved_values):
        return saved_values[0]

    @gr.on([api_key.change], inputs=[api_key], outputs=[local_storage])
    def save_to_local_storage(password):
        gr.Info(
            "✅ 已儲存了 Gemini API 金鑰",
            duration=5,
            visible=True,
        )
        return [password]

    # TODO: allow user upload of data (json with text and img paths; images in img folder)
    gr.Markdown(f"# {story.title}")
    page_radio = gr.Radio(
        label="Page",
        choices=[(k.title(), k) for k in pages.keys()] + [("全部", "all")],
        interactive=True,
    )

    def create_new_chat():
        global CHAT
        if client:
            print("Creating new chat")
            CHAT = client.chats.create(model=MODEL_ID)

    with gr.Tab(label="Text Chat"):
        with gr.Row():
            chatbot = gr.Chatbot(type="messages", label="和我聊聊故事吧！")
        with gr.Row():
            msg = gr.Textbox()
        with gr.Row():
            clear = gr.ClearButton([msg, chatbot])
            clear.click(create_new_chat, inputs=None, outputs=None)

        def user(user_mesage: str, history: list[gr.ChatMessage]):
            return "", history + [gr.ChatMessage(content=user_mesage, role="user")]

        def prepare_chat():
            global CHAT
            if not api_key.value:
                raise gr.Error("請先輸入您的 Gemini API 金鑰！", duration=5)
            global client
            if not client:
                client = genai.Client(api_key=api_key.value)

            if not CHAT:
                CHAT = client.chats.create(model=MODEL_ID)

        def prepare_user_message_context(message: str) -> list[Image.Image | str]:
            assert CHAT
            if not CHAT.get_history():  # add story context
                if selected_choice == "all":
                    user_msgs = ALL_TEXT_IMG  # type: ignore
                else:
                    user_msgs = [f"Page {selected_choice}"]
                    if selected_image:
                        user_msgs.append(Image.open(selected_image))  # type: ignore
                    if selected_text:
                        user_msgs.append(selected_text)
                user_msgs.append(message)
            user_msgs = [message]

            return user_msgs

        def bot(history: list[gr.ChatMessage]):
            assert CHAT
            prepare_chat()
            user_message = history[-1].content
            user_msgs = prepare_user_message_context(user_message, history)
            history.append(gr.ChatMessage(role="assistant", content=""))
            res = CHAT.send_message_stream(message=user_msgs)
            for chunk in res:
                history[-1].content += chunk.text
                print(chunk.text, end="", flush=True)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        # def respond(message: str, chat_history: list[gr.ChatMessage]):

        #     history = CHAT.get_history()
        #     user_msgs: list[gtypes.PartUnionDict] = []
        #     if not history:  # add story context
        #         if selected_choice == "all":
        #             user_msgs = ALL_TEXT_IMG  # type: ignore
        #         else:
        #             user_msgs = [f"Page {selected_choice}"]
        #             if selected_image:
        #                 user_msgs.append(Image.open(selected_image))  # type: ignore
        #             if selected_text:
        #                 user_msgs.append(selected_text)
        #         user_msgs.append(message)
        #     user_msgs = [message]

        #     chat_history.append(
        #         gr.ChatMessage(
        #             content=message,
        #             role="user",
        #         )
        #     )
        #     res = CHAT.send_message(message=user_msgs)
        #     chat_history.append(
        #         gr.ChatMessage(
        #             content=res.text,
        #             role="assistant",
        #         )
        #     )

        #     return "", chat_history

        # msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)
    with gr.Tab(label="Voice Chat"):
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=DEVICE)

        def transcribe(stream, new_chunk):
            sr, y = new_chunk
            
            # Convert to mono if stereo
            if y.ndim > 1:
                y = y.mean(axis=1)
                
            y = y.astype(np.float32)
            y /= np.max(np.abs(y))

            if stream is not None:
                stream = np.concatenate([stream, y])
            else:
                stream = y
            
            # result = asr(device=DEVICE, **{"sampling_rate": sr, "raw": stream})
            # return stream, result.get("text", "")  # Use get() to safely access text field
            return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]
        
        with gr.Row():
            with gr.Column():
                # Define Input Components
                audio_in = gr.Audio(sources=["microphone"], type="numpy", streaming=True) 
                # Note: streaming might require .stream() event listener instead of button.click()
                # For simplicity, let's assume a button trigger for now
                submit_button = gr.Button("Transcribe")
            
            with gr.Column():
                # Define Output Components
                text_out = gr.Textbox(label="Transcription")

        # Define the interaction logic using event listeners
        submit_button.click(
            fn=my_function,          # Your processing function
            inputs=[audio_in, app_state],  # Map components/state to function args
            outputs=[app_state, text_out] # Map function return values to components/state
        )
        
        # If using streaming audio input like the previous example:
        # audio_in.stream(
        #     fn=my_function, 
        #     inputs=[app_state, audio_in], # Order matters for function args
        #     outputs=[app_state, text_out]
        # )


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
        global selected_text, selected_image, selected_choice
        selected_choice = choice
        if choice == "all":
            return "", ""

        page = pages[choice]
        text = page.text.split("\n")
        text = [f"# {line}" for line in text]
        text = "\n".join(text)

        img_name = page.img
        img_path = IMG_DIR / img_name
        selected_text = text

        selected_image = img_path
        return text, img_path


if __name__ == "__main__":
    demo.launch(share=True)
