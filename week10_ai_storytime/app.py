import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import shutil
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from google import genai
from PIL import Image
from transformers import pipeline

# from ai_storytime.asr import asr
from ai_storytime.models import ChatMessage, Story
from ai_storytime.tts import tts
from ai_storytime.utils import get_gemini_api_key

story_name = "story"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_ID = "gemini-2.0-flash"
DATA_DIR = Path(__file__).parent / "data"
TTS_DIR = DATA_DIR / "tts"
TTS_VOICE_PATH = TTS_DIR / "voice.wav"
TTS_VOICE_REF_TRANSCRIPT_PATH = TTS_DIR / "transcription.txt"
STORY_DIR = DATA_DIR / story_name
IMG_DIR = STORY_DIR / "img"
try:
    DEFAULT_API_KEY = get_gemini_api_key()
except ValueError:
    print("Gemini API key not found. Please set it in the Gradio App")
    DEFAULT_API_KEY = None
# DEFAULT_API_KEY = ""
ALL_TEXT_IMG: list[Image.Image | str] = []

TRANSCRIBER = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=DEVICE
)

if not STORY_DIR.exists():
    print(f"Warning: Story directory not found: {STORY_DIR}")
    STORY_DIR.mkdir(parents=True, exist_ok=True)
if not IMG_DIR.exists():
    print(f"Warning: Image directory not found: {IMG_DIR}")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
if not TTS_DIR.exists():
    print(f"Warning: TTS directory not found: {TTS_DIR}")
    TTS_DIR.mkdir(parents=True, exist_ok=True)
if not DATA_DIR.exists():
    print(f"Warning: Data directory not found: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

if TTS_VOICE_PATH.exists():
    print(f"Using TTS voice file: {TTS_VOICE_PATH}")
else:
    print(f"Warning: TTS voice file not found: {TTS_VOICE_PATH}")
if TTS_VOICE_REF_TRANSCRIPT_PATH.exists():
    print(f"Using TTS voice reference transcript: {TTS_VOICE_REF_TRANSCRIPT_PATH}")
    _transcription = TTS_VOICE_REF_TRANSCRIPT_PATH.read_text()
else:
    print(
        f"Warning: TTS voice reference transcript not found: {TTS_VOICE_REF_TRANSCRIPT_PATH}"
    )
    _transcription = ""

client = None
CHAT = None
story = None
pages = None


def load_story_data():
    global ALL_TEXT_IMG, story, pages

    # Reset relevant states when loading new story data
    global selected_choice, past_choices, selected_image, selected_text
    selected_choice = None
    past_choices = set()
    selected_image = None
    selected_text = None

    try:
        with open(STORY_DIR / "story.json", "r", encoding="utf-8") as file:
            story = Story.model_validate_json(file.read())
        pages = story.pages
        ALL_TEXT_IMG = story.gather_text_and_images(IMG_DIR)
        print(f"Story '{story.title}' loaded successfully with {len(pages)} pages.")
        return True  # Indicate success
    except FileNotFoundError:
        print(f"Error: story.json not found in {STORY_DIR}")
        story = None
        pages = None
        ALL_TEXT_IMG = []
        return False  # Indicate failure
    except Exception as e:
        print(f"Error loading story data: {e}")
        story = None
        pages = None
        ALL_TEXT_IMG = []
        return False  # Indicate failure


# load_story_data()  # load default story


selected_image = None
selected_text = None
selected_choice = None
past_choices = set()


def handle_story_upload(filepath, progress=gr.Progress(track_tqdm=True)):
    global story_name, STORY_DIR, IMG_DIR, story, pages, ALL_TEXT_IMG

    try:
        story_name = "uploads"
        STORY_DIR = DATA_DIR / story_name
        IMG_DIR = STORY_DIR / "img"

        # Clean up previous upload directory if it exists
        if STORY_DIR.exists():
            print(f"偵測到舊的上傳目錄，正在清除: {STORY_DIR}")
            progress(0.05, desc="清除舊的資料...")
            shutil.rmtree(STORY_DIR)

        STORY_DIR.mkdir(parents=True, exist_ok=True)  # Ensure base dir exists

        # --- Unzipping (10% - 60%) ---
        progress(0.1, desc="正在解壓縮檔案...")
        print(f"正在解壓縮 {filepath} 到 {STORY_DIR}")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # Note: extractall doesn't provide easy granular progress.
            # For large files, you might iterate zip_ref.infolist()
            # and extract file by file, updating progress more often.
            zip_ref.extractall(STORY_DIR)
        print("解壓縮完成。")

        # --- Loading Data (60% - 90%) ---
        progress(0.6, desc="正在載入故事資料...")
        files = list(STORY_DIR.glob("**/*"))  # List files recursively for info
        print(
            f"解壓縮後檔案列表 (前 10 項): {[str(f.relative_to(STORY_DIR)) for f in files[:10]]}"
        )

        # Load story data and check if successful
        load_successful = load_story_data()
        if not load_successful or not story or not pages:
            raise ValueError("無法從解壓縮的檔案載入故事資料或頁面。請檢查 ZIP 內容。")
        print(f"故事資料 '{story.title}' 載入完成。")

        # --- Finalizing UI Updates (90% - 100%) ---
        progress(0.9, desc="正在更新頁面選項...")

        # Generate updated radio choices for main page display
        main_page_choices = [("全部", "all")]
        if pages:
            main_page_choices = [(k.title(), k) for k in pages.keys()] + [
                ("全部", "all")
            ]

        # Generate updated radio choices for voice reading page selection
        reading_page_choices = []
        if pages:
            reading_page_choices = [(k.title(), k) for k in pages.keys()]

        # Update the Gradio components
        main_page_radio_update = gr.Radio(
            choices=main_page_choices,
            value=None,  # Reset selection
            interactive=True,
        )
        reading_page_radio_update = gr.Radio(
            choices=reading_page_choices,
            value=None,  # Reset selection
            interactive=True,
            # Keep visibility based on current mode (might need adjustment if mode state isn't updated yet)
        )

        # Return the new story title and updated radio components
        story_title_update = f"# {story.title}"

    except FileNotFoundError as e:
        print(f"處理上傳時發生錯誤: 找不到檔案 {e}")
        raise gr.Error(f"處理失敗：找不到必要的檔案或目錄。{e}")
    except zipfile.BadZipFile:
        print(f"處理上傳時發生錯誤: 無效的 ZIP 檔案 {filepath}")
        raise gr.Error("處理失敗：上傳的不是有效的 ZIP 檔案。")
    except Exception as e:
        print(f"處理上傳檔案時發生未預期的錯誤: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging
        raise gr.Error(f"處理上傳時發生未預期的錯誤：{e}")

    progress(1.0, desc="處理完成！")
    print("上傳處理完成。")
    gr.Info("✅ 故事上傳並載入成功！")

    # Return updates for story title, main page radio, and reading page radio
    return story_title_update, main_page_radio_update, reading_page_radio_update


def handle_voice_upload(filepath, progress=gr.Progress(track_tqdm=True)):
    global TTS_VOICE_PATH, TTS_VOICE_REF_TRANSCRIPT_PATH, _transcription

    try:
        # Clean up previous upload directory if it exists
        if TTS_DIR.exists():
            print(f"偵測到舊的上傳目錄，正在清除: {TTS_DIR}")
            progress(0.05, desc="清除舊的資料...")
            shutil.rmtree(TTS_DIR)

        TTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure base dir exists

        # --- Unzipping (10% - 60%) ---
        progress(0.1, desc="正在解壓縮檔案...")
        print(f"正在解壓縮 {filepath} 到 {TTS_DIR}")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(TTS_DIR)
        print("解壓縮完成。")

        # --- Loading Data (60% - 90%) ---
        progress(0.6, desc="正在載入語音資料...")
        files = list(TTS_DIR.glob("**/*"))  # List files recursively for info
        print(
            f"解壓縮後檔案列表 (前 10 項): {[str(f.relative_to(TTS_DIR)) for f in files[:10]]}"
        )

        # Check if the voice file and reference transcript exist
        if not TTS_VOICE_PATH.exists():
            raise FileNotFoundError(f"TTS voice file not found: {TTS_VOICE_PATH}")
        if not TTS_VOICE_REF_TRANSCRIPT_PATH.exists():
            raise FileNotFoundError(
                f"TTS voice reference transcript not found: {TTS_VOICE_REF_TRANSCRIPT_PATH}"
            )
            _transcription = ""
        else:
            _transcription = TTS_VOICE_REF_TRANSCRIPT_PATH.read_text()
            print(f"Using TTS voice reference transcript: {_transcription}")

        # --- Finalizing UI Updates (90% - 100%) ---
        progress(0.9, desc="正在更新頁面選項...")

    except FileNotFoundError as e:
        print(f"處理上傳時發生錯誤: 找不到檔案 {e}")
        raise gr.Error(f"處理失敗：找不到必要的檔案或目錄。{e}")
    except zipfile.BadZipFile:
        print(f"處理上傳時發生錯誤: 無效的 ZIP 檔案 {filepath}")
        raise gr.Error("處理失敗：上傳的不是有效的 ZIP 檔案。")
    except Exception as e:
        print(f"處理上傳檔案時發生未預期的錯誤: {e}")
        import traceback

        traceback.print_exc()
        # Print full traceback for debugging
        raise gr.Error(f"處理上傳時發生未預期的錯誤：{e}")
    progress(1.0, desc="處理完成！")
    print("上傳處理完成。")
    gr.Info("✅ 語音上傳並載入成功！")


with gr.Blocks(
    theme=gr.themes.Citrus()  # type: ignore
) as demo:
    local_storage = gr.BrowserState([DEFAULT_API_KEY])
    with gr.Sidebar():
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="請輸入您的 Gemini API 金鑰",
                    type="password",
                    show_label=True,
                    lines=1,
                    interactive=True,
                    elem_id="api_key",
                )
                story_upload_button = gr.UploadButton(
                    "📖 上傳故事檔案（.zip）",
                    file_types=[".zip"],
                    file_count="single",
                )
                # voice_upload_button = gr.UploadButton(
                #     "🗣️ 上傳語音檔案（.zip）",
                #     file_types=[".zip"],
                #     file_count="single",
                # )

    story_title_md = gr.Markdown(f"# {story.title if story else 'Story Title'}")
    choices = [("全部", "all")]
    if pages is not None:
        choices = [(k.title(), k) for k in pages.keys()] + [("全部", "all")]
    else:
        choices = []
    page_radio = gr.Radio(
        label="Page",
        choices=choices,
        interactive=True,
    )

    # Update the upload button to connect to the outputs, including the new reading page radio
    story_upload_button.upload(
        handle_story_upload,
        inputs=story_upload_button,
        # Outputs now include the reading_page_radio (defined later in the Voice Chat tab)
        outputs=[story_title_md, page_radio, page_radio],  # Add reading_page_radio here
    )

    @demo.load(inputs=[local_storage], outputs=[api_key])
    def load_from_local_storage(saved_values):
        key = saved_values[0] if saved_values and saved_values[0] else ""
        global client, CHAT
        if not key:
            client = None
            CHAT = None
        elif (
            client and client._api_client.api_key != key
        ):  # Re-initialize if key changes
            client = None
            CHAT = None
            print("API Key changed, client/chat will be re-initialized.")
        print(f"Loaded API key: {key}")
        return key

    @gr.on([api_key.change], inputs=[api_key], outputs=[local_storage])
    def save_to_local_storage(password):
        gr.Info(
            "✅ 已儲存了 Gemini API 金鑰",
            duration=5,
            visible=True,
        )
        return [password]

    # --- Helper function to initialize client and chat ---
    # (Can be defined globally or passed around if needed)
    def prepare_chat(current_api_key: str):
        global CHAT, client
        # Ensure API key component is accessible here
        if not current_api_key:
            print(f"API key: {current_api_key}")
            print("API key is missing.")
            raise gr.Error("請先輸入您的 Gemini API 金鑰！", duration=5)
        # Initialize client if needed or if key changed
        if not client or client._api_client.api_key != current_api_key:
            print(
                f"Initializing Gemini Client with new key ending in ...{current_api_key[-4:]}"
            )
            try:
                client = genai.Client(api_key=current_api_key)
                # Optionally test client connection here if possible
                CHAT = None  # Reset chat when client is new
                print("Gemini Client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Gemini Client: {e}")
                raise gr.Error(f"無法初始化 Gemini 客戶端：{e}", duration=10)

        # Initialize chat if needed
        if not CHAT:
            print(f"Creating new Gemini chat with model: {MODEL_ID}")
            try:
                CHAT = client.chats.create(model=MODEL_ID)
                print("New Gemini chat created.")
            except Exception as e:
                print(f"Failed to create Gemini chat: {e}")
                raise gr.Error(f"無法創建 Gemini 聊天：{e}", duration=10)

    with gr.Tab(label="Text Chat"):
        with gr.Row():
            chatbot = gr.Chatbot(type="messages", label="和我聊聊故事吧！")
        with gr.Row():
            msg = gr.Textbox()
        with gr.Row():
            clear = gr.ClearButton([msg, chatbot])

            # Clear button should also reset the global CHAT object
            def clear_text_chat_and_reset():
                global \
                    CHAT, \
                    selected_choice, \
                    past_choices, \
                    selected_image, \
                    selected_text
                selected_choice = None
                selected_image = None
                selected_text = None
                past_choices = set()  # Reset past choices
                # Clear the chat history
                CHAT = None
                print("Text chat cleared and global CHAT reset.")
                # Return empty values for msg and chatbot components
                return (
                    "",
                    [],
                )  # ClearButton handles component clearing implicitly if None returned? Let's be explicit.

            clear.click(clear_text_chat_and_reset, inputs=None, outputs=[msg, chatbot])

        def user(user_mesage: str, history: list[gr.ChatMessage]):
            return "", history + [gr.ChatMessage(content=user_mesage, role="user")]

        def prepare_user_message_context(message: str) -> list[Image.Image | str]:
            # Prepares the initial context message for the LLM if chat history is empty
            user_msgs: list[Image.Image | str] = []
            if selected_choice == "all":
                user_msgs = ALL_TEXT_IMG[:]  # Use a copy
                print(f"Using ALL ({len(user_msgs)}) text/image items as context.")
            elif selected_choice not in past_choices and selected_choice in pages:
                past_choices.add(selected_choice)
                print(f"Selected choice: {selected_choice}")
                print(f"Past choices: {past_choices}")
                user_msgs = [f"Page {selected_choice}"]  # Add page context
                page_data = pages[selected_choice]
                img_path = IMG_DIR / page_data.img if page_data.img else None
                if img_path and img_path.exists():
                    user_msgs.append(Image.open(img_path))
                    print(f"Adding image: {img_path.name}")
                if page_data.text:
                    user_msgs.append(page_data.text)
                    print("Adding page text.")
            else:
                print("Chat has history, sending only new message.")
                user_msgs = []  # If history exists, just send the new message

            user_msgs.append(message)  # Add the actual user message

            return user_msgs

        def bot(api_key: str, history: list[ChatMessage]):
            try:
                # Ensure client and chat are ready
                prepare_chat(api_key)
                if not CHAT:
                    raise ValueError("Chat not initialized")

                user_message = history[-1]["content"]
                # Prepare message context (story elements + user text)
                user_msgs_with_context = prepare_user_message_context(user_message)

                history.append({"role": "assistant", "content": ""})
                print(
                    f"Sending message to LLM (Text Chat)... First part: {str(user_msgs_with_context[0])[:50]}..."
                )
                res = CHAT.send_message_stream(message=user_msgs_with_context)  # type: ignore

                full_response = ""
                for chunk in res:
                    if chunk.text:
                        history[-1]["content"] += chunk.text
                        full_response += chunk.text
                        # Small optimization: yield only every few chunks or based on time?
                        yield history  # Stream intermediate results to UI
                print(f"\nLLM Full Response (Text Chat): {full_response}")

            except Exception as e:
                print(f"Error in text chat bot function: {e}")
                # Update the last message (which is the empty assistant message) with error info
                history[-1]["content"] = f"[Error] {e}"
                yield history  # Show error in the chat UI

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            fn=bot, inputs=[api_key, chatbot], outputs=chatbot
        )

    with gr.Tab(label="Voice Chat"):
        gr.Markdown("## Voice Interaction\nSpeak, get feedback, and hear the response.")

        # --- State Variables for Voice Chat ---
        # State for ASR audio accumulation
        asr_state = gr.State(None)
        # State for voice chat mode ('Free Chat' or 'Story Reading')
        voice_chat_mode_state = gr.State("Free Chat")
        # State for the ID of the page selected for reading
        reading_page_id_state = gr.State(None)
        # State to store the actual text of the selected reading page
        reading_page_text_state = gr.State(None)

        # --- Mode Selection ---
        with gr.Row():
            voice_mode_radio = gr.Radio(
                ["Free Chat", "Story Reading"],
                label="Voice Chat Mode",
                value="Free Chat",
                interactive=True,
            )

        # --- Reading Page Selection (Initially Hidden) ---
        reading_page_choices = []
        if pages:
            reading_page_choices = [(k.title(), k) for k in pages.keys()]

        with gr.Row(visible=False) as reading_page_row:  # Initially hidden row
            reading_page_radio = gr.Radio(
                choices=reading_page_choices,
                label="Select Page to Read",
                interactive=True,
                # value=None # Let the change handler set the value
            )

        # --- Mic Input and ASR Output ---
        with gr.Row(equal_height=True):
            mic_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="麥克風輸入 (串流)",
            )
            asr_output_textbox = gr.Textbox(label="語音識別結果 (即時)")

        # --- LLM and TTS Components ---
        gr.Markdown("---")  # Separator
        with gr.Row():
            with gr.Column(scale=1):
                # Keep ref_audio_input for potential future use or different TTS modes
                ref_audio_input = gr.Audio(
                    label="語音參考音檔 (For TTS Voice Cloning)", type="filepath"
                )
                process_button = gr.Button("獲取回饋與生成語音")
            with gr.Column(scale=2):
                llm_output_textbox = gr.Textbox(label="LLM 模型回饋 / 回應")
                tts_audio_output = gr.Audio(label="合成語音輸出", type="filepath")

        # --- Logic for Mode Change ---
        def handle_mode_change(mode):
            global reading_page_id_state, reading_page_text_state  # Access global state if needed, but prefer returning updates
            print(f"Voice chat mode changed to: {mode}")
            if mode == "Story Reading":
                if not pages:  # Check if story/pages are loaded
                    gr.Warning("請先上傳或載入故事以使用朗讀模式。")
                    # Force back to Free Chat if no pages available? Or just disable reading radio?
                    # Returning updates is safer:
                    return {
                        reading_page_row: gr.Row(visible=True),
                        reading_page_radio: gr.Radio(
                            interactive=False,
                        ),
                        voice_chat_mode_state: "Story Reading",  # Keep state consistent
                        reading_page_id_state: None,
                        reading_page_text_state: None,
                    }
                else:
                    # Show the reading page selection row and make radio interactive
                    return {
                        reading_page_row: gr.Row(visible=True),
                        reading_page_radio: gr.Radio(
                            interactive=True, value=None
                        ),  # Reset selection
                        voice_chat_mode_state: "Story Reading",
                        reading_page_id_state: None,  # Reset page ID
                        reading_page_text_state: None,  # Reset page text
                    }
            else:  # Free Chat mode
                # Hide the reading page selection row
                return {
                    reading_page_row: gr.Row(visible=False),
                    reading_page_radio: gr.Radio(interactive=False, value=None),
                    voice_chat_mode_state: "Free Chat",
                    reading_page_id_state: None,  # Clear page ID
                    reading_page_text_state: None,  # Clear page text
                }

        voice_mode_radio.change(
            fn=handle_mode_change,
            inputs=voice_mode_radio,
            outputs=[
                reading_page_row,
                reading_page_radio,  # Allow updating the radio itself (e.g., disable)
                voice_chat_mode_state,
                reading_page_id_state,
                reading_page_text_state,
            ],
        )

        # --- Logic for Reading Page Selection ---
        def handle_reading_page_selection(page_id):
            if page_id and pages and page_id in pages:
                selected_page_text = pages[page_id].text
                print(
                    f"Selected page '{page_id}' for reading. Text length: {len(selected_page_text)}"
                )
                # Return updates for the state variables
                return page_id, selected_page_text
            else:
                print(f"Invalid page selection '{page_id}' or no pages loaded.")
                # Clear the states if selection is invalid or cleared
                return None, None

        reading_page_radio.change(
            fn=handle_reading_page_selection,
            inputs=reading_page_radio,
            outputs=[reading_page_id_state, reading_page_text_state],
        )

        # --- ASR Streaming Logic (Unchanged) ---
        def transcribe_voice(stream, new_chunk):
            sr, y = new_chunk
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y.astype(np.float32)
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y /= max_val
            else:  # Handle silent chunk
                print("ASR received silent chunk.")
                # Return current stream, don't update text (return None for text component)
                return stream, None

            # Accumulate audio
            if stream is not None:
                if stream.size > 0:
                    accumulated_stream = np.concatenate([stream, y])
                else:
                    accumulated_stream = y
            else:
                accumulated_stream = y

            print(
                f"Running ASR on accumulated stream of length {len(accumulated_stream)}..."
            )
            try:
                # Use the defined asr_pipeline
                # Adjust generate_kwargs if needed for your specific Whisper model/version
                result = TRANSCRIBER({"sampling_rate": sr, "raw": accumulated_stream})
                text_result = result.get("text", "")
                print(f"ASR Output: {text_result}")
                return (
                    accumulated_stream,
                    text_result,
                )  # Return updated stream and new text
            except Exception as e:
                print(f"Error during ASR transcription: {e}")
                # Return current stream, don't update text
                return stream, None

        mic_input.stream(
            fn=transcribe_voice,
            inputs=[asr_state, mic_input],
            outputs=[asr_state, asr_output_textbox],
        )

        # --- Modified LLM + TTS logic Function ---
        def run_voice_llm_tts(
            api_key: str,
            current_asr_text: str,
            mode: str,  # New input: 'Free Chat' or 'Story Reading'
            reading_ref_text: str | None,  # New input: Text of the page to read
            ref_audio_path: str | Path | None = TTS_DIR / "voice.wav",
            transcription: str = "",
        ) -> tuple[str, str | None]:
            llm_response_text = "[LLM] An error occurred."
            tts_path = None

            print(f"Got reading mode: {mode}")

            # --- Input Validation ---
            if not current_asr_text or not current_asr_text.strip():
                return "[LLM] No speech detected to process.", None
            if not ref_audio_path:
                if TTS_VOICE_PATH.exists():
                    ref_audio_path = TTS_VOICE_PATH
                    transcription = _transcription
                else:
                    print(
                        f"Warning: Default TTS voice file not found: {TTS_VOICE_PATH}"
                    )
                    ref_audio_path = None
                    transcription = ""
            # if ref_audio_path and not Path(ref_audio_path).exists():
            #     ref_audio_path = None
            #     transcription = ""

            try:
                # 1. Ensure API key and chat are ready
                prepare_chat(api_key)
                if not CHAT:
                    raise ValueError("Chat not initialized.")

                # 2. Determine LLM prompt based on mode
                if mode == "Story Reading":
                    print("Mode: Story Reading")
                    if not reading_ref_text:
                        return (
                            "[LLM] Error: No reference text selected for reading mode.",
                            None,
                        )

                    # Construct prompt for reading feedback
                    prompt = f"""Please act as a reading coach. The user attempted to read the following text:
Reference Text: "{reading_ref_text}"

The user's speech was transcribed by ASR as:
ASR Output: "{current_asr_text}"

Compare the ASR Output to the Reference Text.
- If the ASR Output perfectly matches the Reference Text, provide brief positive feedback (e.g., "Excellent reading!", "Perfect!").
- If there are differences, point out the specific words or phrases that were misread or missed. Be concise and helpful.
- Provide only the feedback, do not include the reference or ASR text in your response.
"""
                    print("Sending Reading Feedback Prompt to LLM...")
                    llm_res = CHAT.send_message(
                        message=prompt
                    )  # Use send_message for single turn feedback
                    llm_response_text = llm_res.text
                    print(f"LLM Reading Feedback: '{llm_response_text}'")

                else:  # Free Chat mode
                    print("Mode: Free Chat")
                    # Use the existing chat history context logic if needed, or just send the ASR text
                    # For simplicity here, just sending the ASR text directly.
                    # If context is desired, integrate prepare_user_message_context logic.
                    print(f"Sending to LLM (Free Chat): '{current_asr_text}'")
                    llm_res = CHAT.send_message(message=current_asr_text)
                    llm_response_text = llm_res.text
                    print(f"LLM Response (Free Chat): '{llm_response_text}'")

                if not llm_response_text:
                    raise ValueError("LLM returned empty response.")

                # 3. Run TTS on the LLM response (feedback or chat reply)
                if ref_audio_path:  # Only run TTS if reference audio is available
                    print(f"Running TTS on: '{llm_response_text}'")
                    tts_path = tts(
                        path_to_ref_audio=str(ref_audio_path),
                        gen_text=llm_response_text,
                        ref_text=transcription,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    if tts_path is None:
                        print("TTS generation failed.")
                        llm_response_text += "\n[TTS Generation Failed]"
                    else:
                        print(f"TTS generated successfully: {tts_path}")
                else:
                    print("Skipping TTS generation as reference audio is missing.")

            except gr.Error as e:
                print(f"Gradio Error in Voice LLM/TTS: {e}")
                llm_response_text = f"[Error] {e}"
            except Exception as e:
                print(f"Error during Voice LLM/TTS processing: {e}")
                import traceback

                traceback.print_exc()
                llm_response_text = f"[Error] An unexpected error occurred: {e}"
                tts_path = None

            return llm_response_text, tts_path

        # --- Update Button Click Wiring ---
        process_button.click(
            fn=run_voice_llm_tts,
            # Add mode and reading text states to inputs
            inputs=[
                api_key,
                asr_output_textbox,
                voice_mode_radio,  # Pass the mode state
                reading_page_text_state,  # Pass the reading text state
                ref_audio_input,
            ],
            outputs=[llm_output_textbox, tts_audio_output],
        )

        # --- Update Clear Button Logic ---
        clear_voice_button = gr.Button("Clear Voice Outputs & State")

        def clear_voice_outputs_and_state():
            print("Clearing voice chat outputs and resetting states.")
            # Reset ASR state, ASR text, LLM text, TTS audio
            # Also reset mode to default, hide reading page selector, clear reading page state
            return {
                asr_state: None,
                asr_output_textbox: "",
                llm_output_textbox: "",
                tts_audio_output: None,
                voice_mode_radio: "Free Chat",  # Reset mode radio
                reading_page_row: gr.Row(visible=False),  # Hide reading row
                reading_page_radio: gr.Radio(value=None),  # Clear reading selection
                voice_chat_mode_state: "Free Chat",  # Reset mode state
                reading_page_id_state: None,  # Reset reading page ID state
                reading_page_text_state: None,  # Reset reading page text state
            }

        clear_voice_button.click(
            fn=clear_voice_outputs_and_state,
            inputs=None,
            # Update outputs to include the components and states being reset
            outputs=[
                asr_state,
                asr_output_textbox,
                llm_output_textbox,
                tts_audio_output,
                voice_mode_radio,
                reading_page_row,
                reading_page_radio,
                voice_chat_mode_state,
                reading_page_id_state,
                reading_page_text_state,
            ],
        )

    with gr.Row(equal_height=True):
        page_text = gr.Markdown(label="Text")
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
        print(f"Page selection changed to: {choice}")
        selected_choice = choice
        if choice == "all":
            selected_text = None
            selected_image = None
            # Maybe clear chat history when changing pages? Or keep context?
            # global CHAT; CHAT = None # Example: Reset chat on page change
            return (
                "輸入整個故事的內容和圖給模型參考",
                None,
            )  # Display placeholder text/image

        page = pages[choice]
        # text_content = page.text.replace(
        #     "\n", "\n\n"
        # )  # Add double newline for Markdown paragraphs
        text_content = page.text.split("\n")
        text_content = [f"# {line}" for line in text_content]
        text_content = "\n".join(text_content)

        img_name = page.img
        img_path = None
        if img_name:
            img_path_check = IMG_DIR / img_name
            if img_path_check.exists():
                img_path = str(img_path_check)
            else:
                print(f"Warning: Image file not found: {img_path_check}")

        selected_text = page.text  # Store original text if needed elsewhere
        selected_image = img_path  # Store image path

        # Reset chat when page changes?
        # global CHAT; CHAT = None
        # print("Resetting chat due to page change.")

        return text_content, img_path


if __name__ == "__main__":
    # Ensure story data is loaded on startup if default exists
    if not story:
        print("Attempting to load default story on startup...")
        load_story_data()
        # Update radio choices based on loaded data (if Gradio objects are accessible here)
        # This might be tricky; usually updates happen via handlers after launch.
        # The initial choices set during UI definition might be sufficient if load_story_data runs before gr.Blocks.

    demo.launch(share=True, debug=False)
