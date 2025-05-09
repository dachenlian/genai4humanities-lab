import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import shutil
import zipfile
from pathlib import Path

import gradio as gr
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

    with open(STORY_DIR / "story.json", "r", encoding="utf-8") as file:
        story = Story.model_validate_json(file.read())
    pages = story.pages

    ALL_TEXT_IMG = story.gather_text_and_images(IMG_DIR)


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

        # This is assumed to update global 'story' and 'pages'
        # This function itself could potentially accept and use the 'progress' object
        # if it has long-running internal loops.
        load_story_data()
        if not story or not pages:  # Check if loading was successful
            raise ValueError("無法從解壓縮的檔案載入故事資料或頁面。請檢查 ZIP 內容。")
        print(f"故事資料 '{story.title}' 載入完成。")

        # --- Finalizing UI Updates (90% - 100%) ---
        progress(0.9, desc="正在更新頁面選項...")

        # # Unzip the uploaded file
        # with zipfile.ZipFile(filepath, "r") as zip_ref:
        #     zip_ref.extractall(STORY_DIR)

        # files = list(STORY_DIR.glob("*"))
        # rprint(f"Uploaded files: {files}")

        # # Load the story data
        # load_story_data()

        # Generate updated radio choices
        new_choices = [("全部", "all")]
        if pages is not None:
            new_choices = [(k.title(), k) for k in pages.keys()] + [("全部", "all")]
        else:
            new_choices = []

        page_radio = gr.Radio(
            label="Page",
            choices=new_choices,
            interactive=True,
        )

        # Return the new story title and radio choices
        story_title = f"# {story.title if story else 'Story Title'}"
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

    return story_title, page_radio


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

    # Update the upload button to connect to the outputs
    story_upload_button.upload(
        handle_story_upload,
        inputs=story_upload_button,
        outputs=[story_title_md, page_radio],
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
            user_msgs: list[Image.Image | str | None] = []  # Allow None temporarily
            if selected_choice == "all":
                # Ensure ALL_TEXT_IMG doesn't contain None, though it shouldn't
                user_msgs = [
                    item for item in ALL_TEXT_IMG if item is not None
                ]  # Use a copy
                print(f"Using ALL ({len(user_msgs)}) text/image items as context.")
            elif selected_choice not in past_choices and selected_choice in pages:
                past_choices.add(selected_choice)
                print(f"Selected choice: {selected_choice}")
                print(f"Past choices: {past_choices}")
                user_msgs = [f"Page {selected_choice}"]  # Add page context
                page_data = pages[selected_choice]
                img_path = IMG_DIR / page_data.img if page_data.img else None
                image_content = None
                if img_path and img_path.exists():
                    try:
                        image_content = Image.open(img_path)
                        print(f"Adding image: {img_path.name}")
                    except Exception as img_e:
                        print(f"Warning: Could not open image {img_path}: {img_e}")
                if image_content:
                    user_msgs.append(image_content)
                if page_data.text:  # Check if text exists and is not empty/None
                    user_msgs.append(page_data.text)
                    print("Adding page text.")
            else:
                print(
                    "Chat has history or no specific page selected, sending only new message."
                )
                user_msgs = []  # If history exists, just send the new message

            if message:  # Ensure message is not None or empty before appending
                user_msgs.append(message)  # Add the actual user message

            # Filter out any None values that might have slipped in
            final_user_msgs = [item for item in user_msgs if item is not None]

            return final_user_msgs

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

        # # Streaming Version:
        # def transcribe_voice(stream, new_chunk):
        #     sr, y = new_chunk
        #     if y.ndim > 1:
        #         y = y.mean(axis=1)
        #     y = y.astype(np.float32)
        #     max_val = np.max(np.abs(y))
        #     if max_val > 0:
        #         y /= max_val
        #     else:  # Handle silent chunk
        #         print("ASR received silent chunk.")
        #         # Return current stream, don't update text (return None for text component)
        #         return stream, None

        #     # Accumulate audio
        #     if stream is not None:
        #         if stream.size > 0:
        #             accumulated_stream = np.concatenate([stream, y])
        #         else:
        #             accumulated_stream = y
        #     else:
        #         accumulated_stream = y

        #     print(
        #         f"Running ASR on accumulated stream of length {len(accumulated_stream)}..."
        #     )
        #     try:
        #         # Use the defined asr_pipeline
        #         # Adjust generate_kwargs if needed for your specific Whisper model/version
        #         result = TRANSCRIBER({"sampling_rate": sr, "raw": accumulated_stream})
        #         text_result = result.get("text", "")
        #         print(f"ASR Output: {text_result}")
        #         return (
        #             accumulated_stream,
        #             text_result,
        #         )  # Return updated stream and new text
        #     except Exception as e:
        #         print(f"Error during ASR transcription: {e}")
        #         # Return current stream, don't update text
        #         return stream, None

        # # State for ASR audio accumulation (Streaming Version)
        # asr_state = gr.State(None)

        # # UI Layout for Voice Chat (Streaming Version)
        # with gr.Row(equal_height=True):
        #     mic_input = gr.Audio(
        #         sources=["microphone"],
        #         streaming=True,
        #         label="麥克風輸入 (串流)",
        #     )
        #     asr_output_textbox = gr.Textbox(label="語音識別結果 (即時)")

        # # Wire ASR streaming - connects mic input to transcribe_voice function (Streaming Version)
        # mic_input.stream(
        #     fn=transcribe_voice,
        #     inputs=[asr_state, mic_input],
        #     outputs=[asr_state, asr_output_textbox],
        # )
        # # End Streaming Version

        # Non-Streaming Version:
        def transcribe_voice(audio_filepath):
            if audio_filepath is None:
                print("No audio file provided for transcription.")
                return ""  # Return empty string if no file

            print(f"Running ASR on audio file: {audio_filepath}...")
            try:
                # Use the defined TRANSCRIBER pipeline directly on the file path
                result = TRANSCRIBER(audio_filepath)
                text_result = result.get("text", "")
                print(f"ASR Output: {text_result}")
                return text_result  # Return only the transcribed text
            except Exception as e:
                print(f"Error during ASR transcription: {e}")
                return f"[Error during transcription: {e}]"

        # UI Layout for Voice Chat (Non-Streaming Version)
        with gr.Row(equal_height=True):
            mic_input = gr.Audio(
                sources=["microphone"],
                streaming=False,  # Set to False for non-streaming
                type="filepath",  # Ensure output is a filepath
                label="麥克風輸入 (錄音後處理)",
            )
            asr_output_textbox = gr.Textbox(label="語音識別結果")

        # Wire ASR - trigger transcription when recording stops (Non-Streaming Version)
        mic_input.stop_recording(
            fn=transcribe_voice,
            inputs=[mic_input],  # Input is the audio component itself
            outputs=[asr_output_textbox],  # Output is the textbox
        )
        # End Non-Streaming Version

        gr.Markdown("---")  # Separator

        # --- LLM and TTS Components ---
        with gr.Row():
            with gr.Column(scale=1):
                ref_audio_input = gr.Audio(label="語音參考音檔", type="filepath")
                process_button = gr.Button("獲取回饋與生成語音")
            with gr.Column(scale=2):
                llm_output_textbox = gr.Textbox(label="LLM 模型回饋")
                tts_audio_output = gr.Audio(label="合成語音輸出", type="filepath")

        # --- Define LLM + TTS logic Function ---
        def run_voice_llm_tts(
            api_key: str,
            current_asr_text: str,
            ref_audio_path: str | Path | None = TTS_DIR / "voice.wav",
            transcription: str = "",
        ) -> tuple[str, str | None]:
            llm_response_text = "[LLM] An error occurred."  # Default error message
            tts_path = None  # Default
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
                prepare_chat(api_key)  # Pass the API key to the function
                if not CHAT:
                    raise ValueError("Chat not initialized.")

                # 2. Basic input validation
                if not current_asr_text or not current_asr_text.strip():
                    return "[LLM] No ASR text detected to process.", None
                if not ref_audio_path:
                    return "[LLM] Please provide reference audio for TTS.", None

                # 3. Send ASR text to the existing Gemini Chat
                print(f"Sending to LLM (Voice Chat): '{current_asr_text}'")

                # Prepare message context if needed (similar to text chat)
                # Ensure current_asr_text is not None before passing
                user_msgs_with_context_raw = prepare_user_message_context(
                    current_asr_text or ""
                )

                # Explicitly filter None values again before sending
                user_msgs_with_context = [
                    item for item in user_msgs_with_context_raw if item is not None
                ]

                if not user_msgs_with_context:
                    # Handle case where filtering results in an empty list (e.g., only None items were present)
                    print("Warning: No valid content to send to LLM after filtering.")
                    return "[LLM] No valid content to process after filtering.", None

                print(
                    f"Sending message to LLM (Voice Chat)... First part: {str(user_msgs_with_context[0])[:50]}..."
                )
                # Send message with context
                llm_res = CHAT.send_message(message=user_msgs_with_context)  # type: ignore
                # llm_res = CHAT.send_message(message=current_asr_text) # Original line without context
                llm_response_text = llm_res.text
                if not llm_response_text:
                    raise ValueError("LLM returned empty response.")
                print(f"LLM Response (Voice Chat): '{llm_response_text}'")

                # 4. Run TTS on the LLM response using the imported function
                print(f"Running TTS on: '{llm_response_text}'")
                # Assumes 'tts' is the correctly modified function imported from ai_storytime.tts
                tts_path = tts(
                    path_to_ref_audio=str(ref_audio_path),
                    gen_text=llm_response_text,
                    ref_text=transcription,  # Provide reference text for TTS if needed/available
                    device="cuda"
                    if torch.cuda.is_available()
                    else "cpu",  # Pass the correct device literal
                )
                if tts_path is None:
                    print("TTS generation failed.")
                    llm_response_text += (
                        "\n[TTS Generation Failed]"  # Append failure notice
                    )
                else:
                    print(f"TTS generated successfully: {tts_path}")

            except gr.Error as e:  # Catch Gradio specific errors (like missing API key)
                print(f"Gradio Error in Voice LLM/TTS: {e}")
                llm_response_text = f"[Error] {e}"
            except Exception as e:
                print(f"Error during Voice LLM/TTS processing: {e}")
                import traceback  # Add traceback for detailed debugging

                traceback.print_exc()  # Print the full traceback
                llm_response_text = f"[Error] An unexpected error occurred: {e}"
                tts_path = None  # Ensure TTS output is cleared on general error

            # Return LLM text and TTS audio path (or None)
            return llm_response_text, tts_path

        # Wire the button click event
        process_button.click(
            fn=run_voice_llm_tts,
            inputs=[api_key, asr_output_textbox, ref_audio_input],
            outputs=[llm_output_textbox, tts_audio_output],
        )

        # Clear Button for Voice Chat tab components
        clear_voice_button = gr.Button("Clear Voice Outputs")

        def clear_voice_outputs():
            # Reset ASR text, LLM text, Ref audio, TTS audio
            print("Clearing voice chat outputs (ASR text, LLM text, TTS audio).")
            # If using streaming, you might need to return None for asr_state as well
            # return None, "", "", None # For Streaming: State, ASR Text, LLM Text, TTS Audio
            return "", "", None  # For Non-Streaming: ASR Text, LLM Text, TTS Audio

        clear_voice_button.click(
            fn=clear_voice_outputs,
            inputs=None,
            outputs=[
                # If using streaming, add asr_state here:
                # asr_state,
                asr_output_textbox,
                llm_output_textbox,
                tts_audio_output,
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
    demo.launch(share=True, debug=False)
