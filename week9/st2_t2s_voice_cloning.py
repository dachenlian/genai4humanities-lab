# -*- coding: utf-8 -*-
# %% [markdown]
# # 第九週實驗課：語音轉文字、大型語言模型與文字轉語音（含聲音複製）
#
# 本次實驗課目標：
# 1.  **語音轉文字 (S2T)**：將聲音轉成文字。
# 2.  **大型語言模型 (LLM)**：用來修正文字文法。
# 3.  **文字轉語音 (TTS) 與聲音複製**：將文字轉成特定人聲的語音。
#
# 我們會先分別測試，再整合成一個專案。

# %% [markdown]
# ## 安裝所需套件
#
# 執行此區塊來安裝本次實驗所需的 Python 套件。
# `%capture` 會隱藏安裝過程的輸出訊息。

# %%capture install_logs
# !pip install f5-tts sounddevice wavio ipywebrtc notebook autoawq -q # 安裝主要的 Python 套件 (-q 安靜模式)
# # !pip install git+https://github.com/openai/whisper.git # 可選：安裝最新版 Whisper
# # !pip install flash-attn --no-build-isolation # 可選：若環境支援可加速模型
# !apt install ffmpeg libportaudio2 -y -qq # 安裝系統工具 ffmpeg 和 PortAudio (-qq 更安靜)

# %% [markdown]
# ## 測試文字轉語音 (TTS) Gradio 介面 (可選)
#
# `f5-tts` 提供了一個網頁介面方便快速測試。取消下一行的註解並執行，會產生一個公開網址。

# %%
# !f5-tts_infer-gradio --share # 啟動 Gradio 介面

# %% [markdown]
# ## 匯入所需模組
#
# 匯入程式中會用到的 Python 模組。

# %%
import functools
import json
from pathlib import Path
from pprint import pprint
import shlex
import subprocess
import warnings

warnings.filterwarnings('ignore') # 忽略警告訊息

from rich import print as rprint # 美化 print 輸出
import gdown # 從 Google Drive 下載
from huggingface_hub import notebook_login # Hugging Face 登入
import IPython # 顯示音訊播放器等
import torch # PyTorch 深度學習框架
from transformers import ( # Hugging Face 函式庫
    AutoProcessor, AutoTokenizer, TextGenerationPipeline,
    AutoModelForSpeechSeq2Seq, AutomaticSpeechRecognitionPipeline,
    pipeline, BitsAndBytesConfig
)

# 檢查是否有 GPU 可用
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用的運算裝置 (Device): {device}")

# %% [markdown]
# ## Hugging Face Hub 登入 (非必要)
#
# 若要使用需要授權的模型 (如 Llama 3)，請執行 `notebook_login()`。若只用公開模型則跳過。

# %%
# notebook_login()

# %% [markdown]
# ## 下載範例聲音檔案
#
# 從 Google Drive 下載包含多個角色聲音的範例檔案夾。
# `!ls` 用來確認檔案已下載。

# %%
url = "https://drive.google.com/drive/folders/1dq-koI-P-tYJDfE1DtLdNZrUdCJLwcQZ?usp=sharing"
gdown.download_folder(url, quiet=True, use_cookies=False)
!ls week9/ # 列出下載的資料夾內容

# %% [markdown]
# ## 錄製你自己的聲音 (可選)
#
# 使用互動小工具錄音。點擊圓鈕開始/停止。
# **停止錄音後，請執行下一個儲存區塊。**
# (注意：此小工具在某些環境可能無法運作)

# %%
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    print("偵測不到 google.colab.output，互動小工具可能自動啟用或需要不同設定。")

from ipywebrtc import CameraStream, AudioRecorder
import ipywidgets as widgets
from IPython.display import display, Audio

camera = CameraStream(constraints={'audio': True, 'video': False}) # 只取聲音
recorder = AudioRecorder(stream=camera)

print("請點擊下方的圓形按鈕開始錄音，再次點擊則停止錄音。")
display(recorder)

# %% [markdown]
# ## 儲存錄音檔
#
# **請在上方停止錄音後，才執行此區塊。**
# 將錄製的聲音 (通常是 webm 格式) 存檔。
# 下方註解提供轉換成 WAV 格式的 `ffmpeg` 指令。

# %%
def save_kaggle_widget_recording(filename: str = "my_recording.webm"): # 改用更有意義的預設檔名
  """將 ipywebrtc 小工具錄製的音訊儲存到檔案。"""
  if not recorder.audio.value:
    print("錯誤：沒有錄到聲音。")
    return
  try:
    audio_bytes = recorder.audio.value
    with open(filename, 'wb') as f:
      f.write(audio_bytes)
    print(f"錄音已儲存為 {filename}")
    display(Audio(filename)) # 顯示播放器
  except Exception as e:
    print(f"儲存時發生錯誤: {e}")

# 儲存錄音
save_kaggle_widget_recording() # 使用預設檔名 "my_recording.webm"

# --- 選擇性：將錄音檔轉為 WAV 格式 ---
# !ffmpeg -i my_recording.webm -ac 1 -f wav my_recording.wav -y -hide_banner -loglevel error
# if $? -eq 0 ; then echo "成功轉換為 my_recording.wav"; display(Audio("my_recording.wav")); fi

# %% [markdown]
# ## 語音轉文字 (Speech-to-Text, S2T)
#
# 使用 OpenAI Whisper 模型將聲音轉成文字。
# `prepare_whisper_model` 函式會載入模型並設定好 pipeline。
# `@functools.cache` 用於快取模型，避免重複載入。

# %%
@functools.cache
def prepare_whisper_model(model_id: str = "openai/whisper-large-v3") -> AutomaticSpeechRecognitionPipeline: # 改用 v3，turbo 可能非公開
    """載入並設定 Whisper S2T 模型 pipeline。"""
    print(f"正在載入 Whisper 模型: {model_id} ...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" # 使用優化的注意力機制
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30, # 處理 30 秒聲音片段
        batch_size=16,     # 可依 GPU 記憶體調整
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Whisper 模型載入完成。")
    return pipe

# 載入 S2T 模型
s2t = prepare_whisper_model()

# %% [markdown]
# ### 執行語音轉文字
#
# 將 `week9` 資料夾中的第一個 `.wav` 檔進行辨識。

# %%
# 取得 week9 資料夾中所有 wav 檔案路徑
audio_files = list(Path("./week9").glob("*.wav"))
if not audio_files:
    print("錯誤：在 week9 資料夾中找不到任何 .wav 檔案。")
else:
    first_audio_path = str(audio_files[0].resolve())
    print("找到的聲音檔案路徑 (部分):")
    rprint([str(p) for p in audio_files[:5]]) # 只印出前 5 個

    print(f"\n播放第一個聲音檔: {first_audio_path}")
    display(IPython.display.Audio(first_audio_path))

    print("\n正在進行語音轉文字...")
    transcription = s2t(first_audio_path) # 對第一個檔案進行辨識
    print("\n辨識結果:")
    rprint(transcription)

# %% [markdown]
# ## 大型語言模型 (LLM) - 文法修正
#
# 使用 LLM (例如 Qwen) 來修正 S2T 結果的文法錯誤。
# `prepare_llm` 函式載入 LLM pipeline。

# %%
@functools.cache
def prepare_llm(model_id: str = "Qwen/Qwen1.5-7B-Chat-AWQ") -> TextGenerationPipeline: # 更新為 Qwen1.5 AWQ 模型
    """載入並設定 LLM 文字生成 pipeline。"""
    print(f"正在載入 LLM: {model_id} ...")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto", # 自動分配裝置
    )
    print("LLM 載入完成。")
    return pipe

# 載入 LLM
llm = prepare_llm()
tokenizer = llm.tokenizer # 取得分詞器

# %% [markdown]
# ### 測試 LLM 文法修正
#
# 準備中英文錯誤句子範例，並定義一個函式 `prepare_grammar_correction_messages` 來建立 LLM 的輸入。

# %%
# 英文錯誤範例
english_sentences = [
    {"incorrect": "Their going too the park later.", "correct": "They're going to the park later."},
    {"incorrect": "Me and him discussed it.", "correct": "He and I discussed it."},
]
# 中文錯誤範例 (臺灣用語)
chinese_sentences = [
    {"incorrect": "我明天在去。", "correct": "我明天再去。"},
    {"incorrect": "這見衣服必較好看。", "correct": "這件衣服比較好看。"},
]

def prepare_grammar_correction_messages(text: str) -> list[dict]:
    """建立 LLM 文法修正任務的 messages 列表。"""
    system_message = {"role": "system", "content": "你是一位專業的編輯，擅長修正英文和繁體中文的文法與拼寫錯誤。請直接提供修正後的文字，不要包含任何解釋。"}
    user_message = {"role": "user", "content": f"請修正以下文字的文法與拼寫錯誤：\n```\n{text}\n```"}
    return [system_message, user_message]

# --- 測試英文 ---
print("--- 測試英文修正 ---")
text_en_incorrect = english_sentences[0]["incorrect"]
print(f"原始: {text_en_incorrect}")
messages_en = prepare_grammar_correction_messages(text_en_incorrect)
res_en = llm(messages_en, max_new_tokens=50, return_full_text=False)
print(f"修正: {res_en[0]['generated_text'].strip()}")
print(f"預期: {english_sentences[0]['correct']}")
print("-" * 20)

# --- 測試中文 ---
print("--- 測試中文修正 ---")
text_zh_incorrect = chinese_sentences[1]["incorrect"]
print(f"原始: {text_zh_incorrect}")
messages_zh = prepare_grammar_correction_messages(text_zh_incorrect)
res_zh = llm(messages_zh, max_new_tokens=50, return_full_text=False)
print(f"修正: {res_zh[0]['generated_text'].strip()}")
print(f"預期: {chinese_sentences[1]['correct']}")

# %% [markdown]
# ## 文字轉語音 (TTS) 與聲音複製
#
# 使用 `f5-tts` 命令列工具，將文字轉換成指定參考聲音 (`ref_audio`) 的語音。
# `clone_voice` 函式封裝了執行此命令的過程。

# %%
def clone_voice(path_to_ref_audio: str,
                gen_text: str,
                ref_text: str = "", # 參考音檔的文字稿 (選填，但建議提供)
                output_file: str = "tts_output.wav",
                output_dir: str = "tts_output"): # 改用更有意義的預設目錄
    """使用 f5-tts 命令列工具進行聲音複製。"""
    cli_executable = "f5-tts_infer-cli"
    model = "F5TTS_v1_Base"
    output_path = Path(output_dir) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True) # 建立輸出目錄

    command = [
        cli_executable, "--model", model,
        "--ref_audio", str(Path(path_to_ref_audio).resolve()),
        "--ref_text", ref_text,
        "--gen_text", gen_text,
        "--output_dir", str(output_path.parent.resolve()),
        "--output_file", output_path.name,
        "--nfe_step", "64", # 推論步數
        "--device", device,
    ]
    print(f"準備執行 TTS 命令: {shlex.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\nTTS 命令執行成功。")
        print(f"生成的音訊檔案: {output_path}")
        display(IPython.display.Audio(str(output_path))) # 顯示播放器
    except subprocess.CalledProcessError as e:
        print(f"\nTTS 命令執行失敗 (錯誤碼 {e.returncode}):\nstderr: {e.stderr}")
    except FileNotFoundError:
        print(f"\n錯誤：找不到命令 '{cli_executable}'。請確認 f5-tts 已安裝。")
    except Exception as e:
        print(f"\n執行 TTS 時發生錯誤: {e}")

# %% [markdown]
# ### 測試聲音複製
#
# 載入範例聲音的文字稿 (`transcripts.json`)，然後選擇一個聲音 (如 Walken) 來念一段指定的中文文字。

# %%
# 載入文字稿
transcript_file = Path("./week9/transcripts.json")
transcripts = {} # 初始化為空字典
if transcript_file.exists():
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
        print("已載入文字稿檔案。")
    except json.JSONDecodeError:
        print(f"錯誤：無法解析文字稿檔案 {transcript_file}。")
else:
    print(f"警告：找不到文字稿檔案 {transcript_file}。TTS 的 ref_text 將會是空的。")

# --- 選擇聲音並執行 TTS ---
voice_key = "walken.wav" # 選擇要模仿的聲音檔名
ref_audio_path_str = f"./week9/{voice_key}" # 使用相對路徑更通用
ref_audio_path = Path(ref_audio_path_str)

if ref_audio_path.exists():
    ref_transcription = transcripts.get(voice_key, {}).get("transcription", "") # 安全地取得文字稿
    print(f"\n選擇模仿的聲音: {voice_key}")
    if not ref_transcription:
        print("(警告：找不到此聲音的文字稿，ref_text 將為空)")

    text_to_generate = "白日依山盡，黃河入海流。欲窮千里目，更上一層樓。"
    print(f"要生成的文字: {text_to_generate}")

    clone_voice(
        path_to_ref_audio=str(ref_audio_path), # 傳入字串路徑
        ref_text=ref_transcription,
        gen_text=text_to_generate,
        output_file=f"{voice_key.split('.')[0]}_poem.wav" # 更有意義的檔名
    )
else:
    print(f"錯誤：找不到參考聲音檔案 {ref_audio_path_str}")

# %% [markdown]
# ## 整合：從輸入到輸出
#
# `read_for_me` 函式整合了 S2T、LLM 修正、TTS 聲音複製的完整流程。
# 輸入可以是文字 (`input_text`) 或聲音檔 (`input_audio_path`)。

# %%
def read_for_me(
    s2t_model: AutomaticSpeechRecognitionPipeline,
    llm_model: TextGenerationPipeline,
    transcripts_dict: dict, # 改為必要參數
    input_text: str | None = None,
    input_audio_path: str | None = None,
    voice_to_clone_key: str = "spongebob.wav", # 預設模仿海綿寶寶
    output_file: str = "final_output.wav",
    output_dir: str = "tts_final"
):
    """整合 S2T -> LLM -> TTS 的流程。"""
    # --- 1. 檢查與取得輸入文字 ---
    if not input_text and not input_audio_path:
        raise ValueError("必須提供 input_text 或 input_audio_path。")
    if input_text and input_audio_path:
        raise ValueError("input_text 和 input_audio_path 不能同時提供。")
    if voice_to_clone_key not in transcripts_dict:
        print(f"警告：在文字稿字典中找不到 key '{voice_to_clone_key}'。")
        # raise ValueError(f"在 transcripts_dict 中找不到 key: '{voice_to_clone_key}'")

    original_text = ""
    if input_audio_path:
        print(f"步驟 1: S2T ({input_audio_path})")
        audio_file = Path(input_audio_path)
        if not audio_file.exists(): raise FileNotFoundError(f"找不到音檔: {input_audio_path}")
        original_text = s2t_model(str(audio_file))["text"]
        print(f"  >> S2T 結果: {original_text}")
    else:
        original_text = input_text
        print(f"步驟 1: 使用輸入文字: {original_text}")

    # --- 2. LLM 文法修正 ---
    print("\n步驟 2: LLM 文法修正")
    messages = prepare_grammar_correction_messages(original_text)
    # 增加溫度參數讓輸出稍微多樣化，並設定停止符號 (如果模型支援)
    llm_result = llm_model(messages, return_full_text=False, max_new_tokens=len(original_text) + 50,
                           temperature=0.7, eos_token_id=tokenizer.eos_token_id)
    corrected_text = llm_result[0]["generated_text"].strip()
    # 清理常見的模型輸出問題 (例如多餘的引號或標籤)
    corrected_text = corrected_text.replace("```", "").replace("`", "").strip('"').strip()
    print(f"  >> 修正後文字: {corrected_text}")

    # --- 3. TTS 聲音複製 ---
    print(f"\n步驟 3: TTS 聲音複製 (模仿 {voice_to_clone_key})")
    voice_info = transcripts_dict.get(voice_to_clone_key, {}) # 安全取得
    voice_ref_path_str = f"./week9/{voice_to_clone_key}"
    voice_ref_path = Path(voice_ref_path_str)
    voice_ref_text = voice_info.get("transcription", "")

    if not voice_ref_path.exists():
        raise FileNotFoundError(f"找不到參考聲音檔: {voice_ref_path_str}")
    if not voice_ref_text:
         print("(警告：找不到參考聲音文字稿，ref_text 將為空)")

    clone_voice(
        path_to_ref_audio=str(voice_ref_path),
        gen_text=corrected_text,
        ref_text=voice_ref_text,
        output_file=output_file,
        output_dir=output_dir
    )
    print(f"\n流程完成！最終語音已儲存至 {Path(output_dir) / output_file}")

# %% [markdown]
# ### 執行整合流程
#
# 測試 `read_for_me` 函式。
# 範例 1 使用文字輸入，範例 2 嘗試使用之前錄製的聲音檔。

# %%
# 執行整合流程 - 範例 1: 文字輸入，海綿寶寶聲音
print("--- 執行整合流程：範例 1 (文字輸入) ---")
try:
    read_for_me(
        s2t_model=s2t,
        llm_model=llm,
        transcripts_dict=transcripts, # 傳入文字稿字典
        input_text="i has a apple.", # 簡單錯誤英文
        voice_to_clone_key="spongebob.wav",
        output_file="spongebob_corrected_apple.wav"
    )
except Exception as e:
    print(f"執行範例 1 時發生錯誤: {e}")

print("\n" + "="*30 + "\n")

# 執行整合流程 - 範例 2: 聲音輸入 (若存在)，川普聲音
print("--- 執行整合流程：範例 2 (聲音輸入) ---")
my_recording_wav_path = "my_recording.wav" # 假設已轉換成 WAV
my_recording_webm_path = "my_recording.webm" # 或者使用 webm

input_audio_for_test = None
if Path(my_recording_wav_path).exists():
    input_audio_for_test = my_recording_wav_path
elif Path(my_recording_webm_path).exists():
     input_audio_for_test = my_recording_webm_path # Whisper 通常能處理 webm

if input_audio_for_test:
    print(f"使用錄音檔: {input_audio_for_test}")
    try:
        read_for_me(
            s2t_model=s2t,
            llm_model=llm,
            transcripts_dict=transcripts,
            input_audio_path=input_audio_for_test, # 使用錄音檔
            voice_to_clone_key="trump.wav",     # 模仿川普
            output_file="my_recording_trump_voice.wav"
        )
    except Exception as e:
        print(f"執行範例 2 時發生錯誤: {e}")
else:
    print(f"找不到錄音檔 ({my_recording_wav_path} 或 {my_recording_webm_path})，跳過範例 2。")
    print("請確認已成功錄音並執行儲存步驟。")

# %% [markdown]
# ## 使用 Gradio 錄製你自己的聲音 (取代 ipywebrtc)
#
# 這個區塊會啟動一個簡單的 Gradio 應用程式，讓你可以：
# 1.  點擊「錄音」按鈕開始錄音。
# 2.  再次點擊停止錄音。
# 3.  錄音會自動儲存成 `.wav` 檔案。
# 4.  介面會顯示儲存的檔案路徑，並提供播放和下載的選項。
#
# **使用方式：**
# * 執行這個 Python 程式碼區塊。
# * 在出現的 Gradio 介面中錄製聲音。
# * 錄音完成後，複製顯示的 `.wav` 檔案路徑 (例如 `recording_20250417_113000.wav`)。
# * 你可以在後面的 `read_for_me` 函式呼叫中，將這個路徑作為 `input_audio_path` 參數的值來使用。
# * **注意：** 錄音完成後，這個 Gradio App 會持續運行。你可以在完成錄音後手動停止這個 Cell 的執行，或者讓它繼續運行直到你關閉 Notebook。

# %%
def save_recording(audio):
    """
    處理 Gradio 音訊輸入，儲存為 WAV 檔案。

    Args:
        audio: Gradio Audio 元件的回傳值 (設定 type="numpy" 時為 (sample_rate, data) tuple)。

    Returns:
        儲存的 WAV 檔案路徑。
    """
    if audio is None:
        return "錯誤：未偵測到音訊輸入。"

    sample_rate, data = audio

    # 確保 data 是 NumPy array 且為適當的 dtype (例如 int16)
    if not isinstance(data, np.ndarray):
         return f"錯誤：音訊資料類型不正確 ({type(data)})，應為 NumPy array。"

    # 將音訊資料轉換為 int16 (常見的 WAV 格式)
    # 先檢查最大值以避免 clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data_int16 = (data / max_val * 32767).astype(np.int16)
    else:
        data_int16 = data.astype(np.int16) # 如果是靜音

    # 產生帶有時間戳的檔名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    filepath = Path("./") / filename # 儲存在目前工作目錄

    try:
        # 使用 wavio 儲存 WAV 檔案
        wavio.write(str(filepath), data_int16, sample_rate, sampwidth=2) # sampwidth=2 for 16-bit
        print(f"錄音已儲存至：{filepath}")
        return str(filepath)
    except Exception as e:
        print(f"儲存 WAV 檔案時發生錯誤：{e}")
        return f"儲存錯誤：{e}"

# 建立 Gradio 介面
# inputs: 麥克風錄音，回傳 numpy array (sample_rate, data)
# outputs: 顯示檔案路徑 (gr.File) 和播放器 (gr.Audio)
recorder_app = gr.Interface(
    fn=save_recording,
    inputs=gr.Audio(sources=["microphone"], type="numpy", label="點此錄音 (Click to Record)"),
    outputs=[
        gr.File(label="儲存的 WAV 檔案 (Saved WAV File)"),
        gr.Audio(label="播放錄音 (Playback Recording)")
    ],
    title="簡易錄音機 (Simple Audio Recorder)",
    description="錄製聲音並儲存為 WAV 檔案。錄音後，檔案路徑會顯示在下方，可供下載或複製路徑用於後續步驟。",
    allow_flagging="never"
)

# 啟動 Gradio App
# share=True 會產生公開連結，方便在 Colab/Kaggle 等環境使用
# inline=True 會嘗試在 Notebook 中內嵌顯示 (不一定所有環境都支援)
recorder_app.launch(share=True, inline=False) # 建議 share=True, inline=False