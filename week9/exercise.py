# %% [markdown]
# ## 整合：從輸入到輸出 (課堂練習)
#
# `read_for_me` 函式整合了 S2T、LLM 修正、TTS 聲音複製的完整流程。
# 請依照 `FIXME` 的指示，完成函式中缺少的部分。

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
    """
    整合 S2T -> LLM -> TTS 的流程。
    (課堂練習：完成 FIXME 部分)
    """
    # --- 1. 檢查與取得輸入文字 ---
    if not input_text and not input_audio_path:
        raise ValueError("必須提供 input_text 或 input_audio_path。")
    if input_text and input_audio_path:
        raise ValueError("input_text 和 input_audio_path 不能同時提供。")
    if voice_to_clone_key not in transcripts_dict:
        print(f"警告：在文字稿字典中找不到 key '{voice_to_clone_key}'。")

    original_text = ""
    if input_audio_path:
        print(f"步驟 1: S2T ({input_audio_path})")
        audio_file = Path(input_audio_path)
        if not audio_file.exists(): raise FileNotFoundError(f"找不到音檔: {input_audio_path}")

        # FIXME 1: 語音轉文字
        # 提示：使用 s2t_model 這個 pipeline 來處理聲音檔案。
        #      你需要將聲音檔案的路徑 (字串格式，例如 str(audio_file)) 傳遞給 s2t_model。
        #      s2t_model 會回傳一個字典，辨識出的文字在 'text' 這個 key 裡面。
        #      將辨識出的文字存到 original_text 變數中。
        # s2t_result = s2t_model( ... ) # 填入正確的參數
        # original_text = s2t_result[ ... ] # 取得 'text'
        original_text = "FIXME: 語音轉文字結果應存於此" # <--- 請將這一行取代為你的程式碼

        print(f"  >> S2T 結果: {original_text}")
    else:
        original_text = input_text
        print(f"步驟 1: 使用提供的文字: {original_text}")

    # --- 2. LLM 文法修正 ---
    print("\n步驟 2: LLM 文法修正")
    # 準備 LLM 輸入訊息 (這部分已完成)
    messages = prepare_grammar_correction_messages(original_text)

    # FIXME 2: 呼叫 LLM 進行修正
    # 提示：使用 llm_model 這個 pipeline 來處理 messages。
    #      記得設定 return_full_text=False 只取得模型生成的回應。
    #      也可以設定 max_new_tokens (例如 len(original_text) + 50) 來限制輸出長度。
    #      llm_model 會回傳一個列表，取第一個元素 (索引為 0) 的字典，
    #      其中 'generated_text' key 的值就是修正後的文字。
    #      將修正後的文字存到 corrected_text 變數中，並移除前後多餘的空白。
    # llm_result = llm_model( ... , return_full_text=False, max_new_tokens=...) # 填入 messages 和其他參數
    # corrected_text = llm_result[0][ ... ].strip() # 取得 'generated_text' 並清理
    corrected_text = "FIXME: 文法修正結果應存於此" # <--- 請將這一行取代為你的程式碼

    # 清理常見的模型輸出問題 (例如多餘的引號或標籤)
    corrected_text = corrected_text.replace("```", "").replace("`", "").strip('"').strip()
    print(f"  >> 修正後文字: {corrected_text}")

    # --- 3. TTS 聲音複製 ---
    print(f"\n步驟 3: TTS 聲音複製 (模仿 {voice_to_clone_key})")
    # 取得參考聲音的路徑和文字稿 (這部分已完成)
    # 修正：正確從 transcripts_dict 取得 transcription
    voice_info = transcripts_dict.get(voice_to_clone_key, {})
    voice_ref_text = voice_info.get("transcription", "") # 安全取得文字稿
    voice_ref_path_str = f"./week9/{voice_to_clone_key}"
    voice_ref_path = Path(voice_ref_path_str)

    if not voice_ref_path.exists():
        raise FileNotFoundError(f"找不到參考聲音檔: {voice_ref_path_str}")
    if not voice_ref_text:
         print("(警告：找不到參考聲音文字稿，ref_text 將為空)")

    # FIXME 3: 呼叫聲音複製函式
    # 提示：呼叫我們之前定義的 clone_voice 函式。
    #      需要傳遞以下參數：
    #      - path_to_ref_audio: 參考聲音檔案的路徑 (字串格式，這裡是 voice_ref_path)
    #      - gen_text: 要轉換成語音的文字 (這裡是用 LLM 修正過的 corrected_text)
    #      - ref_text: 參考聲音的文字稿 (這裡是 voice_ref_text)
    #      - output_file: 輸出的檔名 (函式參數 output_file)
    #      - output_dir: 輸出的目錄 (函式參數 output_dir)
    # clone_voice(
    #     path_to_ref_audio=str(voice_ref_path),
    #     gen_text= ... , # 填入修正後的文字
    #     ref_text= ... , # 填入參考文字稿
    #     output_file= ... , # 填入輸出檔名
    #     output_dir= ...   # 填入輸出目錄
    # )
    print("FIXME: 應在此處呼叫 clone_voice 函式") # <--- 請將這一行取代為你的程式碼

    # 修正：補上結尾括號
    print(f"\n流程完成！最終語音預計儲存至 {Path(output_dir) / output_file}")

# %% [markdown]
# ### 執行整合流程 (練習用)
#
# 在完成上面的 `FIXME` 部分後，執行此區塊來測試你的 `read_for_me` 函式。
# 範例 1 使用文字輸入，範例 2 嘗試使用之前錄製的聲音檔。

# %%
# 執行整合流程 - 範例 1: 文字輸入，川普聲音 (依照你的範例修改)
print("--- 執行整合流程：範例 1 (文字輸入) ---")
try:
    # 確保 transcripts 字典已載入
    if not transcripts:
         print("錯誤：transcripts 字典是空的，無法執行。請先確認文字稿檔案已成功載入。")
    else:
        read_for_me(
            s2t_model=s2t,
            llm_model=llm,
            transcripts_dict=transcripts, # 傳入文字稿字典
            input_text="i has a apple.", # 簡單錯誤英文
            voice_to_clone_key="trump.wav", # 改為模仿川普
            output_file="trump_corrected_apple_exercise.wav" # 改個檔名避免覆蓋
        )
except NameError as e:
    print(f"執行範例 1 時發生錯誤：似乎有變數未定義 ({e})。請檢查 FIXME 部分是否都已完成。")
except FileNotFoundError as e:
     print(f"執行範例 1 時發生錯誤：找不到檔案 ({e})。請確認參考聲音檔存在。")
except Exception as e:
    print(f"執行範例 1 時發生預期外的錯誤: {e}")

print("\n" + "="*30 + "\n")

# 執行整合流程 - 範例 2: 聲音輸入 (若存在)，海綿寶寶聲音 (範例2改回海綿寶寶)
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
        # 確保 transcripts 字典已載入
        if not transcripts:
            print("錯誤：transcripts 字典是空的，無法執行。請先確認文字稿檔案已成功載入。")
        else:
            read_for_me(
                s2t_model=s2t,
                llm_model=llm,
                transcripts_dict=transcripts,
                input_audio_path=input_audio_for_test, # 使用錄音檔
                voice_to_clone_key="spongebob.wav",     # 模仿海綿寶寶
                output_file="my_recording_spongebob_voice_exercise.wav" # 改個檔名
            )
    except NameError as e:
        print(f"執行範例 2 時發生錯誤：似乎有變數未定義 ({e})。請檢查 FIXME 部分是否都已完成。")
    except FileNotFoundError as e:
        print(f"執行範例 2 時發生錯誤：找不到檔案 ({e})。請確認錄音檔和參考聲音檔都存在。")
    except Exception as e:
        print(f"執行範例 2 時發生預期外的錯誤: {e}")
else:
    print(f"找不到錄音檔 ({my_recording_wav_path} 或 {my_recording_webm_path})，跳過範例 2。")
    print("請確認已成功錄音並執行儲存步驟。")