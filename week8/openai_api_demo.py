# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2 # Or your version
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 使用 Python 呼叫 OpenAI API (兩種方法比較)
#
# 大家好！這份筆記本將帶大家學習如何用 Python 程式來和 OpenAI 的大型語言模型 (像 ChatGPT 背後的模型) 溝通。我們會介紹兩種主要的方法：
#
# 1.  **直接使用 `requests` 函式庫：** 這種方法能讓我們看清楚程式和 OpenAI 伺服器之間是怎麼透過網路溝通的 (HTTP 請求/回應)。
# 2.  **使用 `openai` 官方函式庫 (SDK)：** 這是 OpenAI 官方推薦的方式，語法更簡單，像是請一個專門的助手幫你處理跟 OpenAI 溝通的細節。
#
# 我們會用同一個問題來試試這兩種方法，看看它們的程式碼有什麼不同。

# %% [markdown]
# ## 步驟 1：設定與取得 OpenAI API 金鑰
#
# 不管用哪種方法，都需要一個 OpenAI 的「API 金鑰」(API Key)。這就像是你跟 OpenAI 服務溝通時的秘密通行證。
#
# **⚠️ 保護好你的 API 金鑰！⚠️**
#
# 千萬不要把金鑰直接寫在程式碼裡面，尤其是如果你會把程式碼分享給別人或上傳到網路。
#
# **在 Google Colab 中 (建議這樣做):**
#
# 1.  在 Colab 介面左邊找到一個 **鑰匙 🔑 圖示** (Secrets)。
# 2.  點進去，然後按「+ 新增密鑰」(Add a new secret)。
# 3.  **名稱 (Name)** 的地方輸入：`OPENAI_API_KEY` (大小寫要完全一樣)。
# 4.  **值 (Value)** 的地方貼上你從 OpenAI 網站申請到的金鑰 (通常是 `sk-` 開頭)。
# 5.  記得**打開**「允許筆記本存取」(Notebook access) 這個開關。
# 6.  完成後，下面的程式碼就能安全地讀取你的金鑰了。
#
# **在自己的電腦上或其他環境 (替代方法):**
#
# 1.  如果不是在 Colab，可以用 `python-dotenv` 這個工具。
# 2.  先安裝： 在你的終端機 (Terminal) 或命令提示字元 (Command Prompt) 輸入 `pip install python-dotenv`。
# 3.  在你的 Python 程式檔案所在的資料夾，建立一個叫做 `.env` 的文字檔。
# 4.  打開 `.env` 檔案，在裡面寫一行：`OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxx'` (把 `sk-...` 換成你自己的金鑰)。
# 5.  (進階) 如果你用 Git，記得把 `.env` 加到 `.gitignore` 檔案裡，避免不小心上傳金鑰。
# 6.  下面的程式碼如果偵測到不是在 Colab，就會試著從 `.env` 檔案讀取金鑰。

# %%
# 載入需要的工具
import os
import json # 用來處理和顯示 JSON 資料格式
import sys # 用來檢查是不是在 Colab

# --- 讀取 API 金鑰 ---
api_key = None # 先假設還沒有讀到金鑰
print("🚀 正在偵測環境並讀取 OpenAI API 金鑰...")

# 判斷是否在 Colab 環境
try:
    from google.colab import userdata
    print("✅ 在 Google Colab 環境中。")
    print("   嘗試從 Colab Secrets 讀取 'OPENAI_API_KEY'...")
    api_key = userdata.get("OPENAI_API_KEY")
    if api_key:
        print("   🔑 成功從 Colab Secrets 讀取金鑰！")
    else:
        # Colab 有找到，但裡面沒有這個名字的 secret
        print("   ⚠️ 警告：在 Colab Secrets 中找不到名為 'OPENAI_API_KEY' 的密鑰。")
        print("       請確認你在 Colab Secrets 設定中已新增此密鑰，並啟用了 Notebook access。")

except ImportError:
    # 不能 import google.colab，表示不在 Colab 環境
    print("✅ 在本地或其他非 Colab 環境中。")
    print("   嘗試從 .env 檔案或環境變數讀取 'OPENAI_API_KEY'...")
    try:
        # 試著用 dotenv 讀取 .env 檔案
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("   🔑 成功從 .env 檔案或環境變數讀取金鑰！")
        else:
            print("   ⚠️ 警告：在 .env 檔案或環境變數中找不到 'OPENAI_API_KEY'。")
            print("       請確認你的 .env 檔案存在且格式正確，或已設定環境變數。")
    except ImportError:
        # 如果連 dotenv 都沒有安裝
        print("   ⚠️ 警告：未安裝 python-dotenv。建議安裝 (`pip install python-dotenv`) 以便從 .env 讀取。")
        print("       現在嘗試直接從環境變數讀取...")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("   🔑 成功從環境變數讀取金鑰！")
        else:
            print("   ⚠️ 警告：也無法直接從環境變數讀取 'OPENAI_API_KEY'。")


# --- 最後檢查 ---
if not api_key:
    print("\n❌ 錯誤：未能成功獲取 OpenAI API 金鑰。")
    print("   後續的 API 請求將無法成功。請依照上面的指示設定好金鑰。")
else:
    print("\n👍 API 金鑰已準備就緒，可以繼續進行 API 呼叫。")

# %% [markdown]
# ---
# ## 方法一：使用 `requests` 函式庫直接呼叫
#
# 這種方法就像是我們自己打電話給 OpenAI 伺服器。我們需要準備好所有溝通細節，包括：
# * 伺服器的地址 (URL)
# * 我們的身份證明 (API Key，放在叫做 "Headers" 的地方)
# * 我們要傳達的訊息 (我們的問題/提示，放在叫做 "Payload" 或 "Data" 的地方，通常是 JSON 格式)
# * 溝通的方式 (用 `POST` 方法發送請求)

# %%
# 載入 requests 函式庫 (如果上面沒載入的話)
import requests

print("--- 方法一：使用 requests ---")

# 檢查是否有 API Key，沒有就跳過
if not api_key:
    print("❌ 缺少 API 金鑰，跳過 requests 方法。")
else:
    # 1. 設定 OpenAI API 的地址 (Chat Completions 端點)
    api_url = "https://api.openai.com/v1/chat/completions"

    # 2. 準備請求標頭 (Headers)，包含我們的 API Key
    #    Authorization 裡面的 "Bearer" 是 OpenAI 規定的格式
    headers = {
        "Content-Type": "application/json", # 告訴伺服器我們送的是 JSON
        "Authorization": f"Bearer {api_key}" # 放入我們的金鑰
    }

    # 3. 準備要傳送的資料 (Payload)，使用 Python 字典表示
    payload = {
        "model": "gpt-3.5-turbo", # 指定要使用的 AI 模型
        "messages": [
            # messages 是一個列表，包含對話紀錄
            # "role": "system" 是給 AI 的系統指示 (可選)
            {"role": "system", "content": "你是一位有幫助的助理，擅長用簡單易懂的方式解釋複雜概念。請使用台灣人習慣的正體中文回答。"},
            # "role": "user" 是我們使用者提出的問題/提示
            {"role": "user", "content": "用一個簡單的比喻，解釋什麼是 API？"}
        ],
        "temperature": 0.7 # 控制 AI 回答的創意程度 (0 最固定，1 最隨機)
    }

    # 4. 發送 POST 請求
    print("   正在發送請求至 OpenAI...")
    response_text_requests = "" # 先準備一個空字串放結果
    try:
        # requests.post(網址, 標頭=我們的標頭, 資料=我們的資料)
        # json=payload 會自動把 Python 字典轉成 JSON 格式送出
        response = requests.post(api_url, headers=headers, json=payload)

        # 檢查回應狀態碼，如果不是 200 OK 或類似成功碼，就產生錯誤
        response.raise_for_status() # 這行會檢查錯誤，若有錯會跳到 except

        # 5. 解析伺服器回應的 JSON 資料
        result_data = response.json() # 把收到的 JSON 資料轉成 Python 字典
        print("   ✅ 請求成功！")

        # 6. 從回應中提取 AI 的回答
        #    需要按照 OpenAI 回應的固定格式一層層找： choices -> 第 0 個元素 -> message -> content
        message_content = result_data['choices'][0]['message']['content']
        response_text_requests = message_content.strip() # 去掉頭尾多餘空白

        # (除錯用) 如果想看完整的 OpenAI 回應可以取消下面這行的註解
        # print("\n   伺服器完整回應 (JSON):\n", json.dumps(result_data, indent=2, ensure_ascii=False))

    except requests.exceptions.RequestException as e:
        # 處理網路請求相關的錯誤 (連線失敗、超時、伺服器錯誤等)
        print(f"   ❌ 請求失敗：{e}")
        # 如果有收到伺服器的錯誤回應，也把它印出來
        if e.response is not None:
            print(f"      伺服器回應 ({e.response.status_code}): {e.response.text}")
        response_text_requests = f"請求錯誤: {e}"
    except KeyError as e:
        # 處理從 JSON 回應中提取資料失敗的錯誤
        print(f"   ❌ 解析回應失敗：找不到預期的欄位 {e}")
        print(f"      收到的資料：{result_data}")
        response_text_requests = f"解析回應錯誤: {e}"
    except Exception as e:
        # 處理其他未預期的錯誤
        print(f"   ❌ 發生未預期的錯誤：{e}")
        response_text_requests = f"其他錯誤: {e}"

    # 7. 顯示結果
    print("\n   --- requests 方法得到的回應 ---")
    print(response_text_requests)

# %% [markdown]
# **`requests` 方法回顧：**
#
# 我們一步步準備了 URL、Headers、Payload，然後用 `requests.post` 發送出去，再解析回來的 JSON 資料。這讓我們清楚看到整個 HTTP 互動過程。

# %% [markdown]
# ---
# ## 方法二：使用 `openai` 官方函式庫 (SDK)
#
# 這個方法就像是請了一位專門跟 OpenAI 打交道的助理 (SDK)。我們只需要告訴助理我們要用哪個模型、我們的問題是什麼，助理就會幫我們處理好所有跟伺服器溝通的細節。

# %%
# 載入 openai 函式庫 (如果上面沒載入的話)
# from openai import OpenAI, OpenAIError # 確保已導入

print("--- 方法二：使用 openai SDK ---")

# 檢查是否有 API Key，沒有就跳過
if not api_key:
    print("❌ 缺少 API 金鑰，跳過 openai SDK 方法。")
else:
    # 1. 初始化 OpenAI Client (我們的助理)
    #    需要提供 API Key 給它
    print("   正在初始化 OpenAI Client...")
    response_text_sdk = "" # 準備放結果的字串
    try:
        client = OpenAI(api_key=api_key)

        # 2. 準備要問的問題 (messages 列表，跟 requests 方法一樣)
        messages_for_sdk = [
            {"role": "system", "content": "你是一位有幫助的助理，擅長用簡單易懂的方式解釋複雜概念。請使用台灣人習慣的正體中文回答。"},
            {"role": "user", "content": "用一個簡單的比喻，解釋什麼是 API？"} # 跟上面用一樣的問題
        ]

        # 3. 呼叫助理 (Client) 的功能來發送請求
        print("   正在發送請求至 OpenAI (透過 SDK)...")
        # client.chat.completions.create(...) 是 SDK 提供的方法
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 指定模型
            messages=messages_for_sdk, # 傳入我們的問題
            temperature=0.7 # 其他參數也可以加在這裡
        )

        print("   ✅ 請求成功！")

        # 4. 從回應物件中提取 AI 的回答
        #    SDK 會把回應整理成一個物件，取用內容更方便
        message_content = response.choices[0].message.content
        response_text_sdk = message_content.strip() # 去掉頭尾空白

        # (除錯用) 如果想看完整的 SDK 回應物件可以取消下面這行的註解
        # print("\n   SDK 完整回應物件:\n", response)

    except OpenAIError as e:
        # 處理 OpenAI API 特有的錯誤 (例如金鑰錯誤、額度問題等)
        print(f"   ❌ OpenAI API 錯誤：{e}")
        response_text_sdk = f"OpenAI API 錯誤: {e}"
    except Exception as e:
        # 處理其他未預期的錯誤
        print(f"   ❌ 發生未預期的錯誤：{e}")
        response_text_sdk = f"其他錯誤: {e}"

    # 5. 顯示結果
    print("\n   --- openai SDK 方法得到的回應 ---")
    print(response_text_sdk)


# %% [markdown]
# **`openai` SDK 方法回顧：**
#
# 使用 SDK，我們只需要準備好要問的問題 (`messages`)，然後呼叫 `client.chat.completions.create()` 這個方法，SDK 就會幫我們處理好大部分的溝通細節，程式碼看起來簡潔很多。

# %% [markdown]
# ---
# ## 練習時間 (Practice Time)
#
# 現在輪到你來試試看！請完成下面的練習，填入 `FIXME` 的部分。

# %% [markdown]
# ### 練習一：使用 `requests` - 設定標頭與資料
#
# 請修改下方的程式碼，填入 `FIXME` 的部分：
# 1.  在 `headers` 字典中，完成 `Authorization` 的值，使用我們之前讀取的 `api_key` 變數。
# 2.  在 `payload` 字典中，填入你想使用的 OpenAI 模型名稱 (例如 `"gpt-3.5-turbo"` 或其他)。
# 3.  在 `payload` 字典的 `messages` 列表中，填入你想問 `user` 的問題 `content`。

# %%
# --- 練習一 ---
print("--- 練習一：requests 設定 ---")

if not api_key:
    print("❌ 缺少 API 金鑰，無法進行練習一。")
else:
    # --- 請在這裡修改 ---
    practice_api_url = "https://api.openai.com/v1/chat/completions"

    practice_headers = {
        "Content-Type": "application/json",
        # FIXME 1: 在 Bearer 後面加上正確的 api_key 變數
        "Authorization": f"Bearer {FIXME}"
    }

    practice_payload = {
        # FIXME 2: 填入你想使用的模型名稱 (字串)
        "model": FIXME,
        "messages": [
            {"role": "system", "content": "你是一位樂於助人的助理。"},
            # FIXME 3: 填入你想問的問題 (字串)
            {"role": "user", "content": FIXME}
        ]
        # "temperature": 0.7 # 這個可以保留或刪除
    }
    # --- 修改結束 ---

    # 檢查你的修改
    print("檢查 Headers Authorization:", practice_headers.get("Authorization", "未設定"))
    print("檢查 Payload Model:", practice_payload.get("model", "未設定"))
    print("檢查 Payload User Content:", practice_payload.get("messages", [{}])[-1].get("content", "未設定"))

    # (這裡是發送請求的程式碼，暫時註解掉，避免實際花費 token)
    # print("\n嘗試發送請求 (實際執行會消耗 token)...")
    # try:
    #     response = requests.post(practice_api_url, headers=practice_headers, json=practice_payload)
    #     response.raise_for_status()
    #     result = response.json()
    #     print("請求似乎成功了！")
    #     # print("回應:", json.dumps(result, indent=2, ensure_ascii=False))
    # except Exception as e:
    #     print(f"請求時發生錯誤: {e}")

# %% [markdown]
# ### 練習二：使用 `requests` - 解析回應
#
# 假設你已經成功用 `requests` 發送請求，並且收到了存在 `result_data` 變數中的 JSON 回應 (已轉換成 Python 字典)。這個字典的結構如下：
#
# ```python
# result_data = {
#   "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
#   "object": "chat.completion",
#   "created": 1712695553,
#   "model": "gpt-3.5-turbo-0125",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "API 就像是餐廳裡的服務生。你 (程式) 透過服務生 (API) 跟廚房 (另一項服務) 點餐 (提出請求)，然後服務生再把做好的餐點 (回應) 送回來給你。"
#       },
#       "logprobs": None,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 43,
#     "completion_tokens": 85,
#     "total_tokens": 128
#   },
#   "system_fingerprint": "fp_xxxxxxxxxx"
# }
# ```
#
# 請在下面的程式碼中，填入 `FIXME` 的部分，以正確取出 `assistant` 回答的 `content` 內容。
# (提示：你需要使用中括號 `[]` 來存取字典的鍵 (key) 和列表的索引 (index)。記得列表的第一個元素索引是 `0`。)

# %%
# --- 練習二 ---
print("\n--- 練習二：requests 解析回應 ---")

# 假設這是收到的回應資料
result_data = {
  "id": "chatcmpl-mockid",
  "object": "chat.completion",
  "created": 1712695553,
  "model": "gpt-3.5-turbo-0125",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "API 就像是餐廳裡的服務生。你 (程式) 透過服務生 (API) 跟廚房 (另一項服務) 點餐 (提出請求)，然後服務生再把做好的餐點 (回應) 送回來給你。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"total_tokens": 128}
}

extracted_content = "尚未提取" # 預設值
try:
    # --- 請在這裡修改 ---
    # FIXME: 逐步填入正確的鍵(key)和索引(index)來取得 'content'
    extracted_content = result_data[FIXME][FIXME][FIXME][FIXME]
    # --- 修改結束 ---

    print("提取到的內容:", extracted_content)
except Exception as e:
    print(f"提取時發生錯誤: {e}，請檢查你的 FIXME 填寫是否正確。")

# %% [markdown]
# ### 練習三：使用 `openai` SDK - 初始化與呼叫
#
# 請修改下方的程式碼，填入 `FIXME` 的部分：
# 1.  使用 `OpenAI()` **初始化 (initialize)** OpenAI 的 client 物件，記得傳入 `api_key`。
# 2.  在 `client.chat.completions.create()` 的呼叫中，填入 `model` 參數的值。
# 3.  在 `client.chat.completions.create()` 的呼叫中，填入 `messages` 參數的值 (使用提供的 `messages_for_exercise` 變數)。

# %%
# --- 練習三 ---
print("\n--- 練習三：openai SDK 初始化與呼叫 ---")

if not api_key:
    print("❌ 缺少 API 金鑰，無法進行練習三。")
else:
    # 準備要問的問題
    messages_for_exercise = [
        {"role": "system", "content": "請簡短回答。"},
        {"role": "user", "content": "用一句話說明雲端運算(Cloud Computing)是什麼？"}
    ]
    model_to_use = "gpt-3.5-turbo"

    try:
        # --- 請在這裡修改 ---
        # FIXME 1: 初始化 OpenAI client，傳入 api_key
        client = FIXME(api_key=api_key)

        print("   正在準備發送請求 (透過 SDK)...")
        # FIXME 2: 填入 model 參數的值 (使用 model_to_use 變數)
        # FIXME 3: 填入 messages 參數的值 (使用 messages_for_exercise 變數)
        response = client.chat.completions.create(
            model=FIXME,
            messages=FIXME
        )
        # --- 修改結束 ---

        print("   ✅ 請求似乎成功了！")
        # (這裡是提取回應的程式碼，暫時註解掉)
        # message_content = response.choices[0].message.content
        # print("   回應內容:", message_content.strip())

    except Exception as e:
        print(f"   ❌ 發生錯誤: {e}，請檢查你的 FIXME 填寫。")

# %% [markdown]
# ### 練習四：使用 `openai` SDK - 解析回應
#
# 假設你已經成功用 `openai` SDK 發送請求，並且收到了存在 `response` 變數中的 **回應物件**。這個物件的結構大致如下 (簡化表示)：
#
# ```python
# # response 物件 (不是字典，是 SDK 定義的物件)
# response.choices[0].message.content = "雲端運算就是透過網路使用遠端伺服器提供的計算資源、儲存空間和應用程式。"
# # 其他屬性如 response.id, response.model 等等...
# ```
#
# 請在下面的程式碼中，填入 `FIXME` 的部分，以正確取出 `assistant` 回答的 `content` 內容。
# (提示：SDK 的回應是一個物件，你需要使用 `.` 來存取它的屬性，以及 `[]` 來存取列表的元素。)

# %%
# --- 練習四 ---
print("\n--- 練習四：openai SDK 解析回應 ---")

# 假設這是收到的回應物件 (我們用一個簡單的模擬物件代替)
class MockMessage:
    def __init__(self, content):
        self.content = content
class MockChoice:
    def __init__(self, message_content):
        self.message = MockMessage(message_content)
class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

response = MockResponse("雲端運算就是透過網路使用遠端伺服器提供的計算資源、儲存空間和應用程式。")

extracted_content_sdk = "尚未提取" # 預設值
try:
    # --- 請在這裡修改 ---
    # FIXME: 逐步填入正確的屬性(attribute)和索引(index)來取得 content
    # 提示: response -> choices (列表) -> 第0個元素 -> message (物件) -> content (屬性)
    extracted_content_sdk = response.FIXME[FIXME].FIXME.FIXME
    # --- 修改結束 ---

    print("提取到的內容:", extracted_content_sdk)
except Exception as e:
    print(f"提取時發生錯誤: {e}，請檢查你的 FIXME 填寫是否正確。")


# %% [markdown]
# ---
# ## 練習解答 (Exercise Answers)

# %% [markdown]
# ### 練習一：解答
#
# ```python
# # --- 練習一：解答 ---
# print("--- 練習一：requests 設定 (解答) ---")
#
# if not api_key:
#     print("❌ 缺少 API 金鑰，無法進行練習一。")
# else:
#     practice_api_url = "[https://api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions)"
#
#     practice_headers = {
#         "Content-Type": "application/json",
#         # FIXME 1: 在 Bearer 後面加上正確的 api_key 變數
#         "Authorization": f"Bearer {api_key}" # <--- 解答
#     }
#
#     practice_payload = {
#         # FIXME 2: 填入你想使用的模型名稱 (字串)
#         "model": "gpt-3.5-turbo", # <--- 解答 (或其他有效模型)
#         "messages": [
#             {"role": "system", "content": "你是一位樂於助人的助理。"},
#             # FIXME 3: 填入你想問的問題 (字串)
#             {"role": "user", "content": "台北 101 有多高？"} # <--- 解答 (或其他問題)
#         ]
#     }
#
#     print("檢查 Headers Authorization:", practice_headers.get("Authorization", "未設定"))
#     print("檢查 Payload Model:", practice_payload.get("model", "未設定"))
#     print("檢查 Payload User Content:", practice_payload.get("messages", [{}])[-1].get("content", "未設定"))
# ```

# %% [markdown]
# ### 練習二：解答
#
# ```python
# # --- 練習二：解答 ---
# print("\n--- 練習二：requests 解析回應 (解答) ---")
#
# result_data = {
#   "id": "chatcmpl-mockid", "object": "chat.completion", "created": 1712695553,
#   "model": "gpt-3.5-turbo-0125",
#   "choices": [ { "index": 0, "message": { "role": "assistant", "content": "API 就像是餐廳裡的服務生。你 (程式) 透過服務生 (API) 跟廚房 (另一項服務) 點餐 (提出請求)，然後服務生再把做好的餐點 (回應) 送回來給你。" }, "finish_reason": "stop" } ],
#   "usage": {"total_tokens": 128}
# }
#
# extracted_content = "尚未提取"
# try:
#     # FIXME: 逐步填入正確的鍵(key)和索引(index)來取得 'content'
#     extracted_content = result_data['choices'][0]['message']['content'] # <--- 解答
#
#     print("提取到的內容:", extracted_content)
# except Exception as e:
#     print(f"提取時發生錯誤: {e}")
# ```

# %% [markdown]
# ### 練習三：解答
#
# ```python
# # --- 練習三：解答 ---
# print("\n--- 練習三：openai SDK 初始化與呼叫 (解答) ---")
#
# if not api_key:
#     print("❌ 缺少 API 金鑰，無法進行練習三。")
# else:
#     messages_for_exercise = [
#         {"role": "system", "content": "請簡短回答。"},
#         {"role": "user", "content": "用一句話說明雲端運算(Cloud Computing)是什麼？"}
#     ]
#     model_to_use = "gpt-3.5-turbo"
#
#     try:
#         # FIXME 1: 初始化 OpenAI client，傳入 api_key
#         client = OpenAI(api_key=api_key) # <--- 解答
#
#         print("   正在準備發送請求 (透過 SDK)...")
#         # FIXME 2: 填入 model 參數的值 (使用 model_to_use 變數)
#         # FIXME 3: 填入 messages 參數的值 (使用 messages_for_exercise 變數)
#         response = client.chat.completions.create(
#             model=model_to_use, # <--- 解答
#             messages=messages_for_exercise # <--- 解答
#         )
#
#         print("   ✅ 請求似乎成功了！")
#         # message_content = response.choices[0].message.content
#         # print("   回應內容:", message_content.strip())
#
#     except Exception as e:
#         print(f"   ❌ 發生錯誤: {e}")
# ```

# %% [markdown]
# ### 練習四：解答
#
# ```python
# # --- 練習四：解答 ---
# print("\n--- 練習四：openai SDK 解析回應 (解答) ---")
#
# class MockMessage:
#     def __init__(self, content): self.content = content
# class MockChoice:
#     def __init__(self, message_content): self.message = MockMessage(message_content)
# class MockResponse:
#     def __init__(self, content): self.choices = [MockChoice(content)]
#
# response = MockResponse("雲端運算就是透過網路使用遠端伺服器提供的計算資源、儲存空間和應用程式。")
#
# extracted_content_sdk = "尚未提取"
# try:
#     # FIXME: 逐步填入正確的屬性(attribute)和索引(index)來取得 content
#     extracted_content_sdk = response.choices[0].message.content # <--- 解答
#
#     print("提取到的內容:", extracted_content_sdk)
# except Exception as e:
#     print(f"提取時發生錯誤: {e}")
# ```

# %% [markdown]
# ---
# ## 結論與比較 (Conclusion & Comparison)
#
# （結論部分與之前的版本相同）
#
# 我們看到了兩種呼叫 OpenAI API 的方法：
#
# * **`requests`：** 像手排車，需要自己處理換檔、離合器 (HTTP 細節)，但能讓你完全掌握過程，也適用於任何廠牌的車 (任何 API)。
# * **`openai` SDK：** 像自排車或有專屬司機，你只要設定目的地 (傳入參數)，它就幫你開到好 (處理好 API 呼叫)，但這司機只開 OpenAI 這家公司的車 (只適用 OpenAI API)。
#
# **對於日常使用 OpenAI API，官方的 `openai` SDK 通常是更方便、更推薦的選擇。**
#
# 但了解 `requests` 的用法仍然很有價值，它可以幫助你：
# * 更深入理解網路 API 是如何運作的。
# * 當你需要跟 OpenAI 以外的其他網路服務互動時，知道該怎麼做。
# * 在 SDK 出問題時，有能力自己檢查或嘗試更底層的呼叫。
#