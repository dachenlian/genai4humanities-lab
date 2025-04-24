# AI Storytime - How to Upload Your Own Story

This document explains how to upload your own story for the AI Storytime application.

## Story Structure Requirements

Your story must be provided as a JSON file named `story.json` with the following structure:

```json
{
    "title": "Your Story Title",
    "pages": {
        "cover": {
            "text": "",
            "img": "cover.png"
        },
        "1": {
            "text": "Text for page 1",
            "img": "1.png"
        },
        "2": {
            "text": "Text for page 2",
            "img": "2.png"
        }
        // ... additional pages as needed
    }
}
```

## Image Requirements

1. All images must be placed in a folder named `img` at the same level as your `story.json` file
2. Image filenames must match exactly what's referenced in your JSON file
3. Recommended image format is PNG
4. Include a cover image named according to what's specified in your JSON (e.g., "cover.png")
5. For each page, include a corresponding image (e.g., "1.png", "2.png", etc.)

## How to Upload Your Story

1. 建立遵循上述結構的 `story.json` 檔案
2. 創建一個 `img` 資料夾並將所有圖片檔案添加到其中
3. 確保您的資料夾結構如下：
   ```
   ├── story.json
   └── img/
       ├── cover.png
       ├── 1.png
       ├── 2.png
       └── ... (其他頁面圖片)
   ```
4. 壓縮您的檔案，使 `story.json` 和 `img` 資料夾位於 zip 檔案的根層級
5. 解壓縮後，結構應允許直接存取 story.json 和 img 資料夾（無巢狀資料夾）

## How to Upload Your Voice for Cloning

The application allows you to use your own voice for text-to-speech. To upload your voice:

1. Prepare a clear voice recording (WAV format, recommended max 12 seconds)
2. Optionally, create a text file with the exact transcription of your recording
3. Package your files in a .zip with the following structure:
   ```
   ├── voice.wav        # Your voice recording
   └── transcription.txt  # Optional: Transcription of your recording
   ```
4. Upload using the "Upload Voice File (.zip)" button in the application

Notes:
- The transcription file is optional but recommended for best results
- If not provided, the system will generate a transcription, but this will slow down processing
- For optimal voice cloning, speak clearly and use a quiet recording environment

Example of `transcription.txt` content:
```
This is my voice sample for AI Storytime. I'm recording this to create a personalized voice experience when using the application.
```

## 範例

目前的實現包括"大野狼與七隻小羊"作為範例。您可以參考 `data/story/story.json` 檔案中的結構。

## 提示

- Each page's text can include newline characters (`\n`) for paragraph breaks
- Keep image filenames simple and match them exactly in your JSON
- Test your JSON structure with a validator to ensure it's properly formatted
- Make sure all referenced images exist in your folder

---

# AI 故事時間 - 如何上傳您自己的故事

本文件說明如何為 AI 故事時間應用程式上傳您自己的故事。

## 故事結構要求

您的故事必須以名為 `story.json` 的 JSON 檔案提供，結構如下：

```json
{
    "title": "您的故事標題",
    "pages": {
        "cover": {
            "text": "",
            "img": "cover.png"
        },
        "1": {
            "text": "第 1 頁的文字",
            "img": "1.png"
        },
        "2": {
            "text": "第 2 頁的文字",
            "img": "2.png"
        }
        // ... 視需要增加更多頁面
    }
}
```

## 圖片要求

1. 所有圖片必須放在與 `story.json` 檔案同一層級的 `img` 資料夾中
2. 圖片檔案名稱必須與您的 JSON 檔案中引用的名稱完全相符
3. 建議的圖片格式為 PNG
4. 包括一張按照 JSON 中指定命名的封面圖片（例如："cover.png"）
5. 每一頁都需要有對應的圖片（例如："1.png"、"2.png" 等）

## 如何上傳您的故事

1. 建立遵循上述結構的 `story.json` 檔案
2. 創建一個 `img` 資料夾並將所有圖片檔案添加到其中
3. 確保您的資料夾結構如下：
   ```
   ├── story.json
   └── img/
       ├── cover.png
       ├── 1.png
       ├── 2.png
       └── ... (其他頁面圖片)
   ```
4. 壓縮您的檔案，使 `story.json` 和 `img` 資料夾位於 zip 檔案的根層級
5. 解壓縮後，結構應允許直接存取 story.json 和 img 資料夾（無巢狀資料夾）

## 如何上傳您的聲音進行克隆

本應用程式允許您使用自己的聲音進行文字轉語音。要上傳您的聲音：

1. 準備一段清晰的語音錄音（WAV 格式，建議最長 12 秒）
2. 可選擇建立一個包含您錄音精確文字稿的文字檔
3. 將您的檔案打包成具有以下結構的 .zip：
   ```
   ├── voice.wav         # 您的語音錄音
   └── transcription.txt # 選填：您錄音的文字稿
   ```
4. 使用應用程式中的「上傳語音檔案 (.zip)」按鈕進行上傳

注意事項：
- 文字稿檔案是選填的，但建議提供以獲得最佳效果
- 若未提供，系統將自動生成文字稿，但這會減慢處理速度
- 為獲得最佳語音克隆效果，請清晰發音並在安靜的環境中錄音

`transcription.txt` 內容範例：
```
這是我為 AI 故事時間應用準備的語音樣本。我錄製這段內容是為了在使用應用程式時創造個人化的語音體驗。
```

## 範例

目前的實現包括"大野狼與七隻小羊"作為範例。您可以參考 `data/story/story.json` 檔案中的結構。

## 提示

- 每頁的文字可以包括換行符 (`\n`) 以分隔段落
- 保持圖片檔名簡單，並與您的 JSON 中的名稱完全匹配
- 使用驗證器測試您的 JSON 結構，確保格式正確
- 確保您的資料夾中存在所有引用的圖片


## Bugs

* Must upload a story first before doing Voice Chat
* 