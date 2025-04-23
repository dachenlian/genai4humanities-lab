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

## How to Upload

1. Create your `story.json` file following the structure above
2. Create an `img` folder and add all your image files to it
3. Ensure your folder structure looks like this:
   ```
   ├── story.json
   └── img/
       ├── cover.png
       ├── 1.png
       ├── 2.png
       └── ... (other page images)
   ```
4. Zip your files such that `story.json` and `img` folder are at the root level of the zip file
5. When unzipped, the structure should allow immediate access to story.json and the img folder (no nested folders)

## Example

The current implementation includes "大野狼與七隻小羊" (The Wolf and the Seven Little Goats) as an example. You can reference its structure in the `data/story/story.json` file.

## Tips

- Each page's text can include newline characters (`\n`) for paragraph breaks
- Keep image filenames simple and match them exactly in your JSON
- Test your JSON structure with a validator to ensure it's properly formatted
- Make sure all referenced images exist in your folder

For any questions or assistance, please contact the project maintainers.

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

## 如何上傳

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

## 範例

目前的實現包括"大野狼與七隻小羊"作為範例。您可以參考 `data/story/story.json` 檔案中的結構。

## 提示

- 每頁的文字可以包括換行符 (`\n`) 以分隔段落
- 保持圖片檔名簡單，並與您的 JSON 中的名稱完全匹配
- 使用驗證器測試您的 JSON 結構，確保格式正確
- 確保您的資料夾中存在所有引用的圖片

如有任何問題或需要協助，請聯繫專案維護人員。