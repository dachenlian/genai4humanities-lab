import textwrap
from pathlib import Path

from PIL import Image

SYSTEM_PROMPT = """\
你是一位友善、有耐心且鼓勵人心的閱讀小夥伴，旨在幫助使用者（特別是孩童）提升他們的朗讀技巧與閱讀理解能力。

**你的核心能力：**

你會根據使用者提供的文本或圖片，與他們互動，回答問題，並提供有助於提升閱讀理解和學習興趣的回饋與說明。你的目標是作為一個互動式的學習輔助工具，促進使用者的閱讀參與和理解。

**互動準則：**

* **語言：** 請始終使用**臺灣正體中文（繁體字）**進行所有溝通。
* **語氣：** 保持友善、耐心、支持、清晰且適合兒童的語氣。
* **內容依據：** 所有的回答與分析都必須嚴格基於使用者提供的文本或圖片內容。切勿臆測或添加外部資訊。
* **格式遵循：** 精確遵守使用者請求中指定的任何輸出格式要求（例如 JSON 結構）。
* **誠實原則：** 若根據提供的資訊無法完成任務或回答問題（例如，圖片資訊不足），請直接且委婉地說明情況。
* **安全互動：** 確保所有回應的內容對兒童來說都是安全、恰當且有益的。

你的目標是成為一個可靠且令人愉快的學習夥伴，輔助使用者享受閱讀並從中學習。
"""


def prepare_reading_feedback_prompt(user_reading: str, correct_reading: str) -> str:
    user_prompt = textwrap.dedent(f"""\
        # 指令：比較語音辨識結果與原文

        請比較以下兩段文字：

        第一段是使用者朗讀的語音辨識（ASR）結果：
        「{user_reading}」

        第二段是正確的原文：
        「{correct_reading}」

        請仔細比對，並提供回饋：
        * 如果兩者完全一致，請回覆：「讀得很好，完全正確！」。
        * 如果語音辨識結果中有任何錯誤（例如：讀錯字、漏字、多字），請明確指出錯誤的地方以及正確的內容。
            例如：「在「{{錯誤的詞}}」這個地方，你讀成了「{{辨識出的詞}}」，正確的應該是「{{正確的詞}}」。」或「你漏讀了「{{漏掉的詞}}」。」或「你多讀了「{{多出的詞}}」。」。

        請以清晰、友善的語氣提供回饋。
    """)
    return user_prompt


def prepare_answer_illustration_question_prompt(
    img_path: str | Path, book_text: str, user_question: str
) -> tuple[Image.Image, str]:
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    image = Image.open(img_path)

    user_prompt = textwrap.dedent(f"""\
        # 指令：根據圖片回答使用者問題

        請仔細觀察提供的圖片和故事內容，並根據圖片中的視覺資訊，回答以下使用者提出的問題。

        故事內容：
        「{book_text}」

        使用者問題：「{user_question}」

        請直接根據圖片內容回答，如果圖片資訊不足以回答問題，請說明無法從圖片中找到答案。
    """)
    return (image, user_prompt)
