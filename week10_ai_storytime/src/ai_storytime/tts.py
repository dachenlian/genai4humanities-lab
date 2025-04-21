import shlex
import subprocess
from pathlib import Path
from typing import Literal


def speak(
    path_to_ref_audio: str,
    gen_text: str,
    ref_text: str = "",  # 參考音檔的文字稿 (選填，但建議提供)
    output_file: str = "tts_output.wav",
    output_dir: str = "tts_output",  # 改用更有意義的預設目錄
    device: Literal["cuda", "cpu"] = "cpu",
):
    """使用 f5-tts 命令列工具進行聲音複製。"""
    cli_executable = "f5-tts_infer-cli"
    model = "F5TTS_v1_Base"
    output_path = Path(output_dir) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出目錄

    command = [
        cli_executable,
        "--model",
        model,
        "--ref_audio",
        str(Path(path_to_ref_audio).resolve()),
        "--ref_text",
        ref_text,
        "--gen_text",
        gen_text,
        "--output_dir",
        str(output_path.parent.resolve()),
        "--output_file",
        output_path.name,
        "--nfe_step",
        "64",  # 推論步數
        "--device",
        device,
    ]
    print(f"準備執行 TTS 命令: {shlex.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        print("\nTTS 命令執行成功。")
        print(f"生成的音訊檔案: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nTTS 命令執行失敗 (錯誤碼 {e.returncode}):\nstderr: {e.stderr}")
    except FileNotFoundError:
        print(f"\n錯誤：找不到命令 '{cli_executable}'。請確認 f5-tts 已安裝。")
    except Exception as e:
        print(f"\n執行 TTS 時發生錯誤: {e}")
