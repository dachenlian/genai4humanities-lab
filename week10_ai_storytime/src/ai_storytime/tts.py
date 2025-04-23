import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Literal


def tts(
    path_to_ref_audio: str,
    gen_text: str,
    ref_text: str = "",
    output_dir: str = "tts_output",  # Directory to save generated audio
    device: Literal["cuda", "cpu"] = "cpu",
) -> str | None:  # Added return type hint
    """使用 f5-tts 命令列工具進行聲音複製，並返回生成檔案的路徑。"""
    cli_executable = "f5-tts_infer-cli"
    model = "F5TTS_v1_Base"

    # Ensure output directory exists
    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique filename for the output
    output_filename = f"tts_{uuid.uuid4()}.wav"
    output_filepath = output_path_dir / output_filename

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
        str(output_path_dir.resolve()),
        "--output_file",
        output_filename,  # Use unique filename
        "--nfe_step",
        "64",
        "--device",
        device,
    ]
    print("-" * 20)
    print(f"準備執行 TTS 命令: {shlex.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        print(f"TTS 命令執行成功: 標準輸出:\n{result.stdout}")
        print(f"生成的音訊檔案: {output_filepath}")
        print("-" * 20)
        return str(output_filepath)  # Return the path on success
    except subprocess.CalledProcessError as e:
        print(f"\nTTS 命令執行失敗 (錯誤碼 {e.returncode}):\nstderr:\n{e.stderr}")
        print("-" * 20)
        return None  # Return None on failure
    except FileNotFoundError:
        print(
            f"\n錯誤：找不到命令 '{cli_executable}'。請確認 f5-tts 已安裝且在 PATH 中。"
        )
        print("-" * 20)
        return None
    except Exception as e:
        print(f"\n執行 TTS 時發生未預期的錯誤: {e}")
        print("-" * 20)
        return None
