import functools
from typing import TypedDict

import numpy.typing as npt
import torch
from transformers import (  # Hugging Face 函式庫
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


@functools.cache
def _prepare_whisper_model(
    device: torch.device, model_id: str = "openai/whisper-large-v3-turbo"
) -> AutomaticSpeechRecognitionPipeline:
    """載入並設定 Whisper S2T 模型 pipeline。"""
    print(f"正在載入 Whisper 模型: {model_id} ...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  # 使用優化的注意力機制
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,  # 處理 30 秒聲音片段
        batch_size=16,  # 可依 GPU 記憶體調整
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Whisper 模型載入完成。")
    return pipe


class ASRDictInput(TypedDict):
    sampling_rate: int
    raw: npt.NDArray


def asr(
    inputs: str | npt.NDArray | bytes,
    device: torch.device,
    *args,
    **kwargs,
) -> str:
    """處理音頻並返回轉錄文本。"""
    model = _prepare_whisper_model(device)
    transcription = model(inputs, *args, **kwargs)

    # Handle different return types from the model
    if isinstance(transcription, dict) and "text" in transcription:
        return transcription["text"]
    else:
        return str(transcription)
