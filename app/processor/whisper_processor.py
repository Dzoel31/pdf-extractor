from pathlib import Path
import json
import os
import time

import whisper
import pydub
import torch

from dotenv import load_dotenv

from app.config import ARTIFACT_DIR, OUTPUT_DIR, TEMP_DIR, TEMP_DIR_MAP
from app.utils.helper import (
    ensure_temp_dir,
    check_json_file_exists,
    log_process,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


class WhisperProcessor:
    """
    Initialize the WhisperProcessor with the specified model and parameters.
    """

    def __init__(
        self,
        model_name: str = "large",
        language: str = "en",
        device: str = "auto",
        overwrite: bool = False,
    ):
        self.model_name = model_name
        self.params = {
            "language": language,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.9,
            "word_timestamps": False,
            "verbose": False,
        }
        self.overwrite = overwrite
        self.output_dir = OUTPUT_DIR / "whisper"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info("Loading Whisper model %s on %s", model_name, self.device)
        self.model = whisper.load_model(
            model_name, device=self.device, download_root=str(ARTIFACT_DIR)
        )
        self._postprocess_llm = None

    def _preprocess_audio(
        self,
        file_path: str | Path,
        format: str = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> str:
        """Preprocess audio file to the specified format and sample rate.

        Args:
            file_path (str | Path): Path to the audio file.
            format (str): Desired audio format (default: "wav").
            sample_rate (int): Desired sample rate (default: 16000).
            channels (int): Number of audio channels (default: 1).
        Returns:
            str: Path to the preprocessed audio file.
        """
        ensure_temp_dir(TEMP_DIR / format)
        input_format = Path(file_path).suffix[1:].lower()
        audio = pydub.AudioSegment.from_file(str(file_path), format=input_format)
        audio = audio.set_frame_rate(sample_rate).set_channels(channels)
        temp_file_path = TEMP_DIR / format / f"temp_audio.{format}"
        audio.export(temp_file_path, format=format)
        return str(temp_file_path)

    def transcribe_audio(self, file_path: str | Path):
        """Transcribe audio file using Whisper model.
        Args:
            file_path (str | Path): Path to the audio file.
        """
        file_path = Path(file_path)
        total_start = time.perf_counter()
        try:
            temp_dir = TEMP_DIR / next(
                key
                for key, value in TEMP_DIR_MAP.items()
                if file_path.suffix.lower().split(".")[-1] in value
            )
            ensure_temp_dir(temp_dir)
            output_file = self.output_dir / f"{file_path.stem}.json"
            ensure_temp_dir(self.output_dir)

            if not self.overwrite and check_json_file_exists(output_file):
                logger.info(
                    f"JSON result already exists at {output_file}. Skipping processing."
                )
                yield log_process(
                    "skip",
                    f"JSON result already exists for {file_path.name}. Skipping processing.",
                )
                return

            yield log_process("info", f"Processing audio file: {file_path.name}")
            processed_audio = self._preprocess_audio(file_path)

            yield log_process("info", f"Transcribing audio file: {file_path.name}")

            transcription = self.model.transcribe(
                processed_audio, **self.params
            )
            segments = transcription.get("segments", [])

            chunked_result = self._group_result(
                segments, chunk_length=60, chunk_unit="minutes"
            )

            yield log_process(
                "info", f"Post-processing transcription for {file_path.name}"
            )
            # Post-process the transcribed text
            if self._can_use_postprocess_llm():
                for segment in chunked_result["content"]:
                    segment["content"] = self._text_postprocess(segment["content"])
            else:
                logger.info(
                    "Skipping Gemini transcript post-processing because credentials are not configured."
                )

            chunked_result["total_time"] = round(time.perf_counter() - total_start, 2)

            with open(output_file, "w+", encoding="utf-8") as f:
                json.dump(chunked_result, f, ensure_ascii=False, indent=4)

            yield log_process(
                "success",
                f"Transcription completed for {file_path.name} in {chunked_result['total_time']:.2f} seconds",
            )

        except StopIteration:
            raise ValueError(f"Unsupported file type for {file_path.name}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {file_path} does not exist.")

    def _group_result(
        self, segments, chunk_length: int = 60, chunk_unit: str = "seconds"
    ):
        """
        Group transcription segments by time.

        Args:
            segments (str | list): Transcription segments.
            chunk_length (int): Length of each chunk.
            chunk_unit (str): Unit for chunk_length, either "seconds", "minutes", or "hours".
        Returns:
            dict: Grouped transcription segments.
        """
        unit_multipliers = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
        }
        multiplier = unit_multipliers.get(chunk_unit.lower())
        if not multiplier:
            raise ValueError("chunk_unit must be 'seconds', 'minutes', or 'hours'.")

        chunk_length_sec = chunk_length * multiplier

        result_json = {
            "content": [],
            "total_segments": 0,
        }

        if not segments:
            return result_json

        max_end = max(segment["end"] for segment in segments)
        start = 0

        while start < max_end:
            end = start + chunk_length_sec
            actual_end = min(end, max_end)
            texts = []
            for segment in segments:
                seg_start = segment["start"]
                seg_end = segment["end"]
                if seg_start >= start and seg_end <= end:
                    texts.append(segment["text"].strip())
            if texts:
                result_json["content"].append(
                    {
                        "segment": len(result_json["content"]),
                        "start_time": start,
                        "end_time": actual_end,
                        "duration": round(actual_end - start, 2),
                        "content": " ".join(texts),
                    }
                )
                result_json["total_segments"] += 1
            start = end
        return result_json

    def _text_postprocess(self, text: str) -> str:
        """Post-process the transcribed text."""
        if self._postprocess_llm is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self._postprocess_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
            )
        message = HumanMessage(
            content=f"Perbaiki transkrip ini ke dalam bahasa Indonesia baku dan jelas. Berikan hasilnya langsung, tanpa pembuka, penjelasan, atau penutup. \n {text}"
        )

        response = self._postprocess_llm.invoke([message])
        response_text = response.content.strip()

        return response_text

    @staticmethod
    def _can_use_postprocess_llm() -> bool:
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


if __name__ == "__main__":
    list_model = whisper.available_models()
    print("Available Whisper models:")
    for model in list_model:
        print(f"- {model}")
