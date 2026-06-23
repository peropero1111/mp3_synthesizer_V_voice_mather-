"""Gemini API로 MP3와 사용자가 보유한 Synthesizer V 보이스를 연결한다."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from voices_data import voices

try:
    from google import genai
    from google.genai import types
except ImportError:  
    genai = None
    types = None


DEFAULT_MODEL = "gemini-2.5-flash"
AQ_KEY_PREFIX = "AQ."
MAX_INLINE_AUDIO_BYTES = 18 * 1024 * 1024


class RecommendationError(Exception):
    """사용자에게 보여 줄 수 있는 추천 과정의 오류."""


class GeminiRecommender:
    """MP3를 분석해 사용자가 선택한 보이스 중 하나를 추천한다."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        available_voices: Iterable[str] | None = None,
    ) -> None:
        if genai is None or types is None:
            raise RecommendationError(
                "Gemini 라이브러리가 없습니다. 터미널에서 "
                "`pip install -U google-genai`를 먼저 실행하세요."
            )

        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        if not self.api_key:
            raise RecommendationError(
                "GEMINI_API_KEY 환경변수가 설정되지 않았습니다. "
                "API 키를 코드에 직접 적지 말고 환경변수로 설정하세요."
            )

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.available_voices = self._validate_available_voices(available_voices)
        self.available_voice_set = frozenset(self.available_voices)
        # AQ. 키는 Files API 업로드 대신 오디오를 직접 전송한다. API 키 자체는
        # Gemini Developer API의 x-goog-api-key 인증으로 사용한다.
        self.uses_aq_inline_audio = self.api_key.startswith(AQ_KEY_PREFIX)
        self.client = genai.Client(api_key=self.api_key)
        self.catalog = self._make_catalog(self.available_voices)

    @property
    def connection_mode(self) -> str:
        if self.uses_aq_inline_audio:
            return "Gemini Developer API (AQ. 키 · MP3 직접 전송)"
        return "Gemini Developer API (Files API)"

    @staticmethod
    def _validate_available_voices(
        available_voices: Iterable[str] | None,
    ) -> tuple[str, ...]:
        if available_voices is None:
            return tuple(voices)

        selected = set(available_voices)
        if not selected:
            raise RecommendationError("보유한 보이스를 최소 한 개 선택해 주세요.")

        unknown = selected.difference(voices)
        if unknown:
            raise RecommendationError("목록에 없는 보이스가 선택되었습니다.")

        return tuple(name for name in voices if name in selected)

    @staticmethod
    def _make_catalog(voice_names: Iterable[str]) -> str:
        return "\n".join(
            f"- 이름: {name}\n  특징: {voices[name]}" for name in voice_names
        )

    def recommend(self, mp3_path: str | Path) -> dict[str, Any]:
        """MP3를 분석한 뒤, 사용자가 보유한 보이스 안에서 추천 결과를 반환한다."""
        path = Path(mp3_path)
        self._validate_audio_file(path)

        try:
            if self.uses_aq_inline_audio:
                response = self._generate_with_inline_audio(path)
            else:
                response = self._generate_with_file_api(path)
            return self._validate_response(self._read_json(response))
        except RecommendationError:
            raise
        except Exception as error:
            auth_hint = (
                " AQ. 키를 직접 전송 방식으로 처리했지만 인증되지 않았습니다."
                if self.uses_aq_inline_audio
                else ""
            )
            raise RecommendationError(
                "Gemini가 음악을 분석하지 못했습니다. API 키, 선택한 모델명, "
                "인터넷 연결을 확인한 뒤 다시 시도하세요."
                f"{auth_hint}\n세부 오류: {error}"
            ) from error

    def _generate_with_inline_audio(self, path: Path) -> Any:
        """AQ. 키용 경로: Files API 없이 MP3 바이트를 직접 보낸다."""
        file_size = path.stat().st_size
        if file_size > MAX_INLINE_AUDIO_BYTES:
            max_mebibytes = MAX_INLINE_AUDIO_BYTES // (1024 * 1024)
            raise RecommendationError(
                "AQ. 키에서는 Gemini API로 MP3를 직접 전송합니다. "
                f"파일 크기를 {max_mebibytes}MB 이하로 줄인 뒤 다시 시도하세요."
            )

        audio_part = types.Part.from_bytes(
            data=path.read_bytes(),
            mime_type="audio/mpeg",
        )
        return self.client.models.generate_content(
            model=self.model,
            contents=[audio_part, self._prompt()],
            config={"response_mime_type": "application/json"},
        )

    def _generate_with_file_api(self, path: Path) -> Any:
        """기존 AIza 계열 키용 Gemini Files API 경로."""
        uploaded_file = None
        try:
            uploaded_file = self.client.files.upload(file=str(path))
            uploaded_file = self._wait_until_ready(uploaded_file)
            return self.client.models.generate_content(
                model=self.model,
                contents=[uploaded_file, self._prompt()],
                config={"response_mime_type": "application/json"},
            )
        finally:
            if uploaded_file is not None and getattr(uploaded_file, "name", None):
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

    @staticmethod
    def _validate_audio_file(path: Path) -> None:
        if not path.is_file():
            raise RecommendationError("선택한 MP3 파일을 찾을 수 없습니다.")
        if path.suffix.lower() != ".mp3":
            raise RecommendationError("현재는 .mp3 파일만 선택할 수 있습니다.")
        if path.stat().st_size == 0:
            raise RecommendationError("비어 있는 MP3 파일입니다.")

    def _wait_until_ready(self, uploaded_file: Any) -> Any:
        deadline = time.monotonic() + 120
        while self._state_name(uploaded_file) == "PROCESSING":
            if time.monotonic() >= deadline:
                raise RecommendationError("오디오 처리 시간이 너무 오래 걸립니다. 다시 시도하세요.")
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        state = self._state_name(uploaded_file)
        if state in {"FAILED", "ERROR"}:
            raise RecommendationError("Gemini가 MP3 파일을 처리하지 못했습니다.")
        return uploaded_file

    @staticmethod
    def _state_name(uploaded_file: Any) -> str:
        state = getattr(uploaded_file, "state", None)
        return str(getattr(state, "name", state or "")).upper()

    def _prompt(self) -> str:
        return f"""
너는 Synthesizer V 보이스 추천 도우미다. 사용자가 업로드한 MP3를 듣고
노래의 분위기, 보컬의 성별 인상·음역·발성, 에너지, 장르, 언어를 종합해
아래 목록 중 가장 잘 어울리는 보이스 하나를 추천해라.

이 목록은 사용자가 실제로 보유한 보이스만 담고 있다. 목록 밖의 보이스는
절대로 추천하거나 alternatives에 넣지 마라.

중요 규칙:
1. recommended_voice 값은 반드시 아래 목록의 '이름' 하나를 철자까지 정확히 복사한다.
2. 노래의 원곡자나 가수를 단정하거나 추측하지 않는다. 들리는 음악적 특징만 설명한다.
3. 확실하지 않은 특징은 "~로 들린다"처럼 조심스럽게 표현한다.
4. 응답은 아래 JSON 형식만 사용한다. match_score는 0~100 정수다.
5. alternatives에는 목록 안의 서로 다른 후보를 최대 2개 넣는다. 후보가 하나면 빈 배열을 쓴다.

{{
  "recommended_voice": "목록에 있는 정확한 이름",
  "match_score": 0,
  "reason": "2~4문장 한국어 추천 이유",
  "listening_notes": "파악한 음악적 특징을 한국어로 간단히 정리",
  "alternatives": [
    {{"voice": "목록에 있는 정확한 이름", "reason": "짧은 보조 추천 이유"}}
  ]
}}

[사용자가 보유한, 선택 가능한 보이스 목록]
{self.catalog}
""".strip()

    @staticmethod
    def _read_json(response: Any) -> dict[str, Any]:
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RecommendationError("Gemini가 빈 응답을 반환했습니다.")

        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError as error:
            raise RecommendationError("Gemini 응답이 올바른 JSON 형식이 아닙니다.") from error
        if not isinstance(parsed, dict):
            raise RecommendationError("Gemini 응답 형식이 예상과 다릅니다.")
        return parsed

    def _validate_response(self, result: dict[str, Any]) -> dict[str, Any]:
        voice = result.get("recommended_voice")
        if voice not in self.available_voice_set:
            raise RecommendationError(
                "Gemini가 보유 보이스 목록 밖의 항목을 골랐습니다. 다시 시도하세요."
            )

        try:
            score = int(result.get("match_score", 0))
        except (TypeError, ValueError):
            score = 0

        alternatives: list[dict[str, str]] = []
        raw_alternatives = result.get("alternatives", [])
        if isinstance(raw_alternatives, list):
            for item in raw_alternatives:
                if not isinstance(item, dict):
                    continue
                candidate = item.get("voice")
                if candidate in self.available_voice_set and candidate != voice:
                    alternatives.append(
                        {
                            "voice": candidate,
                            "reason": str(item.get("reason", "보조 후보입니다.")),
                        }
                    )
                if len(alternatives) == 2:
                    break

        return {
            "recommended_voice": voice,
            "match_score": max(0, min(score, 100)),
            "reason": str(result.get("reason", "추천 이유가 제공되지 않았습니다.")),
            "listening_notes": str(result.get("listening_notes", "")),
            "alternatives": alternatives,
        }
