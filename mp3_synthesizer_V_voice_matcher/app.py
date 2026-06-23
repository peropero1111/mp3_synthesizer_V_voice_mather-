"""MP3에 어울리는, 사용자가 보유한 Synthesizer V 보이스를 추천하는 앱."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from gemini_recommender import (
    AQ_KEY_PREFIX,
    MAX_INLINE_AUDIO_BYTES,
    GeminiRecommender,
    RecommendationError,
)
from voices_data import voices


class VoiceRecommenderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Synthesizer V 보이스 추천기")
        self.geometry("800x780")
        self.minsize(680, 580)

        self.selected_path = tk.StringVar()
        self.status = tk.StringVar(value="보유한 보이스를 선택해 주세요.")
        self.selection_summary = tk.StringVar()
        self.voice_variables = {name: tk.BooleanVar(value=False) for name in voices}
        self.uses_aq_inline_audio = self._uses_aq_inline_audio_key()
        self._build_ui()
        self._update_selection_summary()

    @staticmethod
    def _uses_aq_inline_audio_key() -> bool:
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        return api_key.startswith(AQ_KEY_PREFIX)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(6, weight=1)

        ttk.Label(
            container,
            text="mp3 Synthesizer V voice matcher",
            font=("Malgun Gothic", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            container,
            text="보유한 보이스만 선택하면, Gemini가 mp3에 맞는 보이스를 추천합니다.",
        ).grid(row=1, column=0, sticky="w", pady=(5, 8))

        if self.uses_aq_inline_audio:
            max_mebibytes = MAX_INLINE_AUDIO_BYTES // (1024 * 1024)
            connection_text = (
                f"AQ. 키 감지: Gemini API로 MP3를 직접 전송하며 "
                f"{max_mebibytes}MB 이하여야 합니다."
            )
        else:
            connection_text = "일반 Gemini API 키 감지: MP3를 Gemini Files API로 전송합니다."
        ttk.Label(container, text=connection_text, foreground="#356a3c").grid(
            row=2, column=0, sticky="w", pady=(0, 10)
        )

        self._build_voice_selector(container).grid(row=3, column=0, sticky="nsew")

        file_frame = ttk.Frame(container)
        file_frame.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        file_frame.columnconfigure(0, weight=1)
        ttk.Entry(file_frame, textvariable=self.selected_path, state="readonly").grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(file_frame, text="MP3 파일 선택", command=self._choose_file).grid(
            row=0, column=1
        )

        action_frame = ttk.Frame(container)
        action_frame.grid(row=5, column=0, sticky="ew", pady=12)
        self.recommend_button = ttk.Button(
            action_frame,
            text="선택한 보이스 중 추천 받기",
            command=self._start_recommendation,
        )
        self.recommend_button.pack(side="left")
        self.progress = ttk.Progressbar(action_frame, mode="indeterminate", length=150)
        self.progress.pack(side="left", padx=12)
        ttk.Label(action_frame, textvariable=self.status).pack(side="left")

        self.result = scrolledtext.ScrolledText(
            container,
            wrap="word",
            font=("Malgun Gothic", 10),
            state="disabled",
            padx=10,
            pady=10,
        )
        self.result.grid(row=6, column=0, sticky="nsew")
        self._set_result(
            "사용 순서\n\n"
            "1. 위 목록에서 실제로 보유한 보이스를 모두 체크합니다.\n"
            "2. 분석할 MP3 파일을 고릅니다.\n"
            "3. 추천 버튼을 누르면 선택한 보이스 중 하나만 결과로 나옵니다.\n\n"

            "Gemini가 음악을 분석하지 못했습니다. API 키, 선택한 모델명,인터넷 연결을 확인한 뒤 다시 시도하세요. 세부 오류: 503 이 출력되는 경우에는 gemini 서버가 붐비는 ( 503 이 붐비는 것입니다. ) 것이니 이 창을 닫으시고 5-10 분 정도 후에 다시 시도하여 주시기 바랍니다."

        )

    def _build_voice_selector(self, parent: ttk.Frame) -> ttk.LabelFrame:
        selector = ttk.LabelFrame(parent, text="보유한 보이스 선택", padding=8)

        header = ttk.Frame(selector)
        header.pack(fill="x", pady=(0, 7))
        ttk.Label(header, textvariable=self.selection_summary).pack(side="left")
        ttk.Button(header, text="전체 선택", command=self._select_all).pack(side="right")
        ttk.Button(header, text="전체 해제", command=self._clear_all).pack(
            side="right", padx=(0, 6)
        )

        list_frame = ttk.Frame(selector)
        list_frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(list_frame, height=210, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        checkbox_frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")
        checkbox_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.bind(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
        )

        for index, (name, variable) in enumerate(self.voice_variables.items()):
            ttk.Checkbutton(
                checkbox_frame,
                text=name,
                variable=variable,
                command=self._update_selection_summary,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(4, 24), pady=2)
        return selector

    def _select_all(self) -> None:
        for variable in self.voice_variables.values():
            variable.set(True)
        self._update_selection_summary()

    def _clear_all(self) -> None:
        for variable in self.voice_variables.values():
            variable.set(False)
        self._update_selection_summary()

    def _update_selection_summary(self) -> None:
        count = len(self._selected_voices())
        self.selection_summary.set(f"선택됨: {count} / {len(voices)}개")

    def _selected_voices(self) -> list[str]:
        return [name for name, variable in self.voice_variables.items() if variable.get()]

    def _choose_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="분석할 MP3 선택",
            filetypes=[("MP3 파일", "*.mp3"), ("모든 파일", "*.*")],
        )
        if not chosen:
            return

        if self.uses_aq_inline_audio:
            size = Path(chosen).stat().st_size
            if size > MAX_INLINE_AUDIO_BYTES:
                max_mebibytes = MAX_INLINE_AUDIO_BYTES // (1024 * 1024)
                messagebox.showwarning(
                    "MP3 파일이 너무 큼",
                    f"AQ. 키에서는 {max_mebibytes}MB 이하 MP3만 직접 전송할 수 있습니다.",
                )
                return

        self.selected_path.set(chosen)
        self.status.set(f"선택됨: {Path(chosen).name}")

    def _start_recommendation(self) -> None:
        owned_voices = self._selected_voices()
        if not owned_voices:
            messagebox.showwarning(
                "보유 보이스 선택 필요", "먼저 보유한 보이스를 한 개 이상 체크해 주세요."
            )
            return

        path = self.selected_path.get()
        if not path:
            messagebox.showwarning("MP3 파일 필요", "분석할 MP3 파일을 선택해 주세요.")
            return

        self.recommend_button.config(state="disabled")
        self.progress.start(12)
        self.status.set(f"Gemini가 선택한 {len(owned_voices)}개 보이스만 비교하는 중...")
        self._set_result("분석 중입니다. MP3 길이와 인터넷 상태에 따라 잠시 걸릴 수 있습니다.")
        threading.Thread(
            target=self._recommend_in_background,
            args=(path, owned_voices),
            daemon=True,
        ).start()

    def _recommend_in_background(self, path: str, owned_voices: list[str]) -> None:
        try:
            answer = GeminiRecommender(available_voices=owned_voices).recommend(path)
        except RecommendationError as error:
            self.after(0, self._show_error, str(error))
        except Exception as error:
            self.after(0, self._show_error, f"예상하지 못한 오류가 발생했습니다.\n{error}")
        else:
            self.after(0, self._show_recommendation, answer)

    def _show_recommendation(self, answer: dict) -> None:
        lines = [
            f"가장 어울리는 보이스: {answer['recommended_voice']}",
            f"매칭 점수: {answer['match_score']} / 100",
            "",
            "추천 이유",
            answer["reason"],
        ]
        if answer["listening_notes"]:
            lines.extend(["", "들린 음악적 특징", answer["listening_notes"]])
        if answer["alternatives"]:
            lines.extend(["", "선택한 보이스 중 다른 후보"])
            for item in answer["alternatives"]:
                lines.append(f"- {item['voice']}: {item['reason']}")

        self._set_result("\n".join(lines))
        self._finish("추천이 완료되었습니다.")

    def _show_error(self, error: str) -> None:
        self._set_result(f"분석하지 못했습니다.\n\n{error}")
        self._finish("오류가 발생했습니다.")
        messagebox.showerror("추천 오류", error)

    def _finish(self, status: str) -> None:
        self.progress.stop()
        self.recommend_button.config(state="normal")
        self.status.set(status)

    def _set_result(self, text: str) -> None:
        self.result.config(state="normal")
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", text)
        self.result.config(state="disabled")


if __name__ == "__main__":
    VoiceRecommenderApp().mainloop()
