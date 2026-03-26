from __future__ import annotations

from .config import Qwen3VLConfig
from .schema import CandidateProposal
from .scene import PVGScene
from .utils import clamp01, strict_json_load


class Qwen3VLCandidateProposer:
    def __init__(self, config: Qwen3VLConfig) -> None:
        self.config = config
        self._torch = None
        self._model = None
        self._processor = None
        self._load_or_raise()

    def _load_or_raise(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as ModelClass
                except ImportError:
                    from transformers import AutoModelForImageTextToText as ModelClass
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-VL requires a recent transformers installation with multimodal support."
            ) from exc

        torch_dtype = self.config.torch_dtype
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        self._processor = AutoProcessor.from_pretrained(self.config.model_path, trust_remote_code=True)
        self._disable_thinking_template()
        self._model = ModelClass.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )
        self._model.eval()
        self._torch = torch

    def _disable_thinking_template(self) -> None:
        think_prefix = "{{- '<|im_start|>assistant\\n<think>\\n' }}"
        plain_prefix = "{{- '<|im_start|>assistant\\n' }}"
        for owner in (self._processor, getattr(self._processor, "tokenizer", None)):
            if owner is None:
                continue
            chat_template = getattr(owner, "chat_template", None)
            if isinstance(chat_template, str) and think_prefix in chat_template:
                owner.chat_template = chat_template.replace(think_prefix, plain_prefix)

    def propose(
        self,
        scene: PVGScene,
        camera: str,
        frame_index: int,
        context_size: int,
        max_candidates: int,
    ) -> list[CandidateProposal]:
        context_frames = scene.context_frames(camera, frame_index, context_size)
        prompt = self._build_prompt(scene, camera, frame_index, max_candidates)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": frame} for frame in context_frames]
                + [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(self._model.parameters()).device
        inputs = inputs.to(model_device)
        text = self._generate_text(inputs, self.config.max_new_tokens)
        if not self._has_valid_json(text):
            retry_tokens = max(4096, self.config.max_new_tokens * 2)
            text = self._generate_text(inputs, retry_tokens)
        return self._parse_candidates(text, scene, camera, frame_index, max_candidates)

    def _generate_text(self, inputs, max_new_tokens: int) -> str:
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _has_valid_json(self, text: str) -> bool:
        try:
            strict_json_load(text)
            return True
        except Exception:
            return False

    def _build_prompt(self, scene: PVGScene, camera: str, frame_index: int, max_candidates: int) -> str:
        width = scene.metadata.width
        height = scene.metadata.height
        return (
            "You are mining rare and hard-to-reconstruct safety-critical objects in autonomous driving video. "
            "Focus especially on birds, small animals, or small fast unusual objects that are easy to miss and hard to reconstruct. "
            f"The target is the middle frame only from the {camera} camera at frame index {frame_index}. "
            f"The target frame resolution is width={width}, height={height}. "
            "Use surrounding frames only as temporal context. "
            "Do not output reasoning, analysis, or any text before or after the JSON. "
            "Do not say 'Got it' or explain your steps. "
            "Return strict JSON only, with one top-level key named candidates and no extra text. "
            f"Return at most {max_candidates} candidates. "
            "Each candidate must have exactly these keys: "
            "candidate_label, candidate_text_prompt, confidence. "
            "candidate_label must be one of: bird, small_animal, unknown_small_object. "
            "Return at most one candidate per label. "
            "If multiple phrasings fit the same label, keep only the strongest single prompt for that label. "
            "candidate_text_prompt must explicitly emphasize rare, hard-to-reconstruct, bird-like or small-animal targets."
        )

    def _parse_candidates(
        self,
        text: str,
        scene: PVGScene,
        camera: str,
        frame_index: int,
        max_candidates: int,
    ) -> list[CandidateProposal]:
        try:
            payload = strict_json_load(text)
        except Exception as exc:
            raise RuntimeError(
                f"VLM proposal output was not valid strict JSON for camera={camera} frame={frame_index}: {text[:400]!r}"
            ) from exc

        if not isinstance(payload, dict) or set(payload.keys()) != {"candidates"}:
            raise RuntimeError(
                f"VLM proposal output must be a JSON object with exactly one key 'candidates' for camera={camera} frame={frame_index}."
            )

        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise RuntimeError("VLM proposal output field 'candidates' must be a list.")

        allowed_labels = {"bird", "small_animal", "unknown_small_object"}
        parsed_by_label: dict[str, CandidateProposal] = {}
        for raw in raw_candidates[:max_candidates]:
            if not isinstance(raw, dict):
                raise RuntimeError("Each VLM candidate must be a JSON object.")
            if set(raw.keys()) != {
                "candidate_label",
                "candidate_text_prompt",
                "confidence",
            }:
                raise RuntimeError(
                    "Each VLM candidate must contain exactly candidate_label, candidate_text_prompt, confidence."
                )
            label = raw["candidate_label"]
            text_prompt = raw["candidate_text_prompt"]
            confidence = raw["confidence"]
            if label not in allowed_labels:
                raise RuntimeError(f"Unexpected candidate_label: {label}")
            if not isinstance(text_prompt, str) or not text_prompt.strip():
                raise RuntimeError("candidate_text_prompt must be a non-empty string.")
            try:
                conf = clamp01(float(confidence))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("confidence must be numeric.") from exc
            candidate = CandidateProposal(
                camera=camera,
                frame_index=frame_index,
                label=label,
                text_prompt=text_prompt.strip(),
                confidence=conf,
            )
            current = parsed_by_label.get(label)
            if current is None or candidate.confidence > current.confidence:
                parsed_by_label[label] = candidate
        parsed = sorted(parsed_by_label.values(), key=lambda item: item.confidence, reverse=True)
        return parsed[:max_candidates]
