#!/usr/bin/env python3
"""
pope_replay_models.py
=====================
Model wrappers for the cross-model POPE replay (scripts/pope_replay.py). Each
wrapper exposes the same interface as anatomy_prior_probe.MedGemma:
    __init__(cfg, model_path)        # loads model + processor, prints device map
    ask(image_rgb: np.ndarray, prompt: str) -> str   # greedy, single image

Only the image-token / chat-template wrapping differs between models; the prompt
text, decoding (do_sample=False), and max_new_tokens come from cfg and are held
identical to the MedGemma-27B run.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image


class QwenVL:
    """Qwen2.5-VL-7B-Instruct via transformers (general-domain comparison)."""

    def __init__(self, cfg, model_path: str):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Q
        except ImportError:  # older alias
            from transformers import Qwen2VLForConditionalGeneration as _Q
        self.torch = torch
        self.cfg = cfg
        self.model = _Q.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                        device_map="auto")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        dev = getattr(self.model, "hf_device_map", None) or next(self.model.parameters()).device
        print(f"[model] qwen {model_path}\n[model] device map: {dev}", file=sys.stderr)

    def ask(self, image_rgb, prompt: str) -> str:
        torch = self.torch
        if image_rgb is None:                     # text-only (no-image control)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        else:
            img = Image.fromarray(image_rgb).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens,
                                      do_sample=False)
        out = gen[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(out, skip_special_tokens=True).strip()


class LlavaMed:
    """LLaVA-Med v1.5 (Mistral-7B + CLIP-ViT-L/336) — medical specialization
    comparison. The microsoft checkpoint is in original-LLaVA format
    (LlavaMistralForCausalLM), so it loads through the `llava` package rather than
    plain transformers. Image is resized to the model's native 336 px by its own
    processor; we do NOT equalize resolution (organ presence is a coarse signal)."""

    def __init__(self, cfg, model_path: str):
        import torch
        self.torch = torch
        self.cfg = cfg
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                                         DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
            from llava.conversation import conv_templates
        except ImportError as e:
            raise RuntimeError(
                "LLaVA-Med needs the `llava` package: pip install "
                "git+https://github.com/microsoft/LLaVA-Med.git (or haotian-liu/LLaVA). "
                f"import error: {e}")
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._conv_templates = conv_templates
        name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path, model_base=None, model_name=name)
        self.model.eval()
        self._conv_mode = "mistral_instruct"
        dev = next(self.model.parameters()).device
        print(f"[model] llava-med {model_path} ({name})\n[model] device: {dev}", file=sys.stderr)

    def ask(self, image_rgb: np.ndarray, prompt: str) -> str:
        torch = self.torch
        from llava.mm_utils import tokenizer_image_token, process_images
        img = Image.fromarray(image_rgb).convert("RGB")
        cfg = self.model.config
        if getattr(cfg, "mm_use_im_start_end", False):
            img_tok = (self._DEFAULT_IM_START_TOKEN + self._DEFAULT_IMAGE_TOKEN
                       + self._DEFAULT_IM_END_TOKEN)
        else:
            img_tok = self._DEFAULT_IMAGE_TOKEN
        conv = self._conv_templates[self._conv_mode].copy()
        conv.append_message(conv.roles[0], img_tok + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        full = conv.get_prompt()
        input_ids = tokenizer_image_token(full, self.tokenizer, self._IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(self.model.device)
        img_tensor = process_images([img], self.image_processor, cfg)[0]
        # LLaVA-Med loads in float16; match the model weights' dtype exactly.
        img_tensor = img_tensor.unsqueeze(0).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(input_ids, images=img_tensor,
                                      image_sizes=[img.size], do_sample=False,
                                      max_new_tokens=self.cfg.max_new_tokens, use_cache=True)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
