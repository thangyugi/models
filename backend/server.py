"""
Z-Image Studio backend.

FastAPI server bọc Tongyi-MAI/Z-Image-Turbo (diffusers):
  * Thư viện ảnh tham chiếu (khuôn mặt, trang phục, tư thế, bối cảnh) lưu trên đĩa.
  * Tư thế  -> ControlNet Union (pose skeleton) của alibaba-pai.
  * Mặt / trang phục / bối cảnh -> mô tả văn bản (tự viết hoặc tự sinh bằng Qwen2.5-VL)
    được ghép vào prompt, vì Z-Image-Turbo là model text-to-image, không nhận ảnh tham chiếu trực tiếp.
  * Hàng đợi job chạy tuần tự trên GPU; một lần bấm có thể sinh nhiều tổ hợp x nhiều seed.

Chạy:  uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import random
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- config

MODEL_ID = os.environ.get("ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
CONTROLNET_REPO = os.environ.get("ZIMAGE_CONTROLNET_REPO", "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0")
CONTROLNET_FILE = os.environ.get("ZIMAGE_CONTROLNET_FILE", "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors")
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
CAPTION_KEEP_LOADED = os.environ.get("CAPTION_KEEP_LOADED", "0") == "1"
# none | model | sequential  (model = tiết kiệm VRAM, ~16GB trở xuống nên dùng)
OFFLOAD = os.environ.get("ZIMAGE_OFFLOAD", "none").lower()
PRELOAD = os.environ.get("ZIMAGE_PRELOAD", "1") == "1"
MAX_JOBS_PER_REQUEST = int(os.environ.get("MAX_JOBS_PER_REQUEST", "64"))

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
LIB_DIR = DATA_DIR / "library"
OUT_DIR = DATA_DIR / "outputs"
LIB_INDEX = DATA_DIR / "library.json"
for d in (LIB_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["face", "outfit", "pose", "background"]

# --------------------------------------------------------------------------- library store

_lib_lock = threading.Lock()


def _load_lib() -> list[dict]:
    if LIB_INDEX.exists():
        return json.loads(LIB_INDEX.read_text(encoding="utf-8"))
    return []


def _save_lib(items: list[dict]) -> None:
    tmp = LIB_INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LIB_INDEX)


def _get_item(item_id: str) -> dict:
    for it in _load_lib():
        if it["id"] == item_id:
            return it
    raise HTTPException(404, f"Không tìm thấy item {item_id}")


def _update_item(item_id: str, **changes) -> dict:
    with _lib_lock:
        items = _load_lib()
        for it in items:
            if it["id"] == item_id:
                it.update(changes)
                _save_lib(items)
                return it
    raise HTTPException(404, f"Không tìm thấy item {item_id}")


# --------------------------------------------------------------------------- models

gpu_lock = threading.Lock()  # một tác vụ GPU tại một thời điểm


class ModelManager:
    def __init__(self):
        self.torch = None
        self.device = "cpu"
        self.dtype = None
        self.base = None
        self.cnet_pipe = None
        self.active = None
        self.status = "chưa tải"
        self.error: Optional[str] = None
        self.captioner = None
        self.pose_detector = None

    # ---- base pipeline
    def ensure_base(self):
        if self.base is not None:
            return
        import torch
        from diffusers import ZImagePipeline

        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        self.status = f"đang tải {MODEL_ID}..."
        pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
        self.base = pipe
        self._activate(pipe)
        self.status = "sẵn sàng"

    def _activate(self, pipe):
        if self.active is pipe:
            return
        if OFFLOAD in ("model", "sequential") and self.device == "cuda":
            if self.active is not None and hasattr(self.active, "remove_all_hooks"):
                self.active.remove_all_hooks()
            if OFFLOAD == "sequential":
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self.active = pipe

    # ---- controlnet pipeline (dùng chung các thành phần với base)
    def ensure_controlnet(self):
        self.ensure_base()
        if self.cnet_pipe is not None:
            return
        from diffusers import ZImageControlNetModel, ZImageControlNetPipeline
        from huggingface_hub import hf_hub_download

        self.status = "đang tải ControlNet (pose)..."
        path = hf_hub_download(CONTROLNET_REPO, filename=CONTROLNET_FILE)
        cn = ZImageControlNetModel.from_single_file(path, torch_dtype=self.dtype)
        comps = {k: v for k, v in self.base.components.items() if k != "controlnet"}
        self.cnet_pipe = ZImageControlNetPipeline(**comps, controlnet=cn)
        self.status = "sẵn sàng"

    # ---- generation
    def generate(self, prompt: str, width: int, height: int, steps: int, seed: int,
                 control_image: Optional[Image.Image] = None, control_scale: float = 0.75) -> Image.Image:
        torch = self.torch
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(gen_device).manual_seed(seed)
        common = dict(prompt=prompt, width=width, height=height, num_inference_steps=steps,
                      guidance_scale=0.0, generator=generator)
        if control_image is not None:
            self.ensure_controlnet()
            self._activate(self.cnet_pipe)
            return self.cnet_pipe(control_image=control_image.resize((width, height)),
                                  controlnet_conditioning_scale=control_scale, **common).images[0]
        self._activate(self.base)
        return self.base(**common).images[0]

    # ---- pose extraction (OpenPose, chạy CPU)
    def extract_pose(self, img: Image.Image) -> Image.Image:
        if self.pose_detector is None:
            try:
                from controlnet_aux import OpenposeDetector
            except ImportError as e:
                raise HTTPException(400, "Chưa cài controlnet_aux (pip install controlnet_aux)") from e
            self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        out = self.pose_detector(img, include_body=True, include_hand=True, include_face=True,
                                 detect_resolution=512, image_resolution=max(img.size))
        return out.resize(img.size)

    # ---- captioning (Qwen2.5-VL)
    def caption(self, img: Image.Image, category: str) -> str:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if self.captioner is None:
            self.status = f"đang tải {CAPTION_MODEL}..."
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                CAPTION_MODEL, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            ).to(device)
            proc = AutoProcessor.from_pretrained(CAPTION_MODEL, max_pixels=768 * 768)
            self.captioner = (model, proc, device)
            self.status = "sẵn sàng"
        model, proc, device = self.captioner
        instructions = {
            "face": "Describe this person's appearance for a text-to-image prompt: gender, approximate age, "
                    "ethnicity, face shape, eyes, eyebrows, nose, lips, skin tone, hairstyle and hair color, "
                    "makeup, distinctive features. Ignore clothing and background.",
            "outfit": "Describe only the clothing in this image for a text-to-image prompt: every garment, "
                      "cut, colors, materials, patterns, accessories, shoes. Ignore the person's face and background.",
            "pose": "Describe only the body pose in this image for a text-to-image prompt: stance, orientation, "
                    "arms, hands, legs, head direction, camera angle and framing.",
            "background": "Describe only the setting/background of this image for a text-to-image prompt: "
                          "location, lighting, time of day, colors, mood, notable objects.",
        }
        text = instructions.get(category, instructions["face"]) + \
            " Answer with one concise English paragraph (max 70 words), no preamble."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[img.convert("RGB")], return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        result = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if not CAPTION_KEEP_LOADED:
            self.captioner = None
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result


mm = ModelManager()

# --------------------------------------------------------------------------- jobs

jobs: dict[str, dict] = {}
job_order: list[str] = []
job_queue: "queue.Queue[str]" = queue.Queue()


def build_prompt(template: str, parts: dict[str, Optional[dict]]) -> str:
    desc = {k: (v.get("description") or "").strip() if v else "" for k, v in parts.items()}
    template = (template or "").strip()
    if any("{" + k + "}" in template for k in CATEGORIES):
        out = template
        for k in CATEGORIES:
            out = out.replace("{" + k + "}", desc.get(k, ""))
        return out
    labels = {"face": "The person", "outfit": "Wearing", "pose": "Pose", "background": "Setting"}
    segs = [template] if template else []
    for k in CATEGORIES:
        if desc.get(k):
            segs.append(f"{labels[k]}: {desc[k]}")
    return ". ".join(s.rstrip(". ") for s in segs) + "."


def worker():
    if PRELOAD:
        try:
            with gpu_lock:
                mm.ensure_base()
        except Exception as e:  # noqa: BLE001
            mm.error = f"{type(e).__name__}: {e}"
            mm.status = "lỗi khi tải model"
            traceback.print_exc()
    while True:
        jid = job_queue.get()
        job = jobs.get(jid)
        if not job or job["status"] == "cancelled":
            continue
        job["status"] = "running"
        job["started"] = time.time()
        try:
            control = None
            if job.get("pose_image"):
                control = Image.open(LIB_DIR / job["pose_image"]).convert("RGB")
            with gpu_lock:
                mm.ensure_base()
                img = mm.generate(job["prompt"], job["width"], job["height"], job["steps"], job["seed"],
                                  control, job["control_scale"])
            fname = f"{int(time.time())}_{jid}.png"
            img.save(OUT_DIR / fname)
            job["output"] = fname
            job["status"] = "done"
            meta = {k: job[k] for k in ("id", "prompt", "seed", "width", "height", "steps", "labels", "batch")}
            meta.update(file=fname, created=time.time())
            (OUT_DIR / (fname + ".json")).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            mm.status = "sẵn sàng" if mm.base is not None else mm.status
        job["finished"] = time.time()


threading.Thread(target=worker, daemon=True).start()

# --------------------------------------------------------------------------- API

app = FastAPI(title="Z-Image Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/files/library", StaticFiles(directory=LIB_DIR), name="library")
app.mount("/files/outputs", StaticFiles(directory=OUT_DIR), name="outputs")


@app.get("/api/status")
def status():
    pending = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    return {"model": MODEL_ID, "status": mm.status, "error": mm.error, "device": mm.device,
            "offload": OFFLOAD, "pending": pending, "categories": CATEGORIES}


@app.get("/api/library")
def list_library():
    return _load_lib()


@app.post("/api/library")
async def add_library(file: UploadFile = File(...), category: str = Form(...), name: str = Form(""),
                      description: str = Form(""), is_skeleton: bool = Form(False)):
    if category not in CATEGORIES:
        raise HTTPException(400, f"category phải là một trong {CATEGORIES}")
    try:
        img = ImageOps.exif_transpose(Image.open(file.file)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "File không phải ảnh hợp lệ") from e
    img.thumbnail((2048, 2048))
    item_id = uuid.uuid4().hex[:10]
    fname = f"{item_id}.png"
    img.save(LIB_DIR / fname)
    item = {"id": item_id, "category": category, "name": name or Path(file.filename or "").stem or item_id,
            "description": description, "file": fname, "pose_file": fname if is_skeleton else None,
            "created": time.time()}
    with _lib_lock:
        items = _load_lib()
        items.append(item)
        _save_lib(items)
    return item


class ItemPatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None


@app.patch("/api/library/{item_id}")
def patch_library(item_id: str, patch: ItemPatch):
    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "category" in changes and changes["category"] not in CATEGORIES:
        raise HTTPException(400, "category không hợp lệ")
    return _update_item(item_id, **changes)


@app.delete("/api/library/{item_id}")
def delete_library(item_id: str):
    with _lib_lock:
        items = _load_lib()
        keep = [i for i in items if i["id"] != item_id]
        if len(keep) == len(items):
            raise HTTPException(404, "Không tìm thấy")
        gone = next(i for i in items if i["id"] == item_id)
        _save_lib(keep)
    for f in {gone.get("file"), gone.get("pose_file")} - {None}:
        (LIB_DIR / f).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/library/{item_id}/describe")
def describe_item(item_id: str):
    item = _get_item(item_id)
    img = Image.open(LIB_DIR / item["file"])
    try:
        with gpu_lock:
            text = mm.caption(img, item["category"])
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"Không mô tả được ảnh: {type(e).__name__}: {e}") from e
    return _update_item(item_id, description=text)


@app.post("/api/library/{item_id}/extract-pose")
def extract_pose(item_id: str):
    item = _get_item(item_id)
    img = Image.open(LIB_DIR / item["file"]).convert("RGB")
    try:
        skel = mm.extract_pose(img)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"Không trích được tư thế: {type(e).__name__}: {e}") from e
    pose_name = f"{item_id}_pose.png"
    skel.save(LIB_DIR / pose_name)
    return _update_item(item_id, pose_file=pose_name)


class GenerateRequest(BaseModel):
    prompt: str = ""
    face_ids: list[str] = Field(default_factory=list)
    outfit_ids: list[str] = Field(default_factory=list)
    pose_ids: list[str] = Field(default_factory=list)
    background_ids: list[str] = Field(default_factory=list)
    images_per_combo: int = Field(1, ge=1, le=16)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(9, ge=1, le=50)
    seed: int = -1  # -1 = ngẫu nhiên
    control_scale: float = Field(0.75, ge=0.0, le=2.0)
    use_pose_control: bool = True


@app.post("/api/generate")
def generate(req: GenerateRequest):
    lib = {i["id"]: i for i in _load_lib()}

    def pick(ids):
        missing = [i for i in ids if i not in lib]
        if missing:
            raise HTTPException(400, f"Item không tồn tại: {missing}")
        return [lib[i] for i in ids] or [None]

    groups = [pick(req.face_ids), pick(req.outfit_ids), pick(req.pose_ids), pick(req.background_ids)]
    combos = list(itertools.product(*groups))
    total = len(combos) * req.images_per_combo
    if total > MAX_JOBS_PER_REQUEST:
        raise HTTPException(400, f"Quá nhiều ảnh ({total}); tối đa {MAX_JOBS_PER_REQUEST} mỗi lần")
    if req.use_pose_control and any(p and not p.get("pose_file") for p in groups[2]):
        raise HTTPException(400, "Có ảnh tư thế chưa có skeleton. Bấm 'Trích tư thế' trong thư viện, "
                                 "hoặc tắt 'Dùng ControlNet pose'.")
    w, h = req.width // 16 * 16, req.height // 16 * 16
    batch = uuid.uuid4().hex[:8]
    base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
    created = []
    n = 0
    for face, outfit, pose, bg in combos:
        parts = {"face": face, "outfit": outfit, "pose": pose, "background": bg}
        prompt = build_prompt(req.prompt, parts)
        labels = {k: v["name"] for k, v in parts.items() if v}
        for _ in range(req.images_per_combo):
            jid = uuid.uuid4().hex[:10]
            jobs[jid] = {
                "id": jid, "batch": batch, "status": "queued", "prompt": prompt, "labels": labels,
                "seed": base_seed + n, "width": w, "height": h, "steps": req.steps,
                "control_scale": req.control_scale,
                "pose_image": pose["pose_file"] if (pose and req.use_pose_control) else None,
                "created": time.time(), "output": None, "error": None,
            }
            job_order.append(jid)
            job_queue.put(jid)
            created.append(jid)
            n += 1
    return {"batch": batch, "jobs": created}


@app.get("/api/jobs")
def list_jobs(limit: int = 200):
    return [jobs[j] for j in job_order[-limit:]][::-1]


@app.post("/api/jobs/cancel")
def cancel_jobs():
    n = 0
    for j in jobs.values():
        if j["status"] == "queued":
            j["status"] = "cancelled"
            n += 1
    return {"cancelled": n}


@app.get("/api/outputs")
def list_outputs(limit: int = 300):
    metas = []
    for p in sorted(OUT_DIR.glob("*.png.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return metas


@app.delete("/api/outputs/{fname}")
def delete_output(fname: str):
    if "/" in fname or ".." in fname:
        raise HTTPException(400, "tên file không hợp lệ")
    (OUT_DIR / fname).unlink(missing_ok=True)
    (OUT_DIR / (fname + ".json")).unlink(missing_ok=True)
    return {"ok": True}


# Phục vụ frontend đã build (npm run build) nếu có
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
