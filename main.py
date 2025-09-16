import os, time, tempfile, shutil, threading, base64, hashlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from indextts.infer_v2 import IndexTTS2

MODEL_DIR = "./checkpoints"
PROMPT_CACHE_DIR = os.environ.get("PROMPT_CACHE_DIR", os.path.join(tempfile.gettempdir(), "indextts", "prompts"))
USE_FP16 = os.environ.get("USE_FP16", "0") == "1"
USE_CUDA_KERNEL = os.environ.get("USE_CUDA_KERNEL", "0") == "1"

tts = IndexTTS2(
    model_dir=MODEL_DIR,
    cfg_path=os.path.join(MODEL_DIR, "config.yaml"),
    use_fp16=USE_FP16,
    use_deepspeed=False,
    use_cuda_kernel=USE_CUDA_KERNEL,
)

busy_lock = threading.Lock()
app = FastAPI(title="IndexTTS HTTP Service")

@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    code = "HTTP_ERROR"
    if exc.status_code == 400:
        code = "BAD_REQUEST"
    elif exc.status_code == 429:
        code = "BUSY"
    elif exc.status_code == 500:
        code = "INTERNAL_ERROR"
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"code": code, "message": message})

def normalize_emotion_vector(vec: List[float]) -> List[float]:
    k_vec = [0.75,0.70,0.80,0.80,0.75,0.75,0.55,0.45]
    tmp = [k*v for k, v in zip(k_vec, vec)]
    s = sum(tmp)
    if s > 0.8 and s > 0:
        scale = 0.8 / s
        tmp = [x*scale for x in tmp]
    return tmp

def build_vector_from_factors(factors: Dict[str, Any]) -> List[float]:
    try:
        ordered = [
            float(factors["happy"]),
            float(factors["angry"]),
            float(factors["sad"]),
            float(factors["afraid"]),
            float(factors["disgusted"]),
            float(factors["melancholic"]),
            float(factors["surprised"]),
            float(factors["calm"]),
        ]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"emotion_factors missing: {str(e)}")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="emotion_factors values must be numbers")
    return ordered

def run_infer(
    prompt_path: str,
    text: str,
    out_path: str,
    *,
    emo_audio_path: Optional[str],
    emo_alpha: float,
    emo_vector: Optional[List[float]],
    use_emo_text: bool,
    emo_text: Optional[str],
    use_random: bool,
    max_text_tokens_per_segment: int,
    gen_kwargs: Dict[str, Any],
):
    _ = tts.infer(
        spk_audio_prompt=prompt_path,
        text=text,
        output_path=out_path,
        emo_audio_prompt=emo_audio_path,
        emo_alpha=float(emo_alpha),
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=bool(use_random),
        verbose=False,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **gen_kwargs,
    )

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

CAS_DIR = PROMPT_CACHE_DIR

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_base64_to_cas(b64_data: str, suffix: str = ".wav") -> str:
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")
    sha256 = hashlib.sha256(raw).hexdigest()
    _ensure_dir(CAS_DIR)
    filename = f"{sha256}{suffix}"
    final_path = os.path.join(CAS_DIR, filename)
    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        return final_path
    tmp_path = os.path.join(CAS_DIR, f".{filename}.tmp_{os.getpid()}_{int(time.time()*1000)}")
    with open(tmp_path, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    return final_path

def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def acquire_lock_or_429():
    if not busy_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Busy: another synthesis is in progress")

def release_lock():
    busy_lock.release()

def save_upload_to_temp(u):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        shutil.copyfileobj(u.file, f)
        return f.name

class BasePayload(BaseModel):
    prompt_audio: str
    text: str
    max_text_tokens_per_segment: int = 120

class GenerationArgsModel(BaseModel):
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500

class SpeakerModel(BasePayload):
    generation_args: GenerationArgsModel

class ReferenceAudioModel(BasePayload):
    generation_args: GenerationArgsModel
    emotion_audio: str
    emotion_weight: float = 0.8

class EmotionFactorsModel(BaseModel):
    happy: float
    angry: float
    sad: float
    afraid: float
    disgusted: float
    melancholic: float
    surprised: float
    calm: float

class VectorsModel(BasePayload):
    generation_args: GenerationArgsModel
    emotion_factors: EmotionFactorsModel
    emotion_random: bool = False

class TextPromptModel(BasePayload):
    generation_args: GenerationArgsModel
    emotion_text: Optional[str] = None
    emotion_random: bool = False

@app.post("/synthesize/speaker")
def synthesize_speaker(payload: SpeakerModel):
    acquire_lock_or_429()
    try:
        prompt_path = save_base64_to_cas(payload.prompt_audio)
        out_path = os.path.join(tempfile.gettempdir(), f"spk_{int(time.time())}.wav")
        run_infer(
            prompt_path,
            payload.text,
            out_path,
            emo_audio_path=None,  # speaker mode: no external emotion audio
            emo_alpha=1.0,
            emo_vector=None,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
            max_text_tokens_per_segment=payload.max_text_tokens_per_segment,
            gen_kwargs=payload.generation_args.model_dump(),
        )
        b64 = bytes_to_base64(read_file_bytes(out_path))
        return JSONResponse(content=b64)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    finally:
        release_lock()

@app.post("/synthesize/reference")
def synthesize_reference(payload: ReferenceAudioModel):
    acquire_lock_or_429()
    try:
        prompt_path = save_base64_to_cas(payload.prompt_audio)
        emo_path = save_base64_to_cas(payload.emotion_audio)
        # webui: weight scaled by 0.8 for UX
        emo_alpha = float(payload.emotion_weight) * 0.8
        out_path = os.path.join(tempfile.gettempdir(), f"spk_{int(time.time())}.wav")
        run_infer(
            prompt_path,
            payload.text,
            out_path,
            emo_audio_path=emo_path,
            emo_alpha=emo_alpha,
            emo_vector=None,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
            max_text_tokens_per_segment=payload.max_text_tokens_per_segment,
            gen_kwargs=payload.generation_args.model_dump(),
        )
        b64 = bytes_to_base64(read_file_bytes(out_path))
        return JSONResponse(content=b64)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    finally:
        release_lock()

@app.post("/synthesize/vector")
def synthesize_vector(payload: VectorsModel):
    acquire_lock_or_429()
    try:
        factors_obj = payload.emotion_factors.model_dump()
        vec = build_vector_from_factors(factors_obj)
        emo_vec = normalize_emotion_vector(vec)

        prompt_path = save_base64_to_cas(payload.prompt_audio)
        out_path = os.path.join(tempfile.gettempdir(), f"spk_{int(time.time())}.wav")
        run_infer(
            prompt_path,
            payload.text,
            out_path,
            emo_audio_path=None,  # vector mode: ignore emotion audio blending
            emo_alpha=1.0,        # not used for vector mixing in infer()
            emo_vector=emo_vec,
            use_emo_text=False,
            emo_text=None,
            use_random=payload.emotion_random,
            max_text_tokens_per_segment=payload.max_text_tokens_per_segment,
            gen_kwargs=payload.generation_args.model_dump(),
        )
        b64 = bytes_to_base64(read_file_bytes(out_path))
        return JSONResponse(content=b64)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    finally:
        release_lock()

@app.post("/synthesize/text")
def synthesize_text(payload: TextPromptModel):
    acquire_lock_or_429()
    try:
        emo_text = payload.emotion_text
        if emo_text == "":
            raise HTTPException(status_code=400, detail="emotion_text cannot be empty")

        prompt_path = save_base64_to_cas(payload.prompt_audio)
        out_path = os.path.join(tempfile.gettempdir(), f"spk_{int(time.time())}.wav")
        run_infer(
            prompt_path,
            payload.text,
            out_path,
            emo_audio_path=None,  # text mode: emotion derived from text, no audio blending
            emo_alpha=1.0,
            emo_vector=None,
            use_emo_text=True,
            emo_text=emo_text,
            use_random=payload.emotion_random,
            max_text_tokens_per_segment=payload.max_text_tokens_per_segment,
            gen_kwargs=payload.generation_args.model_dump(),
        )
        b64 = bytes_to_base64(read_file_bytes(out_path))
        return JSONResponse(content=b64)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    finally:
        release_lock()