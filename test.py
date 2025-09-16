import argparse
import base64
import os
import sys
import time
from typing import Any, Dict, Optional

import requests


def read_file_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_wav_base64(b64_audio: str, out_path: str) -> None:
    raw = base64.b64decode(b64_audio)
    with open(out_path, "wb") as f:
        f.write(raw)


def post_json_with_retry(url: str, payload: Dict[str, Any], *, retries: int = 5, backoff: float = 1.5) -> requests.Response:
    attempt = 0
    while True:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 429:
            return resp
        attempt += 1
        if attempt > retries:
            return resp
        sleep_s = backoff ** attempt
        time.sleep(sleep_s)


def build_generation_args() -> Dict[str, Any]:
    return {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
    }


def run_speaker(base_url: str, prompt_path: str, text: str, out_dir: str) -> str:
    url = f"{base_url.rstrip('/')}/synthesize/speaker"
    payload_wire = {
        "prompt_audio": read_file_as_base64(prompt_path),
        "text": text,
        "max_text_tokens_per_segment": 120,
        "generation_args": build_generation_args(),
    }
    resp = post_json_with_retry(url, payload_wire)
    if resp.status_code != 200:
        raise RuntimeError(f"speaker failed: {resp.status_code} {resp.text}")
    b64_audio = resp.json()
    out_path = os.path.join(out_dir, "speaker.wav")
    write_wav_base64(b64_audio, out_path)
    return out_path


def run_reference(base_url: str, prompt_path: str, emotion_path: str, text: str, out_dir: str, *, emotion_weight: float = 0.8) -> str:
    url = f"{base_url.rstrip('/')}/synthesize/reference"
    payload_wire = {
        "prompt_audio": read_file_as_base64(prompt_path),
        "emotion_audio": read_file_as_base64(emotion_path),
        "emotion_weight": float(emotion_weight),
        "text": text,
        "max_text_tokens_per_segment": 120,
        "generation_args": build_generation_args(),
    }
    resp = post_json_with_retry(url, payload_wire)
    if resp.status_code != 200:
        raise RuntimeError(f"reference failed: {resp.status_code} {resp.text}")
    b64_audio = resp.json()
    out_path = os.path.join(out_dir, "reference.wav")
    write_wav_base64(b64_audio, out_path)
    return out_path


def run_vector(
    base_url: str,
    prompt_path: str,
    text: str,
    out_dir: str,
    *,
    emotion_factors: Dict[str, float],
    emotion_random: bool = False,
) -> str:
    url = f"{base_url.rstrip('/')}/synthesize/vector"
    payload_wire = {
        "prompt_audio": read_file_as_base64(prompt_path),
        "text": text,
        "max_text_tokens_per_segment": 120,
        "emotion_factors": emotion_factors,
        "emotion_random": bool(emotion_random),
        "generation_args": build_generation_args(),
    }
    resp = post_json_with_retry(url, payload_wire)
    if resp.status_code != 200:
        raise RuntimeError(f"vector failed: {resp.status_code} {resp.text}")
    b64_audio = resp.json()
    out_path = os.path.join(out_dir, "vector.wav")
    write_wav_base64(b64_audio, out_path)
    return out_path


def run_text(
    base_url: str,
    prompt_path: str,
    emotion_text: Optional[str],
    text: str,
    out_dir: str,
    *,
    emotion_random: bool = False,
) -> str:
    url = f"{base_url.rstrip('/')}/synthesize/text"
    payload_wire = {
        "prompt_audio": read_file_as_base64(prompt_path),
        "text": text,
        "max_text_tokens_per_segment": 120,
        "emotion_text": emotion_text,  # key must exist; can be null
        "emotion_random": bool(emotion_random),
        "generation_args": build_generation_args(),
    }
    resp = post_json_with_retry(url, payload_wire)
    if resp.status_code != 200:
        raise RuntimeError(f"text failed: {resp.status_code} {resp.text}")
    b64_audio = resp.json()
    out_path = os.path.join(out_dir, "text.wav")
    write_wav_base64(b64_audio, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test IndexTTS HTTP endpoints (4 fixed scenarios)")
    parser.add_argument("--base-url", default="http://localhost:9010", help="Base URL of the HTTP service")
    parser.add_argument("--examples-dir", default=os.path.join(os.path.dirname(__file__), "examples"), help="Path to examples dir")
    parser.add_argument("--outputs-dir", default=os.path.join(os.path.dirname(__file__), "outputs"), help="Path to outputs dir")
    args = parser.parse_args()

    examples_dir = args.examples_dir
    outputs_dir = args.outputs_dir
    ensure_dir(outputs_dir)

    # Case 1: speaker
    prompt1 = os.path.join(examples_dir, "voice_01.wav")
    text1 = "Translate for me, what is a surprise!"
    if not os.path.isfile(prompt1):
        print(f"Missing file: {prompt1}", file=sys.stderr)
        sys.exit(1)
    print("Running: case1 speaker…")
    out1 = run_speaker(args.base_url, prompt1, text1, outputs_dir)
    print(f"Saved: {out1}")

    # Case 2: reference
    prompt2 = os.path.join(examples_dir, "voice_02.wav")
    emo2 = os.path.join(examples_dir, "emo_hate.wav")
    text2 = "你看看你，对我还有没有一点父子之间的信任了。"
    if not os.path.isfile(prompt2):
        print(f"Missing file: {prompt2}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(emo2):
        print(f"Missing file: {emo2}", file=sys.stderr)
        sys.exit(1)
    print("Running: case2 reference…")
    out2 = run_reference(args.base_url, prompt2, emo2, text2, outputs_dir, emotion_weight=1.0)
    print(f"Saved: {out2}")

    # Case 3: vector (surprised=1.0)
    prompt3 = os.path.join(examples_dir, "voice_03.wav")
    text3 = "哇塞！这个爆率也太高了！欧皇附体了！"
    if not os.path.isfile(prompt3):
        print(f"Missing file: {prompt3}", file=sys.stderr)
        sys.exit(1)
    print("Running: case3 vector…")
    out3 = run_vector(
        args.base_url,
        prompt3,
        text3,
        outputs_dir,
        emotion_factors={
            "happy": 0.0,
            "angry": 0.0,
            "sad": 0.0,
            "afraid": 0.0,
            "disgusted": 0.0,
            "melancholic": 0.0,
            "surprised": 1.0,
            "calm": 0.0,
        },
        emotion_random=False,
    )
    print(f"Saved: {out3}")

    # Case 4: text (emo_text)
    prompt4 = os.path.join(examples_dir, "voice_04.wav")
    emo_text4 = "极度悲伤"
    text4 = "这些年的时光终究是错付了... "
    if not os.path.isfile(prompt4):
        print(f"Missing file: {prompt4}", file=sys.stderr)
        sys.exit(1)
    print("Running: case4 text…")
    out4 = run_text(args.base_url, prompt4, emo_text4, text4, outputs_dir, emotion_random=False)
    print(f"Saved: {out4}")

    print("Done.")


if __name__ == "__main__":
    main()
