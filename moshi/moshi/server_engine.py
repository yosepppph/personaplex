# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Multi-client WebSocket server backed by the continuous-batching engine (Phase 4).

Unlike the single-user `server.py` (one LMGen + global lock, one conversation at a
time), this serves many simultaneous conversations from ONE shared BatchedEngine
running a single batched GPU tick on a worker thread. There is NO global lock: every
WebSocket connection is just a slot producer/consumer on the shared event loop.

Per connection:
  - acquire() a slot (reject with a close if the engine is full),
  - recv_loop : WS opus bytes -> opus_reader,
  - feed_loop : opus_reader PCM -> frame_size chunks -> engine.submit_pcm(idx, ..),
  - out_loop  : slot.out_q (pcm, text_token) -> opus_writer + text -> WS,
  - release() the slot on disconnect.

Per-slot conditioning on join:
  - Phase 3a: the recipe/system text prompt (?text_prompt=) is injected per slot.
  - Phase 3b: the voice prompt (?voice_prompt=) is teacher-forced per slot when
    --voice-prompt-dir points at a directory of .wav voice clips. Without that flag
    (or for voices that only exist as .pt), every connection gets the default voice.
    The .pt embeddings cannot be teacher-forced per slot while other slots stream,
    so the batched engine needs .wav voice clips.

Run (TurboQuant env flags are REQUIRED by the engine):
  PERSONAPLEX_TURBOQUANT_KV=1 PERSONAPLEX_TURBOQUANT_FUSED=1 \
      python -m moshi.server_engine --port 8998 --device cuda --max-slots 16 \
      --voice-prompt-dir /path/to/wav_voices
"""

import argparse
import asyncio
import os
from pathlib import Path
import random
import tarfile
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from .batched_engine import BatchedEngine
from .models import loaders
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]


def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def wrap_with_system_tags(text: str) -> str:
    """Add <system> ... <system> tags as the model expects if missing."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class EngineServer:
    """Transport layer: maps each WebSocket connection onto one engine slot."""

    def __init__(self, engine: BatchedEngine,
                 text_tokenizer: sentencepiece.SentencePieceProcessor,
                 voice_mimi=None,
                 voice_prompt_dir: Optional[str] = None):
        self.engine = engine
        self.text_tokenizer = text_tokenizer
        self.sample_rate = engine.sample_rate
        self.frame_size = engine.frame_size
        # Phase 3b: per-slot voice prompt. `voice_mimi` is a DEDICATED Mimi used only
        # to encode voice WAVs to codes (separate from the engine's streaming mimi).
        # `voice_prompt_dir` holds the .wav voice clips. Codes are cached per voice.
        self.voice_mimi = voice_mimi
        self.voice_prompt_dir = voice_prompt_dir
        self._voice_codes_cache: dict = {}

    def _get_voice_codes(self, voice_name: str):
        """Resolve a voice name to Mimi codes [8, Tv], encoding its .wav once and
        caching the result. Returns None (default voice) when voice support is off,
        the name is empty, or only a .pt (non-WAV) file exists -- the batched engine
        teacher-forces audio codes, which can only come from a WAV."""
        if not voice_name or self.voice_mimi is None or self.voice_prompt_dir is None:
            return None
        if voice_name in self._voice_codes_cache:
            return self._voice_codes_cache[voice_name]
        base = os.path.join(self.voice_prompt_dir, voice_name)
        wav_path = None
        for cand in (base, base + ".wav", os.path.splitext(base)[0] + ".wav"):
            if cand.endswith(".wav") and os.path.exists(cand):
                wav_path = cand
                break
        if wav_path is None:
            logger.warning(
                f"voice '{voice_name}': no .wav found in {self.voice_prompt_dir}; "
                f"using the default voice. The batched engine needs a .wav voice clip "
                f"(the packaged .pt embeddings cannot be teacher-forced per slot).")
            self._voice_codes_cache[voice_name] = None
            return None
        try:
            codes = self.engine.encode_voice_wav(self.voice_mimi, wav_path)
        except Exception as e:
            logger.warning(f"voice '{voice_name}': failed to encode {wav_path} ({e}); "
                           f"using the default voice.")
            codes = None
        self._voice_codes_cache[voice_name] = codes
        return codes

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        clog.log("info", f"incoming connection from {request.remote}")

        # Phase 3a: inject the recipe/system text prompt per slot on join.
        text_prompt = request.query.get("text_prompt", "")
        text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
            if text_prompt else None)

        # Phase 3b: per-slot voice prompt. Resolve the voice name to Mimi codes (None
        # -> default voice). Encoding is cached, so only the first connection per voice
        # pays the cost (a brief one-time hiccup on the event loop).
        voice_name = request.query.get("voice_prompt", "")
        voice_codes = self._get_voice_codes(voice_name)

        idx = self.engine.acquire(text_prompt_tokens=text_prompt_tokens,
                                  voice_codes=voice_codes)
        if idx is None:
            clog.log("warning", "engine full -> rejecting connection")
            await ws.close(code=aiohttp.WSCloseCode.TRY_AGAIN_LATER,
                           message=b"server full")
            return ws
        clog.log("info", f"assigned slot {idx} (active={self.engine.active_count()}, "
                          f"prompt_tokens={len(text_prompt_tokens) if text_prompt_tokens else 0}, "
                          f"voice={voice_name or 'default'}"
                          f"{'' if voice_codes is not None else ' (default)'})")

        opus_reader = sphn.OpusStreamReader(self.sample_rate)
        opus_writer = sphn.OpusStreamWriter(self.sample_rate)
        slot = self.engine.slots[idx]
        close = False

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type in (aiohttp.WSMsgType.CLOSE,
                                          aiohttp.WSMsgType.CLOSED):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    data = message.data
                    if not isinstance(data, bytes) or len(data) == 0:
                        continue
                    if data[0] == 1:  # audio frame
                        opus_reader.append_bytes(data[1:])
                    else:
                        clog.log("warning", f"unknown message kind {data[0]}")
            finally:
                close = True
                clog.log("info", "recv loop closed")

        async def feed_loop():
            # opus -> PCM, sliced into engine-sized frames and pushed to the slot.
            pending = np.zeros(0, dtype=np.float32)
            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                pending = np.concatenate((pending, pcm)) if pending.size else pcm
                while pending.shape[-1] >= self.frame_size:
                    chunk = np.ascontiguousarray(pending[:self.frame_size])
                    pending = pending[self.frame_size:]
                    self.engine.submit_pcm(idx, chunk)

        async def out_loop():
            # Drain this slot's engine output: PCM -> opus, text token -> piece.
            while not close:
                try:
                    pcm_out, text_token = await asyncio.wait_for(
                        slot.out_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                opus_writer.append_pcm(pcm_out)
                if text_token is not None and text_token not in (0, 3):
                    piece = self.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                    await ws.send_bytes(b"\x02" + piece.encode("utf8"))
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        await ws.send_bytes(b"\x00")  # handshake: client may start streaming
        clog.log("info", "sent handshake")
        tasks = [
            asyncio.create_task(recv_loop()),
            asyncio.create_task(feed_loop()),
            asyncio.create_task(out_loop()),
        ]
        try:
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            self.engine.release(idx)
            if not ws.closed:
                await ws.close()
            clog.log("info", f"released slot {idx}, connection closed")
        return ws


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--max-slots", default=16, type=int,
                        help="Number of concurrent conversation slots.")
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Local Moshi checkpoint.")
    parser.add_argument("--mimi-weight", type=str, help="Local Mimi checkpoint.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults to PersonaPlex.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ssl", type=str,
                        help="Directory with key.pem and cert.pem to serve https.")
    parser.add_argument("--voice-prompt-dir", type=str,
                        help="Directory of .wav voice clips for per-slot voice prompts "
                             "(Phase 3b). The client's ?voice_prompt= names a file here. "
                             "Omit to disable voice prompts (default voice for everyone). "
                             "NOTE: must be .wav clips -- the packaged .pt embeddings are "
                             "not usable by the batched engine.")
    args = parser.parse_args()

    if os.environ.get("PERSONAPLEX_TURBOQUANT_KV") != "1" or \
       os.environ.get("PERSONAPLEX_TURBOQUANT_FUSED") != "1":
        raise SystemExit(
            "server_engine requires the fused TurboQuant KV cache. Re-run with:\n"
            "  PERSONAPLEX_TURBOQUANT_KV=1 PERSONAPLEX_TURBOQUANT_FUSED=1 python -m moshi.server_engine ...")

    static_path = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)
    seed_all(42424242)

    # Download config.json (increments the HF download counter; cached after).
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device)
    lm.eval()
    logger.info("moshi loaded")

    engine = BatchedEngine(
        mimi=mimi, lm=lm, device=args.device, max_slots=args.max_slots,
        sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
    logger.info(f"warming up the engine (max_slots={args.max_slots})")
    engine.warmup()

    # Phase 3b: a dedicated Mimi (batch 1) used only to encode voice-prompt WAVs into
    # codes -- kept separate from the engine's max_slots streaming mimi so encoding a
    # voice never disturbs the live batch. Only created when voice prompts are enabled.
    voice_mimi = None
    if args.voice_prompt_dir is not None:
        if not os.path.isdir(args.voice_prompt_dir):
            raise SystemExit(f"--voice-prompt-dir does not exist: {args.voice_prompt_dir}")
        logger.info(f"voice prompts enabled from {args.voice_prompt_dir}")
        voice_mimi = loaders.get_mimi(args.mimi_weight, args.device)
        voice_mimi.streaming_forever(1)

    server = EngineServer(engine, text_tokenizer,
                          voice_mimi=voice_mimi,
                          voice_prompt_dir=args.voice_prompt_dir)

    app = web.Application()

    async def _start_engine(app):
        app["engine_task"] = asyncio.create_task(engine.run())
        logger.info("engine tick loop started")

    async def _stop_engine(app):
        engine.stop()
        task = app.get("engine_task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("engine tick loop stopped")

    app.on_startup.append(_start_engine)
    app.on_cleanup.append(_stop_engine)

    app.router.add_get("/api/chat", server.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI at {protocol}://{host_ip}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
