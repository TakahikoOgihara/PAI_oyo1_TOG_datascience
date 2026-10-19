# -*- coding: utf-8 -*-
"""``--mode play``: watch the policy, or record the submission video.

Two outputs from one code path, which matters because the video has to show
the same thing the reviewer can reproduce interactively:

  * no ``--video``  : ``mujoco.viewer.launch_passive`` -- a real interactive
    window (orbit, zoom, contact forces, joint frames). Under WSL this works
    through WSLg with no extra setup.
  * ``--video out.mp4`` : an offscreen recording with the wrist-camera view
    inset and a caption strip that names the policy, the episode, the outcome
    and the running success rate.

The caption strip is not decoration. The assignment is graded partly on
"表現力: how clearly the video explains what is happening", and a viewer
watching an arm move has no way to know whether they are looking at a scripted
expert, a cloned policy or a distilled image policy -- nor whether the episode
succeeded. Putting the claim and the running score on screen is the difference
between a demo and evidence.
"""

from __future__ import annotations

import functools
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mujoco_sim.env import MuJoCoPickPlaceEnv  # noqa: E402
from mujoco_sim.policies import ExpertController, OnnxPolicy  # noqa: E402

BAR_H = 120        # 3 caption rows, and room for the claim line to wrap to 2
FG = (236, 240, 244)
DIM = (150, 160, 172)
OK = (110, 214, 138)
BAD = (240, 122, 122)


def _pick_policy(env, cfg, onnx_path: str | None):
    """-> (act_fn, reset_hook, label). Explicit path wins; otherwise prefer the
    artifact that actually ships (the visuomotor ONNX), then the RL teacher,
    then the BC policy, and fall back to the scripted expert so ``--mode play``
    works on a fresh checkout with no trained model at all."""
    deploy = os.path.join(REPO_ROOT, "artifacts", "deploy")
    short = "dp" if str(cfg.visuomotor.type) == "diffusion" else "act"
    candidates = [onnx_path] if onnx_path else [
        os.path.join(deploy, f"arm_bcrl006_{short}_policy.onnx"),
        os.path.join(deploy, "arm_bcrl006_rl_policy.onnx"),
        os.path.join(deploy, "arm_bcrl006_bc_policy.onnx"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            pol = OnnxPolicy(p, env, execute_horizon=int(cfg.visuomotor.execute_horizon))
            label = {"act": "ACT 視覚運動方策（手首カメラ）",
                     "diffusion": "Diffusion Policy（手首カメラ）"}.get(
                         pol.policy_type, "状態入力方策（ONNX）")
            return pol, pol, f"{label} - {os.path.basename(p)}"
    expert = ExpertController(env)
    return (lambda o: expert.act()), expert, "スクリプトIKエキスパート（学習済み方策が無いため）"


# A line may break between these pieces, but Japanese typesetting forbids a
# row that STARTS with closing punctuation or ENDS with an opening bracket.
_NO_ROW_START = "、。，．）〕］｝〉》」』】・…‥ー？！?!,.:;)]}%"
_NO_ROW_END = "（〔［｛〈《「『【([{"
_FONT_FILE = None

# Latin, digits, decimal point, the symbols the cards use, kana and kanji. A
# font has to draw ALL of these: DroidSansFallback, the CJK font most likely
# to be sitting on a Linux box, is a *fallback* face with no Latin at all, so
# picking it by existence alone turns every "MimicGen 0.885" into empty boxes.
_PROBE = "A0.5×→、あ漢｜"


def _covers(path: str) -> bool:
    """Does the font at ``path`` actually draw every character in ``_PROBE``?"""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype(path, 24, index=0)
    except Exception:
        return False

    def bitmap(ch: str) -> bytes:
        img = Image.new("L", (48, 48))
        ImageDraw.Draw(img).text((4, 4), ch, fill=255, font=f)
        return img.tobytes()

    notdef = bitmap("\uFFFF")      # the box a font draws for what it lacks
    return all(bitmap(c) != notdef for c in _PROBE)


def _font_file() -> str:
    """Path to a font that can draw Japanese, resolved once per process.

    Pillow's built-in default face is Latin-only, so every kana and kanji in
    the captions would come out as a blank box. The submission video is in
    Japanese, which makes a CJK face a hard requirement rather than a nicety.
    Under WSL the host's Windows fonts are readable through /mnt/c, so this
    works on the laptop the project is developed on as well as on a Linux box
    with fonts-noto-cjk installed.
    """
    global _FONT_FILE
    if _FONT_FILE is not None:
        return _FONT_FILE
    import glob
    candidates = [c for c in [os.environ.get("ARM_BCRL006_FONT", "")] if c]
    for pat in ("/usr/share/fonts/**/NotoSansCJK*.?t?",
                "/usr/share/fonts/**/NotoSansJP*.?t?",
                "/usr/share/fonts/**/ipag*.?t?",
                "/usr/share/fonts/**/*japanese-gothic*",
                "/mnt/c/Windows/Fonts/meiryo.ttc",          # WSL: host fonts
                "/mnt/c/Windows/Fonts/YuGothR.ttc",
                "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc",
                "/mnt/c/Windows/Fonts/msgothic.ttc",
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/usr/share/fonts/**/DroidSansFallback*.?t?"):
        candidates += sorted(glob.glob(pat, recursive=True))
    _FONT_FILE = next((c for c in candidates
                       if os.path.exists(c) and _covers(c)), "")
    if not _FONT_FILE:
        print("[viewer] WARNING: found no font that draws both Japanese and "
              "Latin -- the captions will have blank boxes in them. Install "
              "fonts-noto-cjk, or point ARM_BCRL006_FONT at a .ttf/.ttc that "
              "covers Japanese.")
    else:
        print(f"[viewer] font: {_FONT_FILE}")
    return _FONT_FILE


@functools.lru_cache(maxsize=32)
def font(size: int):
    """A scalable Japanese-capable font, cached per size.

    Cached because it is asked for once per caption row per frame: reopening a
    ~20 MB .ttc twelve thousand times turns a 2-minute render into a nap.
    """
    from PIL import ImageFont
    path = _font_file()
    if path:
        try:
            return ImageFont.truetype(path, size, index=0)
        except Exception as exc:                 # unreadable/!TrueType -> fall back
            print(f"[viewer] WARNING: cannot load {path}: {exc}")
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                        # Pillow < 10.1
        return ImageFont.load_default()


def _pieces(text: str) -> list:
    """Split ``text`` into the atoms a line break may fall between.

    Latin text breaks at spaces; Japanese has no spaces, so every CJK
    character is its own atom. Splitting only on spaces -- which is what this
    used to do -- leaves a Japanese caption as one unbreakable 40-character
    token that runs straight off the right edge of the frame.
    """
    out, buf = [], ""
    for ch in text:
        if ("\u3000" <= ch <= "\u303f" or "\u3040" <= ch <= "\u30ff"
                or "\u3400" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"):
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
            if ch == " ":
                out.append(buf)
                buf = ""
    return out + [buf] if buf else out


def wrap(draw, text: str, f, max_w: int) -> list:
    """Break ``text`` into rows that fit ``max_w`` pixels.

    A caption that runs off the right edge of the frame is not a caption --
    the viewer of the submitted video silently loses the end of the sentence,
    which is exactly the half that carries the claim.
    """
    rows, cur = [], ""
    for piece in _pieces(text):
        if cur and draw.textlength(cur + piece, font=f) > max_w:
            if piece in _NO_ROW_START:       # 「。」で行を始めない: 前の行にぶら下げる
                cur += piece
                continue
            while cur and cur[-1] in _NO_ROW_END:   # 「（」で行を終えない
                piece = cur[-1] + piece
                cur = cur[:-1]
            rows.append(cur.rstrip())
            cur = piece.lstrip()
        else:
            cur += piece
    return rows + [cur.rstrip()] if cur.strip() else rows or [""]


def _caption(width: int, lines: list) -> np.ndarray:
    """The caption strip under each clip."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, BAR_H), (18, 20, 24))
    d = ImageDraw.Draw(img)
    y = 6
    for text, colour, big in lines:
        f = font(21 if big else 17)
        for row in wrap(d, text, f, width - 36):
            d.text((18, y), row, fill=colour, font=f)
            y += 30 if big else 23
    return np.asarray(img)


def _compose(scene: np.ndarray, wrist: np.ndarray, lines: list) -> np.ndarray:
    """Scene view with the wrist camera inset bottom-right + caption strip."""
    h, w = scene.shape[:2]
    inset = wrist.shape[0]
    out = scene.copy()
    y0, x0 = h - inset - 12, w - inset - 12
    out[y0 - 3:y0 + inset + 3, x0 - 3:x0 + inset + 3] = (90, 96, 106)
    out[y0:y0 + inset, x0:x0 + inset] = wrist
    return np.concatenate([out, _caption(w, lines)], axis=0)


def play(cfg, args) -> None:
    import mujoco

    env = MuJoCoPickPlaceEnv(cfg, num_envs=1, with_camera=True, seed=int(cfg.seed))
    act_fn, reset_hook, label = _pick_policy(env, cfg, args.onnx_path)
    obs = env.reset()[0]["policy"]
    reset_hook.reset_idx(torch.arange(1))
    print(f"[play] policy: {label}")

    if not args.video:
        _interactive(env, act_fn, reset_hook, mujoco)
        return

    import imageio.v2 as imageio
    fps = int(round(1.0 / env.control_dt))
    total = int(float(args.video_seconds) * fps)
    renderer = mujoco.Renderer(env.model, 540, 960)
    os.makedirs(os.path.dirname(os.path.abspath(args.video)) or ".", exist_ok=True)
    writer = imageio.get_writer(args.video, fps=fps, quality=8, macro_block_size=1)
    ep = succ = 0
    outcome, outcome_age = "", 0
    t0 = time.perf_counter()
    try:
        for step in range(total):
            wrist = env.camera_rgb_uint8()[0].numpy()
            renderer.update_scene(env.data, camera=int(env.idx.scenecam_id))
            frame = renderer.render()
            rate = f"{succ}/{ep} ({succ / ep:.0%})" if ep else "-"
            lines = [
                ("ARM_BCRL006  myCobot 280 Pi  掴む → 持ち上げる → 5cm 横へ置く   ｜   MuJoCo 20 Hz",
                 FG, True),
                (f"方策: {label}", DIM, False),
                (f"エピソード {ep + 1}　ここまでの成功率: {rate}"
                 + (f"　← {outcome}" if outcome_age > 0 else ""),
                 OK if outcome.startswith("成功") else (BAD if outcome else DIM), False),
            ]
            writer.append_data(_compose(frame, wrist, lines))
            outcome_age = max(0, outcome_age - 1)
            obs_d, _, term, trunc, _ = env.step(act_fn(obs))
            obs = obs_d["policy"]
            if bool(term[0] | trunc[0]):
                ep += 1
                succ += int(term[0])
                outcome = "成功" if bool(term[0]) else "失敗（時間切れ）"
                outcome_age = fps            # hold the verdict on screen for 1 s
                reset_hook.reset_idx(torch.tensor([0]))
            if step % fps == 0:
                print(f"  [play] {step}/{total} frames "
                      f"({step / max(1e-9, time.perf_counter() - t0):.1f} fps)", flush=True)
    finally:
        writer.close()
        renderer.close()
        env.close()
    print(f"[play] wrote {args.video}  ({total / fps:.0f} s at {fps} fps, "
          f"{ep} episodes, {succ} successes)")


def _interactive(env, act_fn, reset_hook, mujoco) -> None:
    import mujoco.viewer
    print("[play] interactive viewer -- close the window to stop")
    obs = env.get_observations()["policy"]
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            t = time.perf_counter()
            obs_d, _, term, trunc, _ = env.step(act_fn(obs))
            obs = obs_d["policy"]
            if bool(term[0] | trunc[0]):
                print(f"[play] episode ended: "
                      f"{'SUCCESS' if bool(term[0]) else 'failure'}")
                reset_hook.reset_idx(torch.tensor([0]))
            v.sync()
            # play at wall-clock speed so what is on screen is what the robot
            # would do at 20 Hz, not a fast-forward
            time.sleep(max(0.0, env.control_dt - (time.perf_counter() - t)))
    env.close()
