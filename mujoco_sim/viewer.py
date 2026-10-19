# -*- coding: utf-8 -*-
"""``--mode play``。方策を見る、または提出動画を録る。

1つの経路から2つの出力。動画が、査読者が対話的に再現できるものと
同じものを映す必要があるから。

  * ``--video`` 無し  : ``mujoco.viewer.launch_passive``。本物の対話
    ウィンドウ（旋回、ズーム、接触力、関節フレーム）。WSL では WSLg を
    通して追加設定なしで動く。
  * ``--video out.mp4`` : 手首カメラの小窓と、方策名・エピソード・結果・
    途中までの成功率を書いた字幕帯を重ねたオフスクリーン録画。

字幕帯は飾りではない。課題は「表現力: 何が起きているかをどれだけ
分かりやすく説明できているか」も採点する。そして、
アームが動くのを見ているだけの人には、それがスクリプトエキスパート
なのか、複製した方策なのか、蒸留した画像方策なのかを知る手段が無い。
エピソードが成功したのかも分からない。主張と途中までの点数を画面に
出すことが、デモと証拠の違いになる。
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

BAR_H = 120        # 字幕3行と、主張の行が2行に折れる余地
FG = (236, 240, 244)
DIM = (150, 160, 172)
OK = (110, 214, 138)
BAD = (240, 122, 122)


def _pick_policy(env, cfg, onnx_path: str | None):
    """-> (act_fn, reset_hook, label)。明示したパスが優先。そうでなければ
    実際に出荷する成果物（視覚運動方策の ONNX）、次に RL の教師、次に
    BC 方策を選び、最後はスクリプトエキスパートに落ちる。だから学習済み
    モデルが何も無いクローンでも ``--mode play`` が動く。"""
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


# ここの切れ目で行が折れてよいが、日本語の組版では行頭に閉じ括弧や
# 句読点、行末に開き括弧が来る折り方は禁じられる。
_NO_ROW_START = "、。，．）〕］｝〉》」』】・…‥ー？！?!,.:;)]}%"
_NO_ROW_END = "（〔［｛〈《「『【([{"
_FONT_FILE = None

# ラテン文字、数字、小数点、カードが使う記号、かなと漢字。フォントは
# この全部を描けないといけない。Linux 機にいちばん転がっている CJK
# フォントの DroidSansFallback は補助用の書体でラテン文字を持たないので、
# 存在だけで選ぶと「MimicGen 0.885」が全部豆腐になる。
_PROBE = "A0.5×→、あ漢｜"


def _covers(path: str) -> bool:
    """``path`` のフォントは ``_PROBE`` の全文字を実際に描けるか。"""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype(path, 24, index=0)
    except Exception:
        return False

    def bitmap(ch: str) -> bytes:
        img = Image.new("L", (48, 48))
        ImageDraw.Draw(img).text((4, 4), ch, fill=255, font=f)
        return img.tobytes()

    notdef = bitmap("\uFFFF")      # 持っていない文字に対してフォントが描く豆腐
    return all(bitmap(c) != notdef for c in _PROBE)


def _font_file() -> str:
    """日本語を描けるフォントのパス。プロセスにつき1回だけ解決する。

    Pillow の既定書体はラテン文字しか持たないので、字幕のかなと漢字は
    全部白い四角になる。提出動画は日本語なので、CJK 書体はあると嬉しい
    ものではなく必須条件。WSL ではホストの Windows フォントが /mnt/c から
    読めるので、開発に使っているノートPCでも、fonts-noto-cjk を入れた
    Linux 機でも同じように動く。探索の順番は下のリストのとおり。
    見つからなければ例外にする。
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
                "/mnt/c/Windows/Fonts/meiryo.ttc",          # WSL: ホスト側のフォント
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
    """日本語が描ける可変サイズのフォント。サイズごとにキャッシュする。

    キャッシュするのは、1フレームの字幕1行ごとに要求されるから。
    約20MBの .ttc を1万2千回開き直すと、2分の描画が仮眠になる。
    """
    from PIL import ImageFont
    path = _font_file()
    if path:
        try:
            return ImageFont.truetype(path, size, index=0)
        except Exception as exc:                 # 読めない、または TrueType でない場合は次へ
            print(f"[viewer] WARNING: cannot load {path}: {exc}")
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                        # Pillow < 10.1
        return ImageFont.load_default()


def _pieces(text: str) -> list:
    """``text`` を、行を折ってよい最小単位に分解する。

    ラテン文字は空白で折れるが、日本語に空白は無いので、CJK の1文字を
    それぞれ1単位にする。空白だけで分けると（以前はそうだった）、
    日本語の字幕は40文字の折れない1トークンになり、フレームの右端から
    そのまま飛び出していく。
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
    """``text`` を ``max_w`` ピクセルに収まる行に折る。

    フレームの右端から出ていく字幕は字幕ではない。提出した動画を見る人は
    文の終わりを黙って失い、しかもそこが主張を担っている半分。
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
    """各クリップの下に出す字幕帯。"""
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
    """シーン視点に、右下へ手首カメラを差し込み、字幕帯を付けたもの。"""
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
                outcome_age = fps            # 判定を1秒画面に残す
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
            # 実時間の速さで再生する。画面に出ているものが、20Hzでロボットが
            # やることそのものになるようにする。早送りにはしない
            time.sleep(max(0.0, env.control_dt - (time.perf_counter() - t)))
    env.close()
