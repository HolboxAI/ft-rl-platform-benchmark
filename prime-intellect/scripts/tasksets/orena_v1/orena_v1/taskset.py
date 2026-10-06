"""orena-v1: ORena surgical VQA (single-turn, image + question -> short answer).

The model sees one image and a question and answers in the requested format.
Reward = exact match after normalization (number rows tolerate ±0.01), matching
FIREWORKS_4EXP_CONFIG.md §8.1 and rewards.py.

VLM wire format: verifiers v1 carries images as OpenAI ``image_url`` content
parts ({"type":"image_url","image_url":{"url":"..."}}). The ORena rows store the
image as {"type":"image","image":"data:image/jpeg;base64,..."}, so we convert to
a typed ImageUrlContentPart here; the null harness then emits the native OpenAI
shape that the vLLM AsyncOpenAI client (and the qwen3.5 renderer) expect.
"""

import json
import re

import verifiers.v1 as vf


def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip(".,;:!?\"'` ")


def number_ok(pred: str, gold: str, tol: float = 0.01) -> bool:
    try:
        p = float(re.sub(r"[^0-9.\-]", "", pred or "") or "nan")
        g = float(re.sub(r"[^0-9.\-]", "", gold or "") or "nan")
        return abs(p - g) <= tol
    except ValueError:
        return False


def extract_answer(gen: str) -> str:
    if "</think>" in gen:
        gen = gen.split("</think>")[-1]
    # VLM rows in the parent repo also use an "Answer: <value>" tail; prefer it.
    m = re.search(r"Answer:\s*(.*?)\s*$", gen, flags=re.S)
    if m:
        return m.group(1).strip()
    return gen.strip()


def exact_score(pred_text: str, gt: dict) -> float:
    fmt = gt.get("answer_format", "open_ended")
    gold = normalize(str(gt.get("answer", "")))
    pred = normalize(pred_text)
    if fmt == "number":
        return 1.0 if number_ok(pred, gold) else 0.0
    return 1.0 if pred == gold else 0.0


def _user_prompt(content):
    """ORena user content (str or [image/text parts]) -> a verifiers prompt."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append(vf.TextContentPart(text=p.get("text", "")))
        elif p.get("type") == "image":
            # data:image/jpeg;base64,....  The qwen3.5 renderer's _load_pil_image
            # handles data: URIs directly, so pass the string through as the url.
            parts.append(
                vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=p.get("image", "")))
            )
    return [vf.UserMessage(content=parts)]


class ORenaData(vf.TaskData):
    ground_truth: dict


class ORenaTask(vf.Task[ORenaData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # Single-turn VQA: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        reply = extract_answer(trace.last_reply or "")
        return exact_score(reply, self.data.ground_truth)


class ORenaConfig(vf.TasksetConfig):
    data_path: str = "/workspace/data/orena/rl/train.jsonl"
    """Local JSONL path. Each row: {messages: [system, user], ground_truth: {answer, answer_format, choices}}."""


class ORenaTaskset(vf.Taskset[ORenaTask, ORenaConfig]):
    def load(self) -> list[ORenaTask]:
        rows = [json.loads(line) for line in open(self.config.data_path)]
        tasks = []
        for i, r in enumerate(rows):
            sys_msg = next(
                (m["content"] for m in r["messages"] if m["role"] == "system"), ""
            )
            user_msg = next(
                (m["content"] for m in r["messages"] if m["role"] == "user"), ""
            )
            tasks.append(
                ORenaTask(
                    ORenaData(
                        idx=i,
                        prompt=_user_prompt(user_msg),
                        system_prompt=sys_msg,
                        ground_truth=r.get("ground_truth", {}),
                    ),
                    self.config.task,
                )
            )
        return tasks
