"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from mars.model.ea_model import EaModel
    from mars.model.kv_cache import initialize_past_key_values
    from mars.model.utils import *

# ---------------------- New: prompt builder helpers ---------------------- #
def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _extract_signature(q):
    # Try common keys used across benchmarks
    sig = _coalesce(
        q.get("signature"),
        q.get("entry_point"),
        q.get("func_signature"),
        q.get("function_signature"),
        default=None
    )
    if isinstance(sig, dict):
        # some datasets store signature under dict like {"name":..., "args":[...]}
        name = sig.get("name", "solution")
        args = sig.get("args", [])
        return f"{name}({', '.join(args)})"
    if isinstance(sig, str) and sig.strip():
        return sig.strip()
    # fallback: try to guess from text (very conservative)
    return "solution"

def _extract_problem_text(q):
    # Prefer explicit fields
    ptxt = _coalesce(
        q.get("prompt"),
        q.get("text"),
        q.get("problem"),
        q.get("question"),
        default=None
    )
    if ptxt is not None:
        return str(ptxt).strip()

    # If it's a multi-turn item, join user turns as problem statement
    turns = q.get("turns")
    if isinstance(turns, list) and turns:
        # Prefer the first user turn as "problem"
        if isinstance(turns[0], str):
            return turns[0].strip()
        try:
            return "\n\n".join([str(t).strip() for t in turns if isinstance(t, str)])
        except Exception:
            pass

    # Last resort: dump the whole object (not ideal, but keeps pipeline running)
    return json.dumps(q, ensure_ascii=False)

def build_chat_messages(tokenizer, q):
    signature = _extract_signature(q)
    problem_text = _extract_problem_text(q)

    # Your template (verbatim)
    prompt = (
        f"Write Python code to solve the task.\n"
        f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
        f"```python\n"
        f"{problem_text}\n"
        f"```\n"
    )

    # Load system prompt from file; fallback to safe default if not found
    system_prompt_path = os.path.join(script_dir, "system_prompt.md")
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        system_prompt = (
            "You are a meticulous coding assistant. Generate correct, efficient, and readable Python code. "
            "Prefer pure-Python standard library solutions unless otherwise necessary. "
            "Avoid extra commentary; focus on the final working implementation."
        )

    msg = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
        },
    ]

    chat = tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat
# ------------------------------------------------------------------------ #


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    shuffled_ids = [q["question_id"] for q in questions]

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) if (num_gpus_total // num_gpus_per_model) > 0 else len(questions)
    if chunk_size == 0:
        chunk_size = len(questions)

    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_mars=args.use_mars,
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # ---------------------- Warmup with the new prompt ---------------------- #
    question = questions[0]
    for _ in range(3):
        torch.manual_seed(0)

        chat = build_chat_messages(tokenizer, question)
        input_ids = tokenizer([chat], add_special_tokens=False).input_ids

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
            log=True,
            is_llama3=True,
        )
        torch.cuda.synchronize()
        _ = time.time() - start_time

        output_ids = output_ids[0][len(input_ids[0]):]
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        if stop_token_ids:
            stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in stop_token_ids]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        _ = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    print('Warmup done')

    # ---------------------- Main evaluation loop ---------------------- #
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)

            chat = build_chat_messages(tokenizer, question)
            input_ids = tokenizer([chat], add_special_tokens=False).input_ids

            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, idx = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time

            output_ids = output_ids[0][len(input_ids[0]):]
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if stop_token_ids:
                stop_token_ids_index = [ii for ii, id in enumerate(output_ids) if id in stop_token_ids]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
            # Remove special tokens seen in some chat templates
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            # For compatibility with your previous JSON schema:
            choices.append({
                "index": i,
                "turns": [output],          # Single-shot code generation per question
                "idxs": [int(idx)],
                "new_tokens": [int(new_token)],
                "wall_time": [total_time],
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "question_id": question.get("question_id", shortuuid.uuid()),
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="YOUR_EA_MODEL_PATH",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="YOUR_MODEL_PATH",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="humaneval",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="total-token = The total number of drafted tokens in the tree + 1",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="depth = The maximum number of draft length - 1",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of drafted tokens in each layer.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default=80,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use_mars",
        action="store_true"
    )

    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
