import argparse
import json
import os
import re
import time
from openai import OpenAI

# Локальные модели в Ollama (если скачал легкие — замени на "qwen2.5:3b" и "deepseek-r1:1.5b")
MODELS = {
    "qwen25": "qwen2.5:7b",
    "deepseek_r1": "deepseek-r1:7b",
}


def split_thinking(raw_text: str) -> tuple[str, str]:
    """Отделяет блок рассуждений <think>...</think> у DeepSeek-R1 от итогового ответа."""
    match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    thinking = match.group(1).strip() if match else ""
    answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return thinking, answer


def check_json(raw_text: str) -> dict:
    """Проверяет, является ли ответ строгим JSON или обернут в markdown."""
    cleaned = raw_text.strip()
    strict_valid = False
    parseable_valid = False

    try:
        parsed_obj = json.loads(cleaned)
        strict_valid = isinstance(parsed_obj, dict)
        parseable_valid = strict_valid
    except json.JSONDecodeError:
        if "```" in cleaned:
            for part in cleaned.split("```"):
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    try:
                        parsed_obj = json.loads(candidate)
                        if isinstance(parsed_obj, dict):
                            parseable_valid = True
                            break
                    except json.JSONDecodeError:
                        pass

    return {
        "strict_json": strict_valid,
        "parseable_json": parseable_valid,
    }


def run_benchmark(prompt_path: str, version_name: str, temperature: float, top_p: float, target_model: str, limit: int):
    # Подключаемся к локальному серверу Ollama (OpenAI-совместимый API)
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open("data/agro_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    if limit > 0:
        cases = cases[:limit]

    os.makedirs("results", exist_ok=True)
    selected = MODELS if target_model == "all" else {target_model: MODELS[target_model]}

    for model_alias, model_id in selected.items():
        print(f"\n=== Модель: {model_alias} ({model_id}) | {version_name} | temp={temperature} ===")
        logs = []

        for case in cases:
            case_id = case["id"]
            start_time = time.time()

            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": case["user_query"]},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=2048,
                )
                elapsed = round(time.time() - start_time, 2)
                raw_content = response.choices[0].message.content or ""
                thinking, answer = split_thinking(raw_content)
                jstats = check_json(answer)

                logs.append({
                    "id": case_id,
                    "category": case["category"],
                    "must_ask_clarifying": case["must_ask_clarifying"],
                    "user_query": case["user_query"],
                    "thinking": thinking,
                    "response": answer,
                    "latency_sec": elapsed,
                    "strict_json": jstats["strict_json"],
                    "parseable_json": jstats["parseable_json"],
                    "error": None,
                })
                print(f"[OK] #{case_id:02d} ({elapsed}s) | strict={jstats['strict_json']} | parseable={jstats['parseable_json']}")

            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                logs.append({
                    "id": case_id,
                    "category": case["category"],
                    "must_ask_clarifying": case["must_ask_clarifying"],
                    "user_query": case["user_query"],
                    "thinking": "",
                    "response": None,
                    "latency_sec": elapsed,
                    "strict_json": False,
                    "parseable_json": False,
                    "error": str(e),
                })
                print(f"[ERR] #{case_id:02d}: {e}")

        if limit > 0:
            print("Тестовый режим (--limit): файл результатов не сохранялся.")
            continue

        payload = {
            "meta": {
                "model_alias": model_alias,
                "model_id": model_id,
                "prompt_version": version_name,
                "temperature": temperature,
                "top_p": top_p,
            },
            "results": logs,
        }

        out_file = f"results/{version_name}_{model_alias}_t{str(temperature).replace('.', '')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Сохранено: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бенчмарк промптов для ИИ-ассистента АПК")
    parser.add_argument("--prompt", default="prompts/v1_baseline.txt")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--model", choices=["all", "qwen25", "deepseek_r1"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число кейсов для быстрого теста")
    args = parser.parse_args()

    run_benchmark(args.prompt, args.version, args.temp, args.top_p, args.model, args.limit)