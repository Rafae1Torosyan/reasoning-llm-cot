import argparse
import os
import torch
from transformers import pipeline
import time
import re


def load_prompts(input_file: str) -> list:
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def extract_final_number(text: str) -> int:
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else -1


def generate_responses(prompts: list, model_name: str, max_tokens: int, temperature: float) -> list:
    os.environ["HF_HOME"] = "D:/hf-cache"
    os.environ["TORCH_HOME"] = "D:/hf-cache"

    print(f"🚀 Loading model: {model_name}")
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    responses = []
    for i, prompt in enumerate(prompts):
        print(f"\n🔄 [{i + 1}/{len(prompts)}] Prompt: {prompt}")
        input_text = prompt + "\nLet's solve this step by step:"
        try:
            start = time.time()
            result = pipe(
                input_text,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=temperature
            )[0]["generated_text"]
            duration = time.time() - start
            print(f"✅ Done in {duration:.1f} sec")
            responses.append(result.strip())
        except Exception as e:
            print(f"⚠️ Error: {e}")
            responses.append("ERROR")
    return responses


def save_and_evaluate(prompts: list, responses: list, expected_answers: list, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    correct = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (prompt, response, expected) in enumerate(zip(prompts, responses, expected_answers), 1):
            predicted = extract_final_number(response)
            is_correct = predicted == expected
            if is_correct:
                correct += 1

            f.write(f"--- Example {i} ---\n")
            f.write(f"Q: {prompt}\n")
            f.write(f"A: {response}\n")
            f.write(f"✅ Predicted: {predicted}, Expected: {expected}, Correct: {is_correct}\n\n")

    accuracy = correct / len(expected_answers) * 100
    print(f"\n🎯 Accuracy: {correct}/{len(expected_answers)} = {accuracy:.1f}%")
    print(f"📄 Saved results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought Reasoning Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Model name from HuggingFace Hub")
    parser.add_argument("--input", type=str, default="data/prompts.txt", help="Path to input prompts")
    parser.add_argument("--output", type=str, default="results/output.txt", help="Path to save results")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (set 0 for greedy)")
    args = parser.parse_args()

    # Эталонные правильные ответы
    expected_answers = [5, 4, 7, 7, 3, 5, 10, 5, 3, 5]

    prompts = load_prompts(args.input)
    responses = generate_responses(prompts, args.model, args.max_tokens, args.temperature)
    save_and_evaluate(prompts, responses, expected_answers, args.output)


if __name__ == "__main__":
    main()
