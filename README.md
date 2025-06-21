# 🧠 Reasoning with Language Models (Chain of Thought)

This project explores how well different open-source language models perform on simple arithmetic and logic tasks when guided with *Chain-of-Thought* (CoT) prompting.

## 📌 Goal

To compare reasoning performance of small vs large LLMs using a set of basic math and logic questions.

---

## 💡 Models Compared

| Model                          | Accuracy (10 tasks) |
| ------------------------------ | ------------------- |
| `google/flan-t5-base`          | ✅ 1 / 10           |
| `declare-lab/flan-alpaca-base` | ✅ 1 / 10           |
| `google/flan-t5-large`         | ✅ 8 / 10           |

*All tasks use CoT-style prompting with **`Let's think step by step`**.*

---

## 📁 Project Structure

```
reasoning-llm-cot/
├── data/
│   └── prompts.txt             # Task prompts (1 per line)
├── results/
│   ├── base.txt                # Output from flan-t5-base
│   ├── large.txt               # Output from flan-t5-large
│   └── alpaca.txt              # Output from flan-alpaca
├── main.py                     # Main evaluation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ▶️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run reasoning and evaluation:

```bash
python main.py \
  --model google/flan-t5-large \
  --input data/prompts.txt \
  --output results/large.txt \
  --max_tokens 50 \
  --temperature 0.0
```

### 🔁 Supported models:

You can try any of these by changing the `--model` parameter:

```bash
--model google/flan-t5-small
--model google/flan-t5-base
--model google/flan-t5-large
--model declare-lab/flan-alpaca-base
```

> ❗ Larger models (like `flan-t5-xl`, `flan-t5-xxl`) require powerful GPUs and are **not recommended** for local CPU execution.

---

## 🧪 Dataset

The dataset includes 10 simple reasoning problems:

```
There are 3 apples on the table. You put 2 more. How many apples are on the table now?
You have 6 pencils. You give 2 to a friend. How many pencils do you have now?
There are 10 ducks in a pond. 3 swim away. How many ducks are left?
...
```

Each problem has a known correct answer. The model output is parsed to extract the final number and compared to the expected result.

---

## 📊 Sample Output (flan-t5-large)

```
Q: There are 3 apples on the table. You put 2 more. How many apples are on the table now?
A: Let's solve step by step. 3 + 2 = 5. The answer is 5.
✅ Predicted: 5, Expected: 5, Correct: True
```

---

## 🛠️ Dependencies

Minimal setup for Hugging Face Transformers + PyTorch:

```text
transformers
torch
accelerate
```

(See `requirements.txt` for full list.)

---

## 🖚 Conclusion

- Larger models like `flan-t5-large` show much stronger reasoning capabilities, even on very basic tasks.
- Chain-of-Thought prompting improves reliability.
- Great showcase for evaluating reasoning abilities in small vs large open models.

---
 
**License**: MIT
