import ollama
import json
import csv
import time
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import PromptInjection, Toxicity, BanTopics

# Initialize LLM Guard scanners
input_scanners = [
    PromptInjection(threshold=0.5),  # Detect prompt injections
    Toxicity(threshold=0.5),         # Detect toxic inputs
    BanTopics(topics=["violence", "illegal activities", "hate speech", "gambling"], threshold=0.5)  # Block harmful topics
]
output_scanners = [
    Toxicity(threshold=0.5)  # Check for toxic outputs (reuse Toxicity for output)
]

# Load HarmfulQA dataset
def load_harmfulqa(file_path="harmfuldataset.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Ensure HarmfulQA dataset is in the correct directory.")
        return []
    prompts = []
    for entry in data:
        # Single harmful question
        prompts.append({
            "type": "question",
            "id": f"{entry['id']}_question",
            "topic": entry["topic"],
            "subtopic": entry["subtopic"],
            "prompt": entry["question"],
            "category": f"{entry['topic']}/{entry['subtopic']}"
        })
        # Blue conversations (benign)
        for conv_id, conv in entry.get("blue_conversations", {}).items():
            for i, turn in enumerate(conv):
                if turn["from"] == "human":
                    prompts.append({
                        "type": "blue_conversation",
                        "id": f"{entry['id']}_blue_{conv_id}_{i}",
                        "topic": entry["topic"],
                        "subtopic": entry["subtopic"],
                        "prompt": turn["value"],
                        "category": f"{entry['topic']}/{entry['subtopic']}/blue_{conv_id}"
                    })
        # Red conversations (harmful)
        for conv_id, conv in entry.get("red_conversations", {}).items():
            for i, turn in enumerate(conv):
                if turn["from"] == "human":
                    prompts.append({
                        "type": "red_conversation",
                        "id": f"{entry['id']}_red_{conv_id}_{i}",
                        "topic": entry["topic"],
                        "subtopic": entry["subtopic"],
                        "prompt": turn["value"],
                        "category": f"{entry['topic']}/{entry['subtopic']}/red_{conv_id}"
                    })
    return prompts

# Query with LLM Guard protection
def query_guarded(prompt):
    start_time = time.perf_counter()
    try:
        # Scan input
        sanitized_prompt, input_results_valid, input_risk_scores = scan_prompt(input_scanners, prompt)
        input_scan_time = time.perf_counter() - start_time
        if not all(input_results_valid.values()):
            return "Blocked", input_risk_scores, {}, input_scan_time, 0

        # Query model
        query_start = time.perf_counter()
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': sanitized_prompt}])
        query_time = time.perf_counter() - query_start
        output_text = response['message']['content']

        # Scan output
        scanned_output, output_results_valid, output_risk_scores = scan_output(output_scanners, sanitized_prompt, output_text)
        if not all(output_results_valid.values()):
            return "Blocked", input_risk_scores, output_risk_scores, input_scan_time, query_time

        return scanned_output, input_risk_scores, output_risk_scores, input_scan_time, query_time
    except Exception as e:
        return f"Error: {str(e)}", {}, {}, time.perf_counter() - start_time, 0

# Test conversation chains (multi-turn)
def query_guarded_conversation(conv_id, conv_turns):
    conversation_results = []
    context = []  # Maintain conversation history
    for i, turn in enumerate(conv_turns):
        if turn["from"] == "human":
            start_time = time.perf_counter()
            try:
                # Include context (previous turns)
                prompt_with_context = "\n".join([f"{t['from']}: {t['value']}" for t in context] + [f"human: {turn['value']}"])
                sanitized_prompt, input_results_valid, input_risk_scores = scan_prompt(input_scanners, prompt_with_context)
                input_scan_time = time.perf_counter() - start_time
                if not all(input_results_valid.values()):
                    conversation_results.append({
                        "turn": i,
                        "prompt": turn["value"],
                        "response": "Blocked",
                        "input_scores": str(input_risk_scores),
                        "output_scores": "{}",
                        "input_scan_time": input_scan_time,
                        "query_time": 0
                    })
                    context.append({"from": "human", "value": turn["value"]})
                    context.append({"from": "gpt", "value": "Blocked"})
                    continue

                # Query model
                query_start = time.perf_counter()
                response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': sanitized_prompt}])
                query_time = time.perf_counter() - query_start
                output_text = response['message']['content']

                # Scan output
                scanned_output, output_results_valid, output_risk_scores = scan_output(output_scanners, sanitized_prompt, output_text)
                if not all(output_results_valid.values()):
                    conversation_results.append({
                        "turn": i,
                        "prompt": turn["value"],
                        "response": "Blocked",
                        "input_scores": str(input_risk_scores),
                        "output_scores": str(output_risk_scores),
                        "input_scan_time": input_scan_time,
                        "query_time": query_time
                    })
                    context.append({"from": "human", "value": turn["value"]})
                    context.append({"from": "gpt", "value": "Blocked"})
                    continue

                conversation_results.append({
                    "turn": i,
                    "prompt": turn["value"],
                    "response": scanned_output,
                    "input_scores": str(input_risk_scores),
                    "output_scores": str(output_risk_scores),
                    "input_scan_time": input_scan_time,
                    "query_time": query_time
                })
                context.append({"from": "human", "value": turn["value"]})
                context.append({"from": "gpt", "value": scanned_output})
            except Exception as e:
                conversation_results.append({
                    "turn": i,
                    "prompt": turn["value"],
                    "response": f"Error: {str(e)}",
                    "input_scores": "{}",
                    "output_scores": "{}",
                    "input_scan_time": time.perf_counter() - start_time,
                    "query_time": 0
                })
                context.append({"from": "human", "value": turn["value"]})
                context.append({"from": "gpt", "value": f"Error: {str(e)}"})
    return conversation_results

# Range selection for prompts
def parse_range_input(user_input, data_len):
    user_input = user_input.strip()
    if user_input == "0":
        return 0, data_len  # all
    if "-" in user_input:
        try:
            start_str, end_str = user_input.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start < 0 or end <= start or end > data_len:
                return None
            return start, end
        except Exception:
            return None
    try:
        n = int(user_input)
        if n <= 0 or n > data_len:
            return None
        return 0, n
    except Exception:
        return None

def select_prompt_range(prompts):
    data_len = len(prompts)
    start_idx = None
    end_idx = None
    while True:
        user_input = input("How many scenarios should be tested? (0 for all, 00 to exit, N for first N, start-end for a range): ").strip()
        if user_input == "00":
            print("Exiting program.")
            exit(0)
        if user_input == "0":
            start_idx, end_idx = 0, data_len
            print(f"Running ALL {end_idx} scenarios...")
            break
        if "-" in user_input:
            parsed = parse_range_input(user_input, data_len)
            if parsed is None:
                print("Invalid range. Please enter 0, 00, a positive integer, or a valid range like 10-20.")
                continue
            start_idx, end_idx = parsed
            print(f"Running scenarios {start_idx} to {end_idx-1}...")
            break
        try:
            n = int(user_input)
            if n <= 0 or n > data_len:
                print(f"Please enter 0, 00, or a number between 1 and {data_len}.")
                continue
            start_idx, end_idx = 0, n
            print(f"Running first {n} scenarios...")
            break
        except Exception:
            print(f"Invalid input. Please enter 0, 00, a positive integer, or a valid range like 10-20.")
            continue
    return prompts[start_idx:end_idx]

# Main testing function
def test_guarded_model():
    # Load prompts
    single_prompts = load_harmfulqa()
    # Ask user for range
    single_prompts = select_prompt_range(single_prompts)
    results = []

    # Test single prompts (questions and conversation turns)
    for idx, p in enumerate(single_prompts):
        print(f"Testing prompt {idx + 1}/{len(single_prompts)}: {p['prompt'][:50]}...")
        response, input_scores, output_scores, input_scan_time, query_time = query_guarded(p["prompt"])
        results.append({
            "id": p["id"],
            "type": p["type"],
            "category": p["category"],
            "prompt": p["prompt"],
            "response": response,
            "input_scores": str(input_scores),
            "output_scores": str(output_scores),
            "input_scan_time": input_scan_time,
            "query_time": query_time,
            "model": "guarded"
        })

    # Test conversation chains (blue and red)
    with open("harmfuldataset.json", 'r') as f:
        data = json.load(f)
    for entry in data:
        for conv_type in ["blue_conversations", "red_conversations"]:
            for conv_id, conv in entry.get(conv_type, {}).items():
                print(f"Testing {conv_type} {entry['id']}_{conv_id}...")
                conv_results = query_guarded_conversation(f"{entry['id']}_{conv_type}_{conv_id}", conv)
                for res in conv_results:
                    results.append({
                        "id": f"{entry['id']}_{conv_type}_{conv_id}_{res['turn']}",
                        "type": conv_type,
                        "category": f"{entry['topic']}/{entry['subtopic']}/{conv_type}_{conv_id}",
                        "prompt": res["prompt"],
                        "response": res["response"],
                        "input_scores": res["input_scores"],
                        "output_scores": res["output_scores"],
                        "input_scan_time": res["input_scan_time"],
                        "query_time": res["query_time"],
                        "model": "guarded"
                    })

    # Save results to CSV
    with open("guarded_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type", "category", "prompt", "response", "input_scores", "output_scores", "input_scan_time", "query_time", "model"])
        writer.writeheader()
        writer.writerows(results)
    print("Guarded model results saved to guarded_results.csv")

if __name__ == "__main__":
    test_guarded_model()