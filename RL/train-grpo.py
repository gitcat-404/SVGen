import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_pe_model, TaskType
import re
import math
import argparse # Import the argparse library

def load_and_format_data(json_file_path):
    """Load data from a JSON file and format it for the trainer."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    formatted_data = []

    for item in data:
        messages = []
        messages.append({
            "role": "system",
            "content": "Please generate the reasoning process and final SVG code separately, according to the following format:<think>\n...\n</think>\n``svg\n...\n```"
        })
        for msg in item["conversations"]:
            if msg["from"] == "human":
                messages.append({
                    "role": "user",
                    "content": msg["value"].strip()
                })
        gpt_response = next((m["value"] for m in item["conversations"] if m["from"] == "gpt"), "")
        if not (gpt_response.startswith("``svg") and gpt_response.endswith("```")):
            # Corrected the print warning logic to ensure it only prints when necessary
            # print(f"Warning: Incorrect response format: {gpt_response[:200]}...")
            pass

        formatted_data.append({
            "prompt": messages,
            "answers": gpt_response
        })
    return Dataset.from_list(formatted_data)

def svg_format_reward(completions, **kwargs):
    """Simplified reward function: only checks if the completion ends with ```"""
    rewards = []
    # print("\n" + "="*50 + " Reward function evaluation start " + "="*50)

    for i, completion in enumerate(completions):
        content = completion[0]["content"].strip()
        # print(f"\n==== Generation result #{i+1} ====")
        # print("Content preview: " + (content[:200] + ("..." if len(content) > 200 else "")))
        ends_correctly = content.endswith("```")
        score = 1.0 if ends_correctly else -10.0
        # print(f"\nEnding check: {'√' if ends_correctly else 'X'}")
        # print(f"Score: {score:.1f}")
        rewards.append(score)

    # total_reward = sum(rewards)
    # print("\n" + "="*40)
    # print(f"Total reward for this batch: {total_reward:.2f} (Average: {total_reward/len(rewards):.2f})")
    # print("="*50 + " Reward function evaluation end " + "="*50 + "\n")
    return rewards

def path_count_reward(completions, answers, beta=1.0, gamma=0.5, **kwargs):
    """Reward function to evaluate the number of SVG paths."""
    rewards = []
    # print("\n" + "="*60 + " Path count evaluation " + "="*60)
    for idx, (completion, answer) in enumerate(zip(completions, answers)):
        gen_content = completion[0]["content"]
        gt_content = answer if isinstance(answer, str) else answer[0]["content"]

        def count_paths(svg_code):
            return len(re.findall(r"<path[^>]*>", svg_code, re.IGNORECASE))

        try:
            gen_svg = gen_content.split("``svg")[-1].split("```")[0]
            n_gen = count_paths(gen_svg)
            gt_svg = gt_content.split("``svg")[-1].split("```")[0]
            n_gt = count_paths(gt_svg)

            if n_gen >= n_gt:
                reward = 1.0
            else:
                diff = n_gt - n_gen
                reward = beta * math.exp(-gamma * diff)
            # print(f"\n==== Sample #{idx+1} ====")
            # print(f"[Generated SVG] Path count: {n_gen} | [Reference] Path count: {n_gt}")
        except Exception as e:
            # print(f"Parsing failed: {str(e)}")
            reward = 0.0
        rewards.append(reward)

    # total_reward = sum(rewards)
    # avg_reward = total_reward / len(rewards) if rewards else 0
    # print("\n" + "="*40)
    # print(f"Total path match reward: {total_reward:.3f} (Average: {avg_reward:.3f})")
    # print("="*60 + " Evaluation end " + "="*60 + "\n")
    return rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a model for SVG generation using GRPO.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the initial model checkpoint to be fine-tuned."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the training data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store logs and checkpoints during training."
    )
    parser.add_argument(
        "--final_model_path",
        type=str,
        default="final_svg_model",
        help="Directory to save the final trained model and tokenizer (default: final_svg_model)."
    )
    args = parser.parse_args()
    
    # Initialize model and tokenizer
    print(f"Loading model from '{args.model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    print("Model loaded.")

    try:
        print(f"Loading data from '{args.dataset_path}'...")
        dataset = load_and_format_data(args.dataset_path)
        print(f"Data loaded successfully with {len(dataset)} records.")
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")

    # Training configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir, # Use argument
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=6,
        max_prompt_length=512,
        max_completion_length=4000,
        num_train_epochs=1,
        save_steps=50,
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard"
    )

    # Initialize the trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer, # In newer versions of TRL, the tokenizer should be passed via this argument
        reward_funcs=[svg_format_reward, path_count_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Training finished. Saving final model to '{args.final_model_path}'...")
    trainer.save_model(args.final_model_path) # Use argument
    tokenizer.save_pretrained(args.final_model_path) # Use argument
    print("Model saved successfully!")
