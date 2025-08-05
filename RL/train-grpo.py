import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import re
import math

def load_and_format_data(json_file_path):
    """从JSON文件加载数据并格式化为system内容"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    formatted_data = []

    for item in data:
        messages = []

        # 使用固定的system提示词
        messages.append({
            "role": "system",
            "content": "Please generate the reasoning process and final SVG code separately, according to the following format:<think>\n...\n</think>\n``svg\n...\n```"
        })

        # 添加用户消息
        for msg in item["conversations"]:
            if msg["from"] == "human":
                messages.append({
                    "role": "user",
                    "content": msg["value"].strip()
                })

        # 验证回复格式
        gpt_response = next((m["value"] for m in item["conversations"] if m["from"] == "gpt"), "")
        if not (gpt_response.startswith("svg") and gpt_response.endswith("```")):
            print(f"警告: 回复格式不正确: {gpt_response[:200]}...")

        formatted_data.append({
            "prompt": messages,
            "answers": gpt_response
        })
        print(formatted_data[0])
    return Dataset.from_list(formatted_data)

def svg_format_reward(completions, **kwargs):
    """简化版奖励函数：仅检查是否以```结尾"""
    rewards = []
    print("\n" + "="*50 + " 奖励函数评估开始 " + "="*50)

    for i, completion in enumerate(completions):
        content = completion[0]["content"].strip()

        # 打印完整输出内容（前200字符避免过长）
        print(f"\n==== 第 {i+1} 个生成结果 ====")
        print("内容预览：" + (content[:200] + ("..." if len(content) > 200 else "")))

        # 核心检查：是否以```结尾
        ends_correctly = content.endswith("```")

        # 设置奖励/惩罚
        score = 1.0 if ends_correctly else -10.0

        # 打印结果
        print(f"\n结尾检查：{'√' if ends_correctly else 'X'}")
        print(f"得分：{score:.1f}")
        rewards.append(score)

    # 打印本轮统计
    total_reward = sum(rewards)
    print("\n" + "="*40)
    print(f"本轮总奖励：{total_reward:.2f}（平均：{total_reward/len(rewards):.2f}）")
    print("="*50 + " 奖励函数评估结束 " + "="*50 + "\n")

    return rewards

def path_count_reward(completions, answers, beta=1.0, gamma=0.5, **kwargs):
    rewards = []
    print("\n" + "="*60 + " 路径数量评估 " + "="*60)
    for idx, (completion, answer) in enumerate(zip(completions, answers)):
        # 提取生成内容和参考答案
        gen_content = completion[0]["content"]
        gt_content = answer if isinstance(answer, str) else answer[0]["content"]

        # 统计路径数量
        def count_paths(svg_code):
            return len(re.findall(r"<path[^>]*>", svg_code, re.IGNORECASE))

        try:
            # 从生成内容提取SVG代码
            gen_svg = gen_content.split("``svg")[-1].split("```")[0]
            n_gen = count_paths(gen_svg)
            # 从参考答案提取SVG代码
            gt_svg = gt_content.split("``svg")[-1].split("```")[0]
            n_gt = count_paths(gt_svg)

            # 计算奖励（新规则）
            if n_gen >= n_gt:
                reward = 1.0
                calc_info = f">参考路径数，直接奖励1.0"
            else:
                diff = n_gt - n_gen
                reward = beta * math.exp(-gamma * diff)
                calc_info = f"{beta}*exp(-{gamma}*{diff}) = {reward:.3f}"
            # 打印详细信息
            print(f"\n==== 样本 {idx+1} ====")
            print(f"[生成SVG] 路径数: {n_gen} | [参考答案] 路径数: {n_gt}")
            print(f"奖励计算: {calc_info}")
        except Exception as e:
            print(f"解析失败: {str(e)}")
            reward = 0.0
        rewards.append(reward)

    # 打印统计信息
    total_reward = sum(rewards)
    avg_reward = total_reward / len(rewards) if rewards else 0
    print("\n" + "="*40)
    print(f"总路径匹配奖励: {total_reward:.3f}（平均：{avg_reward:.3f}）")
    print("="*60 + " 评估结束 " + "="*60 + "\n")
    return rewards

if __name__ == '__main__':
    # 初始化模型和tokenizer
    model_name = "/gemini/space/thu/zhaozhiyuan/wfy/checkpoint-4108"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    try:
        dataset = load_and_format_data("/gemini/space/thu/zhaozhiyuan/wfy/json_data/colored_1_test.json")
        print(f"数据加载成功，示例system内容: {dataset[0]['prompt'][0]['content'][:100]}...")
    except Exception as e:
        raise ValueError(f"数据加载失败: {str(e)}")

    # 训练配置
    training_args = GRPOConfig(
        output_dir="svg_model_output-4",
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

    # 初始化训练器
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[svg_format_reward, path_count_reward],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("final_svg_model")
    tokenizer.save_pretrained("final_svg_model")
    print("训练完成，模型已保存！")