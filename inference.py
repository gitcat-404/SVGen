import time
import os
import pandas as pd
import re
import argparse
from vllm import LLM, SamplingParams

def main(args):
    """
    Main function to generate SVG files based on descriptions using a VLLM model.
    """
    output_folder = args.output_folder
    model_path = args.model_path
    csv_file_path = args.csv_file_path
    prompt_type = args.prompt_type

    print(f"Model Path: {model_path}")
    print(f"CSV File Path: {csv_file_path}")
    print(f"Output Folder: {output_folder}")
    print(f"Prompt Type: {prompt_type}")

    os.makedirs(output_folder, exist_ok=True)
    try:
        df = pd.read_csv(csv_file_path)
        descriptions = df['desc'].tolist()
        ids = df['id'].tolist()
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except KeyError as e:
        print(f"Error: The CSV file must contain '{e.args[0]}' column.")
        return

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.2, top_p=0.7, max_tokens=8000)
    # Create an LLM instance.
    llm = LLM(model=model_path,
              tensor_parallel_size=2, max_model_len=4096,
              max_num_seqs=50, trust_remote_code=True,
              tokenizer=model_path, tokenizer_mode='auto')

    start_time = time.time()

    for i, desc in enumerate(descriptions):
        # Dynamically select the prompt based on the prompt_type argument
        if prompt_type == 'qwen':
            prompt = f"<|im_start|>system\nPlease generate the reasoning process and final SVG code separately, according to the following format:<think>\n...\n</think>\n```svg\n...\n```<|im_end|>\n<|im_start|>user\nPlease generate an SVG icon that meets the following description: {desc}<|im_end|>\n<|im_start|>assistant\n"
        elif prompt_type == 'starcoder':
            prompt = f"Please generate the reasoning process and final SVG code separately, according to the following format:<think>\n...\n</think>\n```svg\n...\n```\n\nHuman:Please generate an SVG icon that meets the following description: {desc}\nAssistant:"
        elif prompt_type == 'llama':
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"Please generate the reasoning process and final SVG code separately, according to the following format:<think>\n...\n</think>\n```svg\n...\n```\n<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Please generate an SVG icon that meets the following description: {desc}<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
            
        output = llm.generate(prompt, sampling_params)
        print(f"Generated output for ID {ids[i]}: {output}")
        
        svg_code = output[0].outputs[0].text
        # Use regex to extract content within ```svg ... ```
        match = re.search(r'```svg\n(.*?)\n```', svg_code, re.DOTALL)
        
        if match:
            svg_content = match.group(1)
        else:
            print(f"SVG content not found for id: {ids[i]}")
            continue

        file_name = f"{ids[i]}.svg"
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(svg_content)
        print(f"Successfully saved {file_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SVG files from text descriptions using a VLLM model.")
    
    parser.add_argument(
        '--output_folder', 
        type=str, 
        default="generated_svgs", 
        help='The folder where generated SVG files will be saved.'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help='Path to the VLLM model directory.'
    )
    parser.add_argument(
        '--csv_file_path', 
        type=str, 
        required=True, 
        help='Path to the CSV file containing "id" and "desc" columns.'
    )
    parser.add_argument(
        '--prompt_type', 
        type=str, 
        required=True, 
        choices=['qwen', 'starcoder', 'llama'],
        help='The type of prompt to use for the model.'
    )

    args = parser.parse_args()
    main(args)