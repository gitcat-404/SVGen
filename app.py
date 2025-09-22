import gradio as gr
import torch
import os
from PIL import Image
import cairosvg
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from threading import Thread

# Please modify this path to the folder where your model is stored locally
MODEL_PATH = "Models/SVGen-Qwen2.5-Coder-7B-Instruct" 

# Define the required system prompt as a global constant for easy management
SYSTEM_PROMPT = """Please generate the reasoning process and final SVG code separately, according to the following format:<think>
...
</think>
```svg
...
```"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

def load_model():
    """
    Loads the tokenizer and model from the specified path.
    This function will only be executed once, either on the first inference or at startup.
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print(f"Loading model and tokenizer from '{MODEL_PATH}'...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,  # If your GPU does not support bfloat16, you can change this to torch.float16
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model and tokenizer loaded successfully!")

def generate_text_to_svg(
    text_description: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int
):
    """
    Streams the generation of an SVG. Displays the thinking process in real-time
    and parses the SVG upon completion.
    """
    if not text_description or text_description.strip() == "":
        yield "", "Error: Please enter a text description.", None
        return

    load_model()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please generate an SVG icon that meets the following description: {text_description}."}
    ]
    
    # 2. Use the tokenizer's `apply_chat_template` method to generate the final prompt
    #    This method automatically adds special tokens (like <|im_start|>) as required by the model, ensuring the format is correct
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # We need the string, not the token IDs
        add_generation_prompt=True # Adds a prompt at the end to let the model know it's its turn to generate a response
    )


    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_kwargs = dict(
        **inputs,
        generation_config=generation_config,
        streamer=streamer
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    think_content = ""
    full_output = ""
    think_stopped = False
    final_think_content = ""
    
    print("Starting stream generation...")
    for new_text in streamer:
        full_output += new_text
        
        if not think_stopped:
            think_content += new_text
            yield think_content, "Generating SVG code...", None
            
            if "</think>" in think_content:
                think_stopped = True
                think_content_clean = think_content.split("</think>")[0]
                final_think_content = think_content_clean
                yield final_think_content, "Thinking complete, continuing to generate SVG...", None

    thread.join()
    print("Stream generation finished.")
    print("\n" + "="*30 + " Full Model Output START " + "="*30)
    print(full_output)
    print("="*30 + " Full Model Output END " + "="*30 + "\n")

    if '<svg' in full_output and '</svg>' in full_output:
        start_index = full_output.find('<svg')
        end_index = full_output.rfind('</svg>') + len('</svg>')
        svg_code = full_output[start_index:end_index].strip()
        
        try:
            print("Rendering final preview at 1024x1024 resolution...")
            png_data = cairosvg.svg2png(
                bytestring=svg_code.encode('utf-8'),
                output_width=1024,
                output_height=1024
            )
            png_image = Image.open(io.BytesIO(png_data))
            yield final_think_content, svg_code, png_image
        except Exception as e:
            error_message = f"SVG rendering failed: {e}"
            print(error_message)
            yield final_think_content, svg_code, None
            
    else:
        error_message = "Error: SVG code not found in the final output."
        print(error_message)
        yield final_think_content, error_message, None

def create_interface():
    """Creates and returns the Gradio interface."""
    
    example_texts = [
        "An orange building with pillars, topped by a triangular roof and a small orange circle.",
        "A golden harp with curved arms and vertical strings, resting on a yellow base.",
        "A gray traffic light with red, yellow, and green circular lights.",
        "A yellow sad face with maroon eyes, pink cheeks, and a curved frown.",
        "A colorful rainbow with red, yellow, green, and blue arcs, flanked by light blue clouds.",
        "Two overlapping yellow folders with a red circular warning symbol.",
        "Red folder with blue and white elements, featuring a blue Visual Studio logo.",
        "A light blue cloud with three rain lines below, symbolizing light rain.",
        "A light pink circular background with a white clipboard icon.",
        "A blue gear with a white center and a smaller blue circle inside.",
        "Blue background, white chart outline, cyan base, and bars in white.",
    ]
    
    with gr.Blocks(title="SVGen Demo", theme=gr.themes.Soft(primary_hue=gr.themes.colors.orange,secondary_hue="red",font=[
            "sans-serif"])) as demo:
        gr.Markdown("# SVGen demo")
        gr.Markdown("Enter a text description below, and the model will generate the corresponding SVG vector graphics code and a preview for you.")
        
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="SVG Description",
                    placeholder="e.g., a red circle with a blue square inside",
                    lines=4
                )
                
                with gr.Accordion("Advanced Parameters", open=False):
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Top P")
                    max_new_tokens = gr.Slider(minimum=64, maximum=10000, value=4096, step=64, label="Max New Tokens")
                
                generate_btn = gr.Button("Generate SVG", variant="primary")
                
                gr.Examples(
                    examples=[[text] for text in example_texts],
                    inputs=[text_input],
                    label="Example Descriptions (Click to use)",
                    examples_per_page=7
                )
                
                with gr.Accordion("Model Thought Process", open=True):
                    think_output = gr.Textbox(
                        label="Thought Process", 
                        lines=5, 
                        interactive=False,
                        show_label=False
                    )

            # --- Right Panel (Final Output Area) ---
            with gr.Column(scale=1):
                svg_output = gr.Textbox(
                    label="Generated SVG Code",
                    lines=10,
                    max_lines=20,
                    show_copy_button=True
                )

                png_preview = gr.Image(
                    label="SVG Preview", 
                    type="pil", 
                    height=420, 
                    interactive=False
                )

        generate_btn.click(
            fn=generate_text_to_svg,
            inputs=[text_input, temperature, top_p, max_new_tokens],
            outputs=[think_output, svg_output, png_preview],
            queue=True
        )
        
    return demo

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if not os.path.exists(MODEL_PATH):
        print("="*50)
        print("Warning: Model path is not set correctly!")
        print(f"Please modify the 'MODEL_PATH' variable in the script to your local model folder path.")
        print("Current path: ", MODEL_PATH)
        print("="*50)
    else:
        load_model()
    
    app = create_interface()
    app.launch()