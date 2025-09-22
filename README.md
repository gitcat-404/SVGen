# <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6830606cef839b9401da6de3/WUZR5uvh22Y0CMZ-K4OOC.png" alt="Logo" width="60" height="60" style="vertical-align: middle;"/> SVGen: Interpretable Vector Graphics Generation with Large Language Models

# 🎉 Accepted by ACM MM 2025
<div align="center" style="line-height: 1.2;">


[![arXiv](https://img.shields.io/badge/arXiv-2508.09168-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2508.09168)
[![Dataset SVG-MetaData](https://img.shields.io/badge/Dataset-SVG--MetaData-informational?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/gitcat-404/SVG-MetaData)
[![Dataset SVG-1M-Json](https://img.shields.io/badge/Dataset-SVG--1M--Json-informational?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/gitcat-404/SVG-1M-Json)
[![Model SVGen-Qwen2.5-3B](https://img.shields.io/badge/Model-SVGen--Qwen2.5--3B-FFA500?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/gitcat-404/SVGen-Qwen2.5-3B-Instruct)
[![Model SVGen-Llama-3.2-3B](https://img.shields.io/badge/Model-SVGen--Llama--3.2--3B-FFA500?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/gitcat-404/SVGen-Llama-3.2-3B-Instruct)
[![Model SVGen-StarCoder2-3B](https://img.shields.io/badge/Model-SVGen--StarCoder2--3B-FFA500?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/gitcat-404/SVGen-StarCoder2-3B)
[![Model SVGen-Qwen2.5-Coder-7B](https://img.shields.io/badge/Model-SVGen--Qwen2.5--Coder--7B-FFA500?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/gitcat-404/SVGen-Qwen2.5-Coder-7B-Instruct)
</div>

## 1. Introduction
SVGen is an end-to-end model that generates high-quality SVG code from text. We fine-tuned a Large Language Model on our custom SVG-1M dataset using curriculum learning, Chain-of-Thought (CoT), and reinforcement learning.
## 2. Dependencies
This repo is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Sincere thanks to their excellent work!
### 2.1 Clone the Repository
```bash
git clone https://github.com/gitcat-404/SVGen.git
cd SVGen
```
### 2.2 Create Conda Environment
```bash
conda create -n svgen python=3.10 -y
conda activate svgen
```
### 2.3 Dependencies for cairosvg
```bash
sudo apt update
sudo apt install libcairo2 libcairo2-dev
```
### 2.4 Python Dependencies
```bash
pip install torch==2.5.1+cu124 torchvision==0.20.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
```
## 3. How to use
### 3.1 Download Model Weights
For this demonstration, we will be using our top-performing model, [SVGen-Qwen2.5-Coder-7B-Instruct](https://huggingface.co/gitcat-404/SVGen-Qwen2.5-Coder-7B-Instruct). Please download the model weights from Hugging Face and store them under the Models/ path.
```bash
pip install huggingface_hub
hf download gitcat-404/SVGen-Qwen2.5-Coder-7B-Instruct --local-dir Models/SVGen-Qwen2.5-Coder-7B-Instruct
```
### 3.2 Interactive demo
```bash
python app.py
```
### 3.3 Inference
Taking the test data for this task as an example, we will write the prompts that need to be inferred into a CSV file, such as the example provided in `test/color_test.csv`.
```bash
python inference.py \
    --model_path Models/SVGen-Qwen2.5-Coder-7B-Instruct \
    --csv_file_path test/color_test.csv \
    --prompt_type qwen \
    --output_folder "results/qwen_outputs"
```
## 4. Test
Our evaluation metrics in this article are: Fréchet Inception Distance (FID), CLIPScore-T2I, CLIPScore-I2I, Preference Scores (HPS), and Aesthetic Score. The specific implementation is available in `test/` To test the model, first download `sac+logos+ava1-l14-linearMSE.pth` from [Huggingface](https://huggingface.co/haor/aesthetics/tree/main) and `hpc.pt` from [GitHub](https://github.com/tgxs002/align_sd?tab=readme-ov-file), placing them in the test/pretrain_weight/ folder. Next, modify `test/test.py` by filling in the folder containing the previously generated images, then run:"
```bash
python test/test.py
```
## 5.Train
All experiments are executed on 8×NVIDIA A800 GPUs using the AdamW optimizer (learning rate 4e-5), with a maximum sequence length of 8,000 tokens.In order to reproduce our experiments, please first download all data from 🤗[SVG-1M-Json](https://huggingface.co/datasets/gitcat-404/SVG-1M-Json) and place it in the `json_data/` folder. We have provided the training scripts in the `config/` folder. After modifying the configurations, you can execute them sequentially:
```bash
sh config/train_stage1.sh
sh config/train_stage2.sh
sh config/train_stage3.sh
sh config/train_stage_RL.sh
```
## 6.Get SVG metadata from website
This dataset was collected by web scraping public content from [IconFont](https://www.iconfont.cn/) and is intended for non-commercial academic research and technical exchange purposes only.
```bash
python Spider/CrawlingSVG_Icon.py
```
## 💕 Acknowledgments:
We would like to extend our sincerest thanks to the projects and websites that inspired this work, specifically:
[IconFont](https://www.iconfont.cn/)
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
[LLM4SVG](https://github.com/ximinng/LLM4SVG)
[OmniSVG](https://github.com/OmniSVG/OmniSVG)
[Star-vector](https://github.com/joanrod/star-vector)
## Citation
```
@article{wang2025svgen,
  title={SVGen: Interpretable Vector Graphics Generation with Large Language Models},
  author={Wang, Feiyu and Zhao, Zhiyuan and Liu, Yuandong and Zhang, Da and Gao, Junyu and Sun, Hao and Li, Xuelong},
  journal={arXiv preprint arXiv:2508.09168},
  year={2025}
}
```
