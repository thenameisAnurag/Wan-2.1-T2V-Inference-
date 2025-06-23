
# Wan2.1 Text-to-Video Generation with Fashion Context

![image](https://github.com/user-attachments/assets/d594fd82-fe67-48ad-bf0c-2f697d50a7b2)


<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.1">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a> &nbsp&nbsp | &nbsp&nbsp 🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Technical Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat Group</a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>
</p>

---

## Overview

**Wan2.1** is a state-of-the-art model designed to generate high-quality videos from text prompts, specifically tailored for fashion video generation. With support for 480P and 720P resolutions, it leverages a powerful architecture built using the Flow Matching framework, integrated with Diffusion Transformers. The model uses a T5 Encoder to encode multilingual text input, enhanced by cross-attention in each transformer block to embed text effectively within the model's structure.

This README provides an overview of how to set up the model, use it for video generation, and how you can customize the pipeline for different use cases like fashion video creation.

---

## Model Features

* **Text-to-Video Generation**: Given a detailed text prompt, the model generates videos that align with the input description.
* **Multiple Resolutions Supported**: Supports 480P and 720P by default.
* **Faster Inference**: Optimized for both memory and speed, with CPU offloading for better resource management.
* **Attention Slicing**: Uses attention slicing for memory efficiency during inference.
* **Versatile for Fashion**: Tailored for generating fashion-related content, creating a perfect fit for the fashion industry.

---

## Installation

### 1. Clone the repository

```sh
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

> **Note**: Make sure to have **torch >= 2.4.0** installed in your environment.

---

## Model Download

You can download the models from either **Hugging Face** or **ModelScope**.

| Models  | Download Link                                                          | Notes                       |
| ------- | ---------------------------------------------------------------------- | --------------------------- |
| T2V-14B | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Supports both 480P and 720P |

---

## Architecture
![image](https://github.com/user-attachments/assets/0bc19f72-e99d-4270-b8f8-5e483207d3c9)

**Wan2.1** uses the **Flow Matching** framework in the paradigm of mainstream **Diffusion Transformers**.

* **Text Encoding**: Utilizes a **T5 Encoder** to encode multilingual text input, embedding it effectively in the model.
* **Cross-Attention**: Each transformer block applies cross-attention to integrate the encoded text.
* **MLP Layer**: An MLP with a **Linear layer** and **SiLU layer** processes time embeddings and predicts six modulation parameters.
* **Shared MLP**: The MLP is shared across transformer blocks, with each block learning unique biases, improving performance at the same parameter scale.



## Code Usage

The code is designed to allow you to generate videos using the **Wan2.1** model, leveraging **CUDA** for accelerated performance. You can modify the **height**, **width**, and **prompt** based on your requirements.


## Usage Flow

1. **Input Prompt**: Provide a prompt that describes the video you want to generate.
2. **Model Setup**: Set up the model, configure the scheduler, enable slicing for memory efficiency, and load the model.
3. **Generate Video**: Use the model to generate a video based on the input prompt and desired resolution.
4. **Export Video**: Save the generated frames as a video file.


---

### Prompt and Negative Prompt (Sample) 
prompt = "A stylish man sips espresso outside a Paris café, dressed in a tailored camel trench coat, black turtleneck, slim-fit trousers, and leather loafers. A beret rests on his head as he reads a fashion magazine. The cobblestone street, wrought-iron chairs, and blooming flowers enhance the chic yet casual Parisian vibe."
negative_prompt = "Avoid sloppy, unkempt appearance; no casual t-shirts, baggy jeans, or sneakers. Not wearing athletic wear, baseball caps, or bright neon colors. No tropical print shirts or cargo shorts. Setting is not a fast food restaurant, shopping mall, or suburban street. Not holding a smartphone or taking selfies. No crowded tourist attractions or modern glass buildings in background. Avoid plastic furniture, concrete sidewalks, or chain store signage. Not wearing sunglasses or headphones. No busy traffic or parked cars visible. Not a rainy or overcast day. No neon lighting or nightclub atmosphere. Not surrounded by tourists with cameras. No modern technology visible. Not casual American streetwear aesthetic."

### Output of the video (1280 * 720) 

https://github.com/user-attachments/assets/32e62fc4-8331-4e0e-aa1c-a7a0804fa3e6



## Future Improvements

* **Fine-Tuning**: You can fine-tune the model on domain-specific data, such as fashion-specific video datasets, to improve performance on niche use cases.
* **Video Length**: Extend the length of the generated video by modifying the number of frames and adjusting the model's capabilities.
* **Multilingual Support**: Support for different languages by enhancing the model with multilingual training data.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to update the README content based on any further customizations or details you want to include.
