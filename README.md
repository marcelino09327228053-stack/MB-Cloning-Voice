# MB Cloning Voice

Local voice cloning studio built with Python, Gradio, and Qwen3-TTS 1.7B.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11
- Compatible NVIDIA CUDA GPU (8 GB VRAM recommended)
- Qwen3-TTS package and runtime dependencies
- Qwen3-TTS 1.7B Base model downloaded separately

## Setup

This is a source-code release, not a standalone installer. Install the compatible Python dependencies and Qwen3-TTS from the upstream project, download the Qwen3-TTS-12Hz-1.7B-Base model separately, and launch the application with the included Windows launcher after configuring the environment. Dependency compatibility should be verified on the target PC.

Upstream Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
Model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base

## Privacy

Model weights, local voice samples, generated recordings, virtual environments, and machine-specific logs are not included. Only clone voices with appropriate permission.

## Attribution

This application uses Qwen3-TTS, developed by the Qwen team. Qwen3-TTS is distributed separately under Apache License 2.0.
