
pip3 install torch torchvision torchaudio # 12.4
# For LUMI
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install diffusers transformers accelerate bitsandbytes peft xformers -U

conda install conda-forge::huggingface_hub conda-forge::datasets conda-forge::tqdm conda-forge::wandb -y
