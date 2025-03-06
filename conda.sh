# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# pip install bitsandbytes
pip install diffusers transformers accelerate bitsandbytes peft xformers -U

# conda install conda-forge::transformers conda-forge::xformers conda-forge::accelerate conda-forge::peft conda-forge::bitsandbytes -y
conda install conda-forge::huggingface_hub conda-forge::datasets conda-forge::tqdm conda-forge::wandb -y