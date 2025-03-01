import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from diffusers import FluxPipeline  # Importing Flux.1 model

# Set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust based on model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations using with_transform
def transform_fn(examples):
    examples["image"] = [transform(img) for img in examples["image"]]
    return examples


# Load CUB dataset using HuggingFace datasets
dataset = load_dataset("Donghyun99/CUB-200-2011")
train_dataset = dataset["train"].with_transform(transform_fn)
val_dataset = dataset["test"].with_transform(transform_fn)

# Get the corresponding loader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# Load pre-trained Flux.1 model and modify classifier
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, device_map="balanced")
model = pipe.transformer  # Get the transformer model from the pipeline

if False:
    prompt = "A cat holding a sign that says hello world"
    out = pipe(
        prompt=prompt,
        guidance_scale=0.,
        height=768,
        width=1360,
        num_inference_steps=100,
        max_sequence_length=256,
        generator=torch.Generator(device).manual_seed(0)
        ).images[0]
    out.save("image.png")


# FluxTransformer2DModel doesn't have an fc layer, need to access the final projection layer
num_ftrs = model.proj_out.in_features  # Get input features from projection layer
model.fc = nn.Linear(num_ftrs, 200)  # CUB has 200 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device).bfloat16(), labels.to(device)
            optimizer.zero_grad()

            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            latents = latents.to(device)

            outputs = model(latents, timestep=torch.tensor([100]))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"], batch["label"]
                images, labels = images.to(device).bfloat16(), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_flux1_cub.pth")
            print("Best model saved!")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
