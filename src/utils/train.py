"""Contains the training loop for the distilling encoders' embedding space."""

import torch
from tqdm import tqdm


def train_model(
    model, 
    dataloader,
    model_path: str, 
    epochs=3, 
    lr=1e-4, 
    optimizer=None, 
    scheduler=None
):
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    
    for epoch in tqdm(range(epochs), desc="Training"):
        total_loss = 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for batch in batch_pbar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.6f}"})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        batch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.6f}"})

        # Save both model and training states
        torch.save(model.state_dict(), model_path)
        torch.save(
            {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 
            model_path.replace('.pth', '_checkpoint.pth')
        )