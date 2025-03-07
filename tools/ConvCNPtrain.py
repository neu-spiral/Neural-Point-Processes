import torch
from sklearn.metrics import r2_score
import torch.optim as optim
from npf import CNPPFLoss, CNPFLoss

def apply_mask_to_tensor(X, Y, batch_size, height, width, pins, values):
    """
    Apply the given mask to the tensor and fill the mask locations with the given values.
    
    Args:
        mask (torch.Tensor): The mask tensor (batch_size, height, width, 1).
        tensor (torch.Tensor): The tensor where the values will be applied (batch_size, height, width, y_dim).
        batch_size (int): The size of the batch.
        height (int): The height of the grid.
        width (int): The width of the grid.
        n_points (int): The number of points (e.g., context or target).
        values (torch.Tensor): The values to place into the tensor (batch_size, n_points, y_dim).
    
    Returns:
        torch.Tensor: The tensor with values placed at the locations defined by the mask.
    """
    n_points = values.shape[1]
    for i in range(batch_size):
        for j in range(n_points):
            # Assuming pins stores x, y coordinates and values stores corresponding values
            x, y = pins[i, j, 0].int().cpu().numpy(), pins[i, j, 1].int().cpu().numpy()  # Extract x, y coordinates from pins
            # Ensure valid coordinates (within bounds)
            if 0 <= x < width and 0 <= y < height:
                X[i, x, y, 0] = 1.0  
                Y[i, x, y, 0] = values[i, j, 0]  
    return X, Y


def process_pins_into_mask(pins, values, height, width, batch_size, n_pins, device, partial_percent):
    """
    Convert pins (coordinates) into masks and fill them with corresponding values.
    Controls how much of the context is revealed based on partial_percent.
    
    Args:
        pins (torch.Tensor): The pins representing coordinates (batch_size, n_pins, 2).
        values (torch.Tensor): The values corresponding to the pins (batch_size, n_pins, 1).
        height (int): The height of the image/grid.
        width (int): The width of the image/grid.
        batch_size (int): The size of the batch.
        n_pins (int): The number of pins (context or target).
        device (torch.device): The device where tensors should be moved.
        partial_percent (float): The percentage of context revealed (e.g., 0.5 means reveal half).
    
    Returns:
        tuple: (X_cntxt, Y_cntxt, X_trgt, Y_trgt) - Masks and values for context and target.
    """
    # Create empty mask tensors (batch_size, height, width, 1)
    X_cntxt = torch.zeros(batch_size, height, width, 1, dtype=torch.float32).to(device)
    X_trgt = torch.zeros(batch_size, height, width, 1, dtype=torch.float32).to(device)
    
    # Fixed number of target points (second half of the points)
    num_context = n_pins // 2  # The target is always the second half
    pins_target = pins[:, num_context:, :]
    values_target = values[:, num_context:, :]
    
    # Adjust context points based on partial_percent (first half)
    revealed_context = int(num_context * partial_percent)  # Only reveal a portion of the first half
    pins_context = pins[:, :revealed_context, :]
    values_context = values[:, :revealed_context, :]
    
    # Apply the values to the context and target masks
    X_cntxt, Y_cntxt = apply_mask_to_tensor(X_cntxt, X_cntxt.clone(), batch_size, height, width, pins_context, values_context)
    X_trgt, Y_trgt = apply_mask_to_tensor(X_trgt, X_trgt.clone(), batch_size, height, width, pins_target, values_target)

    return X_cntxt, Y_cntxt, X_trgt, Y_trgt


def process_batch_grid(batch, device, partial_percent=0.5):
    """
    Main function to process a batch of images, pins, and outputs into the appropriate format
    for GridConvCNP. It converts 2D image coordinates into 1D indices for the mask, and splits the
    data into context and target sets. Also controls the percentage of context revealed using partial_percent.
    
    Args:
        batch (dict): Dictionary containing 'pins' and 'outputs' (both as lists of tensors).
        device (torch.device): The device to which tensors should be moved.
        partial_percent (float): The percentage of context revealed (e.g., 0.5 means reveal half).
    
    Returns:
        tuple: (X_cntxt, Y_cntxt, X_trgt, Y_trgt) - Masks and values for context and target.
    """
    pins = [tensor.to(device) for tensor in batch['pins']]  # pins (input points, coordinates)
    values = [tensor.to(device) for tensor in batch['outputs']]  # values (GT values at the grid points)

    # Extract height and width from the first image (assuming all images have the same size)
    height, width = batch['image'][0].shape[-2], batch['image'][0].shape[-1]
    batch_size, n_pins = len(pins), pins[0].shape[0]  # Get batch size and number of points

    # Stack pins (batch_size, n_pins, 2)
    pins_tensor = torch.stack(pins, dim=0)  # (batch_size, n_pins, 2)
    
    # Stack values (batch_size, n_pins, 1)
    values_tensor = torch.stack(values, dim=0).unsqueeze(-1)  # (batch_size, n_pins, 1)

    # Process pins into masks and fill with values for context and target
    X_cntxt, Y_cntxt, X_trgt, Y_trgt = process_pins_into_mask(pins_tensor, values_tensor, height, width, batch_size, n_pins, device, partial_percent)
    
    data = {
        'X_cntxt': X_cntxt,
        'Y_cntxt': Y_cntxt,
        'X_trgt': X_trgt,
        'Y_trgt': Y_trgt
    }
    
    return data

    
class ConvCNPTrainer:
    def __init__(self, train_loader, val_loader, test_loader, model_2d, criterion, experiment_id, optimizer_class=optim.Adam, scheduler_class=torch.optim.lr_scheduler.StepLR, device='cuda', early_stop_patience=5, **kwargs):
        # Initialize the model
        self.model = model_2d().to(device)
        
        # DataLoader assignments
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        
        # Optimizer and Scheduler initialization
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs.get('optimizer_params', {}))
        self.scheduler = scheduler_class(self.optimizer, **kwargs.get('scheduler_params', {}))
        
        # Initialize other parameters
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.best_val_loss = float('inf')
        self.experiment_id = experiment_id
        self.early_stop_patience = early_stop_patience
        self.epochs_since_improvement = 0
        

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(self.train_loader):
            data = process_batch_grid(batch, self.device)
            X_cntxt = data['X_cntxt'].to(self.device)
            Y_cntxt = data['Y_cntxt'].to(self.device)
            X_trgt = data['X_trgt'].to(self.device)
            Y_trgt = data['Y_trgt'].to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(X_cntxt, Y_cntxt, X_trgt, Y_trgt)
            
            if isinstance(self.criterion, CNPPFLoss):
                # CNPPFLoss requires both X_trgt and Y_trgt
                loss = self.criterion(pred, X_trgt, Y_trgt)
            elif isinstance(self.criterion, CNPFLoss):
                # CNPFLoss only requires Y_trgt
                loss = self.criterion(pred, Y_trgt)
            else:
                raise ValueError("Unsupported loss criterion type.")
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                data = process_batch_grid(batch, self.device)
                X_cntxt = data['X_cntxt'].to(self.device)
                Y_cntxt = data['Y_cntxt'].to(self.device)
                X_trgt = data['X_trgt'].to(self.device)
                Y_trgt = data['Y_trgt'].to(self.device)
                
                # Forward pass
                pred = self.model(X_cntxt, Y_cntxt, X_trgt)
                if isinstance(self.criterion, CNPPFLoss):
                    # CNPPFLoss requires both X_trgt and Y_trgt
                    loss = self.criterion(pred, X_trgt, Y_trgt)
                elif isinstance(self.criterion, CNPFLoss):
                    # CNPFLoss only requires Y_trgt
                    loss = self.criterion(pred, Y_trgt)
                else:
                    raise ValueError("Unsupported loss criterion type.")
                
                running_loss += loss.item()
        
        avg_loss = running_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Train for one epoch
            avg_train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

            # Save checkpoint if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'./history/ConvCNP/{self.experiment_id}/best_model_local.pt')
                print("Model checkpoint saved!")
                self.epochs_since_improvement = 0  # Reset counter after improvement
            else:
                self.epochs_since_improvement += 1

            # Early stopping condition
            if self.epochs_since_improvement >= self.early_stop_patience:
                print("Early stopping triggered!")
                break

            # Learning rate decay
            self.scheduler.step()

        print("Training complete.")

    def evaluate(self, partial_percent=0.5):
        # Reload the model from the checkpoint
        self.model.load_state_dict(torch.load(f'./history/ConvCNP/{self.experiment_id}/best_model_global.pt'))
        self.model.to(self.device)
        self.model.eval()

        total_loss, total_r2, global_r2 = evaluate_ConvCNP(self.model, self.test_loader, self.device, partial_percent)
        print(f"Total Loss: {total_loss}, Total R²: {total_r2}, Global R²: {global_r2}")
        return total_loss, total_r2, global_r2

def evaluate_ConvCNP(model, dataloader, device, partial_percent=0, hidden_samples=0.5):
    model.eval()
    total_loss = 0.0
    total_r2 = 0.0
    all_preds = []
    all_true = []

    if partial_percent == 0:
        print("Error: partial_percent cannot be 0.")
        return None
    else:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                data = process_batch_grid(batch, device, partial_percent)
                X_cntxt = data['X_cntxt'].to(device) #torch.Size([64, 32, 32, 1])  
                Y_cntxt = data['Y_cntxt'].to(device) #torch.Size([64, 32, 32, 1])
                X_trgt = data['X_trgt'].to(device)
                Y_trgt = data['Y_trgt'].to(device)

                # Forward pass
                pred = model(X_cntxt, Y_cntxt, X_trgt)
                y_pred = pred[0].mean  # torch.Size([1, 64, 32, 32, 1])

                # Calculate per-image R² scores
                r2_scores = 0
                mse_scores = 0

                for i in range(len(y_pred)):
                    # Ensure y_pred and y_true are of the same size and shape
                    y_pred_i = y_pred.squeeze()[i]  # Ensure it's 1D or 2D
                    y_true_i = Y_trgt[i].squeeze()  # True values for the context points

                    # Mask for context points (use X_cntxt as mask for locations)
                    mask_i = X_trgt[i].bool().squeeze()

                    # Apply the mask to both the predicted and true values
                    y_pred_i_masked = y_pred_i[mask_i]  # Select the predicted values for context points
                    y_true_i_masked = y_true_i[mask_i]  # Select the true values for context points

                    # Ensure the sizes match
                    if y_pred_i_masked.size() != y_true_i_masked.size():
                        print(f"Shape mismatch: y_pred[{i}].size() = {y_pred_i_masked.size()}, y_true[{i}].size() = {y_true_i_masked.size()}")
                        continue  # Skip this iteration if there's a size mismatch
    
                    # Compute MSE and R² for the context points
                    mse = torch.mean((y_pred_i_masked - y_true_i_masked) ** 2)
                    if torch.allclose(y_true_i_masked, y_true_i_masked[0], atol=1e-10):
                        if torch.allclose(y_pred_i, y_true_i[0], atol=1e-10):
                            r2 = 1.0  # Perfect prediction for constant target
                        else:
                            r2 = 0.0  # Incorrect prediction for constant target
                    else:
                        # Compute R² score for the current image
                        r2 = r2_score(y_pred_i_masked.cpu().numpy(), y_true_i_masked.cpu().numpy())
                    r2_scores += r2
                    
                    
                    
                    mse_scores += mse.item()
                    all_preds.append(y_pred_i_masked)
                    all_true.append(y_true_i_masked)

                # Accumulate total R² and loss
                total_r2 += r2_scores / len(y_pred)
                total_loss += mse_scores / len(y_pred)

                # Collect all predictions and true values for global R² calculation
                

        # Compute average total R² and loss over all batches
        total_r2 /= len(dataloader)
        total_loss /= len(dataloader)
        
        # Compute global R² over all concatenated predictions and true values
        all_preds = torch.cat(all_preds, dim=0)
        all_true = torch.cat(all_true, dim=0)
        global_r2 = r2_score(all_preds.squeeze().cpu().numpy(), all_true.squeeze().cpu().numpy())
        
        return total_loss, total_r2, global_r2