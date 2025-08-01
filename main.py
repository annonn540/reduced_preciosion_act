import torch
import time
import yaml
import gc
from utils import log_results, get_dataset, get_model, plot_results

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class MemoryTracker:
    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        
    def start_tracking(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()
        
    def get_peak_memory(self):
        if self.is_cuda:
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2
            return peak_memory
        else:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            return memory_mb

def calculate_model_memory_footprint(model, input_shape, device):
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    grad_memory = param_memory
    
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    activation_memory = 0
    hooks = []
    
    def activation_hook(module, input, output):
        nonlocal activation_memory
        if isinstance(output, torch.Tensor):
            activation_memory += output.numel() * output.element_size() / 1024**2
        elif isinstance(output, (list, tuple)):
            for tensor in output:
                if isinstance(tensor, torch.Tensor):
                    activation_memory += tensor.numel() * tensor.element_size() / 1024**2
    
    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(activation_hook))
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    total_estimated_mb = param_memory + grad_memory + activation_memory
    return total_estimated_mb

def train_model(model, train_loader, criterion, optimizer, device, memory_tracker):
    model.train()
    correct, total, loss_total = 0, 0, 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    return loss_total / len(train_loader), correct / total

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct, total, loss_total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_total += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return loss_total / len(test_loader), correct / total

def run_experiment(models, activations, datasets, epochs=10, batch_size=128, learning_rate=0.001, precisions=["fp4"], softmax_precision="fp32"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Softmax precision: {softmax_precision}")
    
    memory_tracker = MemoryTracker(device)

    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            train_loader, test_loader, input_shape, num_classes = get_dataset(dataset_name, batch_size=batch_size)
            print(f"Dataset loaded - Input shape: {input_shape}, Classes: {num_classes}")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        for precision in precisions:
            for model_name in models:
                for activation_name in activations:
                    print(f"\n{'-'*30}")
                    print(f"Training {model_name} ({precision}) with {activation_name} on {dataset_name} (softmax: {softmax_precision})")
                    print(f"{'-'*30}")
                    
                    try:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        model = get_model(model_name, input_shape, num_classes, activation_name, precision, softmax_precision)
                        model = model.to(device)
                        
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = torch.nn.CrossEntropyLoss()
                        
                        num_params = sum(p.numel() for p in model.parameters())
                        print(f"Model parameters: {num_params:,}")
                        
                        estimated_memory = calculate_model_memory_footprint(model, input_shape, device)
                        print(f"Estimated memory footprint: {estimated_memory:.2f} MB")

                        for epoch in range(1, epochs + 1):
                            memory_tracker.start_tracking()
                            start_time = time.time()

                            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, memory_tracker)
                            test_loss, test_acc = test_model(model, test_loader, criterion, device)

                            peak_memory = memory_tracker.get_peak_memory()
                            epoch_time = time.time() - start_time

                            print(f"Epoch {epoch} - Peak Memory: {peak_memory:.2f} MB")

                            log_results({
                                "model": model_name,
                                "activation": activation_name,
                                "dataset": dataset_name,
                                "precision": precision,
                                "epoch": epoch,
                                "train_acc": train_acc,
                                "train_loss": train_loss,
                                "test_acc": test_acc,
                                "test_loss": test_loss,
                                "time": epoch_time,
                                "memory": peak_memory,
                                "estimated_memory": estimated_memory
                            })
                            
                    except Exception as e:
                        print(f"Error training {model_name} ({precision}) with {activation_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

if __name__ == "__main__":
    try:
        config = load_config("config.yaml")
        run_experiment(
            models=config["models"],
            activations=config["activations"],
            datasets=config["datasets"],
            epochs=config.get("epochs", 10),
            batch_size=config.get("batch_size", 128),
            learning_rate=config.get("learning_rate", 0.001),
            precisions=config.get("precisions", ["fp4"]),
            softmax_precision=config.get("softmax_precision", "fp32")
        )
        plot_results("final_results.csv")
        
    except Exception as e:
        print(f"Error running experiments: {e}")
        import traceback
        traceback.print_exc()