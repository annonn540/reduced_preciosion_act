import importlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import csv
import os

sys.path.append(os.getcwd())

def get_model(name, input_shape, num_classes, activation, precision="fp4", use_softmax=False, softmax_precision="fp32"):
    fp2_model_map = {
        "cnn": "fp2_models.fp2_cnn.FP2CNN",
        "gcn": "fp2_models.fp2_gcn.FP2GCN",
        "mlpmixer": "fp2_models.fp2_mlpmixer.FP2MLPMixer",
        "mobilenet": "fp2_models.fp2_mobilenetv2.FP2MobileNetV2",
        "resnet": "fp2_models.fp2_resnet18.FP2ResNet18", 
        "vit": "fp2_models.fp2_vit.FP2VisionTransformer"
    }

    fp3_model_map = {
        "cnn": "fp3_models.fp3_cnn.FP3CNN",
        "gcn": "fp3_models.fp3_gcn.FP3GCN",
        "mlpmixer": "fp3_models.fp3_mlpmixer.FP3MLPMixer",
        "mobilenet": "fp3_models.fp3_mobilenetv2.FP3MobileNetV2",
        "resnet": "fp3_models.fp3_resnet18.FP3ResNet18", 
        "vit": "fp3_models.fp3_vit.FP3VisionTransformer"        
    }

    fp4_model_map = {
        "cnn": "fp4_models.fp4_cnn.FP4CNN",
        "gcn": "fp4_models.fp4_gcn.FP4GCN",
        "mlpmixer": "fp4_models.fp4_mlpmixer.FP4MLPMixer",
        "mobilenet": "fp4_models.fp4_mobilenetv2.FP4MobileNetV2",
        "resnet": "fp4_models.fp4_resnet18.FP4ResNet18", 
        "vit": "fp4_models.fp4_vit.FP4VisionTransformer"
    }

    fp6_model_map = {
        "cnn": "fp6_models.fp6_cnn.FP6CNN",
        "gcn": "fp6_models.fp6_gcn.FP6GCN",
        "mlpmixer": "fp6_models.fp6_mlpmixer.FP6MLPMixer",
        "mobilenet": "fp6_models.fp6_mobilenetv2.FP6MobileNetV2",
        "resnet": "fp6_models.fp6_resnet18.FP6ResNet18", 
        "vit": "fp6_models.fp6_vit.FP6VisionTransformer"
    }
    
    fp8_model_map = {
        "cnn": "fp8_models.fp8_cnn.FP8CNN",
        "gcn": "fp8_models.fp8_gcn.FP8GCN",
        "mlpmixer": "fp8_models.fp8_mlpmixer.FP8MLPMixer",
        "mobilenet": "fp8_models.fp8_mobilenetv2.FP8MobileNetV2",
        "resnet": "fp8_models.fp8_resnet18.FP8ResNet18", 
        "vit": "fp8_models.fp8_vit.FP8VisionTransformer"
    }

    fp32_model_map = {
        "cnn": "fp32_models.fp32_cnn.FP32CNN",
        "gcn": "fp32_models.fp32_gcn.FP32GCN",
        "mlpmixer": "fp32_models.fp32_mlpmixer.FP32MLPMixer",
        "mobilenet": "fp32_models.fp32_mobilenetv2.FP32MobileNetV2",
        "resnet": "fp32_models.fp32_resnet18.FP32ResNet18", 
        "vit": "fp32_models.fp32_vit.FP32VisionTransformer"
    }
    
    if precision.lower() == "fp2":
        model_map = fp2_model_map
    elif precision.lower() == "fp3":
        model_map = fp3_model_map
    elif precision.lower() == "fp4":
        model_map = fp4_model_map
    elif precision.lower() == "fp6":
        model_map = fp6_model_map
    elif precision.lower() == "fp8":
        model_map = fp8_model_map
    elif precision.lower() == "fp32":
        model_map = fp32_model_map
    else:
        raise ValueError(f"Unsupported precision: {precision}. Choose from ['fp2', 'fp3', 'fp4', 'fp6', 'fp8', 'fp32']")
    
    if name not in model_map:
        raise ValueError(f"Unknown model: {name}. Available: {list(model_map.keys())}")
    
    module_path, class_name = model_map[name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    if name == "gcn":
        return model_class(input_shape, num_classes=num_classes, activation=activation)
    elif name == "mlpmixer":
        return model_class(num_classes=num_classes, activation=activation)
    elif name == "vit":
        img_size = input_shape[1] if len(input_shape) == 3 else 28  # Default for MNIST
        return model_class(img_size=img_size, num_classes=num_classes, activation=activation)
    elif name == "cnn":
        return model_class(num_classes=num_classes, activation=activation, use_softmax=use_softmax, softmax_precision=softmax_precision)
    else:
        return model_class(num_classes=num_classes, activation=activation)

def get_dataset(name, batch_size=64):
    try:
        if name == "cifar10":
            from datasets import cifar10
            return cifar10.load_data(batch_size=batch_size)
        elif name == "cifar100":
            from datasets import cifar100
            return cifar100.load_data(batch_size=batch_size)
        else:
            raise ValueError(f"Unknown dataset: {name}. Available: ['cifar10', 'cifar100']")
    except ImportError as e:
        raise ImportError(f"Could not import dataset module '{name}': {e}")

_log_results_initialized = False 

def log_results(info):
    global _log_results_initialized

    precision = info.get('precision', 'fp4')

    print(f"[{info['model']} ({precision}) + {info['activation']} on {info['dataset']} | Epoch {info['epoch']}]")
    print(f"Train Acc: {info['train_acc']:.4f} | Train Loss: {info['train_loss']:.4f}")
    print(f"Test Acc:  {info['test_acc']:.4f} | Test Loss:  {info['test_loss']:.4f}")
    print(f"Time: {info['time']:.2f}s | Peak Memory: {info['memory']:.2f} MB")
    print(f"Estimated Memory: {info.get('estimated_memory', 0):.2f} MB\n")

    csv_file = "final_results.csv"
    fieldnames = [
        "model", "activation", "dataset", "precision", "epoch",
        "train_acc", "train_loss", "test_acc", "test_loss",
        "time", "memory", "estimated_memory"
    ]

    mode = 'w' if not _log_results_initialized else 'a'

    with open(csv_file, mode=mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not _log_results_initialized:
            writer.writeheader()
            _log_results_initialized = True

        writer.writerow({
            "model": info["model"],
            "activation": info["activation"],
            "dataset": info["dataset"],
            "precision": info["precision"],
            "epoch": info["epoch"],
            "train_acc": f"{info['train_acc']:.4f}",
            "train_loss": f"{info['train_loss']:.4f}",
            "test_acc": f"{info['test_acc']:.4f}",
            "test_loss": f"{info['test_loss']:.4f}",
            "time": f"{info['time']:.2f}",
            "memory": f"{info['memory']:.2f}",
            "estimated_memory": f"{info.get('estimated_memory', 0):.2f}"
        })

def plot_results(csv_path="final_results.csv", save_dir="plots"):
    if not os.path.exists(csv_path):
        print(f"[!] CSV file '{csv_path}' not found.")
        return

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    for col in ['model', 'activation', 'dataset', 'precision']:
        df[col] = df[col].astype(str).str.strip()

    group_cols = ['model', 'activation', 'dataset']
    grouped = df.groupby(group_cols)

    for keys, group in grouped:
        model, activation, dataset = keys
        title_base = f"{model.upper()} + {activation.upper()} | {dataset}"
        file_base = f"{model}_{activation}_{dataset}".replace(" ", "_")

        sns.set_theme(style="whitegrid")

        def plot_metric(metric, ylabel, filename_suffix, line_style_map):
            plt.figure(figsize=(10, 6))

            color_map = {
                "train": "tab:blue",
                "test": "tab:green"
            }

            for precision, sub in group.groupby("precision"):
                linestyle = line_style_map.get(precision.lower(), "-")

                plt.plot(
                    sub["epoch"],
                    sub[f"train_{metric}"],
                    label=f"{precision.upper()} - Train",
                    linestyle=linestyle,
                    color=color_map["train"]
                )

                plt.plot(
                    sub["epoch"],
                    sub[f"test_{metric}"],
                    label=f"{precision.upper()} - Test",
                    linestyle=linestyle,
                    color=color_map["test"],
                    linewidth=2
                )

            plt.title(f"{ylabel} vs Epoch\n{title_base}")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{file_base}_{filename_suffix}.png")
            plt.close()

        style_map = {
            "fp2": "dashed",
            "fp3": "dashdot", 
            "fp4": "dotted",
            "fp6": (0, (5, 2, 1, 2)),
            "fp8": (0, (3, 1, 1, 1)),
            "fp32": "solid"
        }

        plot_metric("acc", "Accuracy", "accuracy", style_map)
        plot_metric("loss", "Loss", "loss", style_map)

        plt.figure(figsize=(10, 6))
        for precision, sub in group.groupby("precision"):
            linestyle = style_map.get(precision.lower(), "-")
            plt.plot(sub["epoch"], sub["memory"], label=f"{precision.upper()}", linestyle=linestyle)
        plt.title(f"Peak Memory vs Epoch\n{title_base}")
        plt.xlabel("Epoch")
        plt.ylabel("Memory (MB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{file_base}_memory.png")
        plt.close()

        if 'estimated_memory' in group.columns and not group['estimated_memory'].isna().all():
            plt.figure(figsize=(12, 6))
            for precision, sub in group.groupby("precision"):
                linestyle = style_map.get(precision.lower(), "-")
                plt.plot(sub["epoch"], sub["memory"], 
                        label=f"{precision.upper()} - Peak", linestyle=linestyle, linewidth=2)
                plt.plot(sub["epoch"], sub["estimated_memory"], 
                        label=f"{precision.upper()} - Estimated", linestyle=linestyle, alpha=0.7)
            plt.title(f"Memory Usage: Peak vs Estimated\n{title_base}")
            plt.xlabel("Epoch")
            plt.ylabel("Memory (MB)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{file_base}_memory_comparison.png")
            plt.close()

        plt.figure(figsize=(10, 6))
        for precision, sub in group.groupby("precision"):
            linestyle = style_map.get(precision.lower(), "-")
            plt.plot(sub["epoch"], sub["time"], label=f"{precision.upper()}", linestyle=linestyle)
        plt.title(f"Epoch Time vs Epoch\n{title_base}")
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{file_base}_time.png")
        plt.close()

        print(f"[✓] Grouped plots saved for: {title_base}")