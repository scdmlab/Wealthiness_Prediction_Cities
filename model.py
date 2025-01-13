import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, classification_report, confusion_matrix

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.gate(x)

class MMOEModel(nn.Module):
    def __init__(self, base_model, num_experts=5, num_tasks=3, hidden_dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Base model (Swin Transformer)
        self.base_model = base_model
        self.feature_dim = self.base_model.config.hidden_size
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(self.feature_dim, hidden_dim) 
            for _ in range(num_experts)
        ])
        
        # Gate networks (one per task)
        self.gates = nn.ModuleList([
            GateNetwork(self.feature_dim, num_experts)
            for _ in range(num_tasks)
        ])
        
        # Task-specific towers
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 2)  # Binary output per task
            ) for _ in range(num_tasks)
        ])
        
        # Final classifier for combining task outputs
        self.final_classifier = nn.Linear(num_tasks * 2, 3)  # 3 classes
        
    def forward(self, pixel_values, labels=None):
        # Get base features
        base_outputs = self.base_model(pixel_values, output_hidden_states=True)
        features = base_outputs.hidden_states[-1][:, 0, :]  # Use [CLS] token
        
        # Expert outputs
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden_dim]
        
        # Gate outputs for each task
        gate_outputs = [gate(features) for gate in self.gates]  # List of [batch, num_experts]
        
        # Task-specific outputs
        task_outputs = []
        for task_id, gate_output in enumerate(gate_outputs):
            # Weighted combination of experts for this task
            gate_output = gate_output.unsqueeze(-1)  # [batch, num_experts, 1]
            task_input = (expert_outputs * gate_output).sum(dim=1)  # [batch, hidden_dim]
            
            # Task-specific tower
            task_output = self.task_towers[task_id](task_input)  # [batch, 2]
            task_outputs.append(task_output)
            
        # Combine task outputs
        combined_output = torch.cat(task_outputs, dim=1)  # [batch, num_tasks*2]
        logits = self.final_classifier(combined_output)  # [batch, 3]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

def build_classification_trainer(args, train_set, val_set, feature_extractor):
    assert args.num_classes == 3  # MMOE is designed for 3-class classification
    
    # Load base model
    base_model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swin-large-patch4-window12-384-in22k",
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Create MMOE model
    model = MMOEModel(
        base_model=base_model,
        num_experts=5,
        num_tasks=3,
        hidden_dim=512
    )
    
    arguments = TrainingArguments(
        args.log_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=3,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=True,
        # Add weight decay and learning rate scheduling
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        # Add early stopping
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=feature_extractor,
        data_collator=collect_fn2("microsoft/swin-large-patch4-window12-384-in22k"),
        compute_metrics=compute_matrics_for_classification,
    )
    
    return trainer, model