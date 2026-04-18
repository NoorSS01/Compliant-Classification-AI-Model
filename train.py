import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "data.csv"
OUTPUT_DIR = "./saved_model"

def load_data(filepath):
    """Loads dataset, splits into train/validation, and formats labels."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text', 'label'])
    
    unique_labels = sorted(df['label'].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}
    
    df['label_id'] = df['label'].map(label2id)
    
    # 80-20 Train/Test split via Stratified sampling
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label_id']
    )
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
    
    # Remove pandas index column if it exists
    if '__index_level_0__' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(['__index_level_0__'])
        val_dataset = val_dataset.remove_columns(['__index_level_0__'])
        
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    return dataset, label2id, id2label


def preprocess(examples, tokenizer):
    """Tokenizes the text data with truncation and padding."""
    return tokenizer(examples['text'], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    """Computes evaluation metrics: Accuracy and F1 Score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {"accuracy": accuracy, "f1_score": f1}


def train():
    """Main function to configure and run the Hugging Face Trainer."""
    # 1. Load Data Setup
    dataset, label2id, id2label = load_data(DATA_FILE)
    
    # 2. Tokenizer & Preprocessing
    print(f"Initializing tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
    
    # Dynamic padding for efficiency
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 3. Model Initialization
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",       # evaluate at the end of each epoch
        save_strategy="epoch",       # save model at the end of each epoch
        load_best_model_at_end=True, # load best model based on metric
        metric_for_best_model="f1_score",
        logging_dir='./logs',
        logging_steps=5,
        report_to="none"             # cleanly run locally without wandb prompts
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 6. Train execution
    print("\nStarting training...\n")
    trainer.train()
    
    # 7. Evaluate on validation set and Print Detailed Metrics
    print("\n--- Final Evaluation ---")
    eval_results = trainer.evaluate()
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score (Weighted): {eval_results['eval_f1_score']:.4f}")
    
    # Generate full classification report
    print("\nGenerating Classification Report...")
    val_predictions = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(val_predictions.predictions, axis=-1)
    labels = val_predictions.label_ids
    
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(labels, preds, target_names=target_names)
    print("\n" + report)
    
    # 8. Save production-ready Model
    print(f"\nSaving best state model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training pipeline complete! Model is ready for inference in app.py!")

if __name__ == "__main__":
    train()
