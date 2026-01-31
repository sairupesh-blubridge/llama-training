
import os
import glob
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# Configuration
OUTPUT_DIR = "/workspace/2b_hindi_indicbart_shards"
TOKENIZER_NAME = "ai4bharat/IndicBART"
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # Set via: export HF_TOKEN=your_token
TARGET_TOKENS = 2_000_000_000 
TOKENS_PER_SHARD = 100_000_000 
BATCH_SIZE = 1000

def process_and_save_shards(output_dir: str, tokenizer_name: str, target_tokens: int, tokens_per_shard: int):
    """Download, tokenize, and save Hindi FineWeb2 data as numpy shards."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Tokens per shard: {tokens_per_shard:,}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=HF_TOKEN, trust_remote_code=True)
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    print(f"Vocab size: {vocab_size}")
    print(f"EOS token ID: {eos_token_id}")
    
    # Load Hindi FineWeb2 dataset (streaming)
    print("\nLoading Hindi FineWeb2 dataset (streaming)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="hin_Deva",
        split="train",
        streaming=True,
    )
    
    # Initialize
    current_shard = []
    current_token_count = 0
    total_tokens = 0
    shard_idx = 0
    
    print("\nProcessing and tokenizing...")
    
    batch_texts = []
    
    for example in tqdm(dataset, desc="Processing examples"):
        batch_texts.append(example["text"])
        
        if len(batch_texts) >= BATCH_SIZE:
            # Tokenize batch
            tokenized = tokenizer(
                batch_texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            
            for tokens in tokenized["input_ids"]:
                # Add EOS token
                tokens_with_eos = tokens + [eos_token_id]
                current_shard.extend(tokens_with_eos)
                current_token_count += len(tokens_with_eos)
                
                # Save shard if full
                if current_token_count >= tokens_per_shard:
                    shard_array = np.array(current_shard[:tokens_per_shard], dtype=np.uint32)
                    shard_file = output_path / f"hindi_shard_{shard_idx:05d}.npy"
                    np.save(shard_file, shard_array)
                    
                    total_tokens += len(shard_array)
                    shard_idx += 1
                    
                    # Keep remainder
                    current_shard = current_shard[tokens_per_shard:]
                    current_token_count = len(current_shard)
                    
                    if total_tokens >= target_tokens:
                        break
            
            batch_texts = []
            if total_tokens >= target_tokens:
                break
    
    # Summary
    print("\n" + "="*60)
    print("TOKENIZATION COMPLETE")
    print("="*60)
    print(f"Total shards: {shard_idx}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output directory: {output_path}")
    
    # Save metadata
    metadata = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "total_tokens": total_tokens,
        "num_shards": shard_idx,
        "tokens_per_shard": tokens_per_shard,
        "source": "HuggingFaceFW/fineweb-2 (hin_Deva)",
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    process_and_save_shards(
        output_dir=OUTPUT_DIR,
        tokenizer_name=TOKENIZER_NAME,
        target_tokens=TARGET_TOKENS,
        tokens_per_shard=TOKENS_PER_SHARD,
    )
