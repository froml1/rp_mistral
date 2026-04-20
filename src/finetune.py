"""
Fine-tune Mistral 7B on confirmed RP annotations using Unsloth + QLoRA.

Requirements:
  pip install unsloth datasets trl

Usage:
  python src/finetune.py [--epochs 3] [--output mistral-rp]

Output:
  data/finetune/mistral-rp-Q4_K_M.gguf  → ready for Ollama
"""

import argparse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "finetune"

MAX_SEQ_LENGTH = 2048
LORA_RANK      = 8
LORA_ALPHA     = 16
BATCH_SIZE     = 1
GRAD_ACC       = 8       # effective batch = 8
LEARNING_RATE  = 2e-4
WARMUP_RATIO   = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=3,          help="Training epochs")
    parser.add_argument("--output",  type=str,   default="mistral-rp", help="Output model name")
    parser.add_argument("--dataset", type=str,   default=str(DATA_DIR / "dataset.jsonl"))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[error] Dataset not found: {dataset_path}")
        print("Run first: python src/dataset_builder.py")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    from unsloth import FastLanguageModel

    print("Loading Mistral 7B with QLoRA 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "mistralai/Mistral-7B-Instruct-v0.3",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,          # auto-detect
        load_in_4bit   = True,          # QLoRA — fits in ~8 Go VRAM
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_RANK,
        lora_alpha          = LORA_ALPHA,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        lora_dropout        = 0,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM
        random_state        = 42,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    from datasets import load_dataset

    print(f"Loading dataset from {dataset_path}...")
    raw = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"  {len(raw)} training examples")

    # Apply ChatML template
    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize         = False,
            add_generation_prompt = False,
        )
        return {"text": text}

    dataset = raw.map(format_example, remove_columns=raw.column_names)

    # ── Train ─────────────────────────────────────────────────────────────────
    from trl import SFTTrainer
    from transformers import TrainingArguments

    out_dir = DATA_DIR / f"{args.output}-lora"

    trainer = SFTTrainer(
        model            = model,
        tokenizer        = tokenizer,
        train_dataset    = dataset,
        dataset_text_field = "text",
        max_seq_length   = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            output_dir              = str(out_dir),
            num_train_epochs        = args.epochs,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACC,
            warmup_ratio            = WARMUP_RATIO,
            learning_rate           = LEARNING_RATE,
            fp16                    = True,
            logging_steps           = 10,
            save_strategy           = "epoch",
            optim                   = "adamw_8bit",
            seed                    = 42,
            report_to               = "none",
        ),
    )

    print(f"Training for {args.epochs} epoch(s)...")
    trainer.train()

    # ── Export GGUF ───────────────────────────────────────────────────────────
    gguf_path = DATA_DIR / args.output
    print(f"Exporting GGUF → {gguf_path}-Q4_K_M.gguf")
    model.save_pretrained_gguf(
        str(gguf_path),
        tokenizer,
        quantization_method = "q4_k_m",
    )

    # ── Generate Modelfile for Ollama ─────────────────────────────────────────
    gguf_file   = gguf_path.parent / f"{args.output}-Q4_K_M.gguf"
    modelfile   = DATA_DIR / "Modelfile"
    modelfile.write_text(
        f'FROM {gguf_file}\n'
        'PARAMETER temperature 0.2\n'
        'PARAMETER num_ctx 4096\n'
        'SYSTEM """You are a narrative analyst specialized in French roleplay transcripts. '
        'Extract structured information from RP scenes and return valid JSON."""\n',
        encoding="utf-8",
    )

    print("\n=== Done ===")
    print(f"GGUF model : {gguf_file}")
    print(f"Modelfile  : {modelfile}")
    print("\nNext steps:")
    print(f"  ollama create {args.output} -f {modelfile}")
    print(f"  RP_MODEL={args.output} python src/interface.py")


if __name__ == "__main__":
    main()
