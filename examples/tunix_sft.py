# -*- coding: utf-8 -*-
"""Tunix SFT Guide Script

Adapted for local execution outside of Google Colab and launched on remote TPU v6e-8 slice via Kinetic.
"""

import json
import logging
import os

import etils.epath as _epath
import jax
import nest_asyncio
import optax
import qwix
import wandb
from dotenv import load_dotenv
from flax import nnx
from huggingface_hub import snapshot_download
from tunix.examples.data import translation_dataset as data_lib
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger, peft_trainer, utils
from tunix.sft.utils import show_hbm_usage

import kinetic
import kinetic.credentials

load_dotenv()
kinetic.credentials.ensure_credentials = lambda *args, **kwargs: None

# Monkey-patch etils.epath to ignore mode argument in mkdir.
# This is a workaround for permission issues in some environments when creating directories.
_orig_mkdir = _epath.Path.mkdir


def safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
  return _orig_mkdir(self, parents=parents, exist_ok=exist_ok)


_epath.Path.mkdir = safe_mkdir

nest_asyncio.apply()

if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
  wandb.login(key=os.environ["WANDB_API_KEY"])
else:
  os.environ["WANDB_MODE"] = "disabled"
  logging.info("WANDB_API_KEY not found. Running wandb in disabled mode.")

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
  logging.info(
    "KAGGLE credentials not found. Skipping interactive kagglehub login."
  )

if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
  logging.info("HF_TOKEN found in environment.")
else:
  logging.info("HF_TOKEN not found. Ensure Hugging Face is authenticated.")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

model_id = "google/gemma-3-270m-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Data
BATCH_SIZE = 32  # Adjust based on TPU memory & model size.
MAX_TARGET_LENGTH = 256  # Adjusted based on your TPU memory and model size.

# Model Setup
# Adjust mesh based on your TPU memory and model size.
# MESH_COUNTS defines the number of devices along each axis of the mesh.
# The axes are named ("fsdp", "tp") where fsdp is Fully Sharded Data Parallel
# and tp is Tensor Parallel.
NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
  MESH_COUNTS = (1, 4)  # 1 for fsdp, 4 for tp
elif NUM_TPUS == 4:
  MESH_COUNTS = (1, 4)  # Use all 4 devices for tensor parallel
elif NUM_TPUS == 1:
  MESH_COUNTS = (1, 1)
else:
  raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [
  MESH_COUNTS,
  ("fsdp", "tp"),
]

# LoRA/QLoRA Configuration
USE_QUANTIZATION = True  # Set to True for QLoRA, False for LoRA
RANK = 16
ALPHA = float(2 * RANK)

# Train
MAX_STEPS = 100
EVAL_EVERY_N_STEPS = 20
NUM_EPOCHS = 3

# Checkpoint saving
FULL_CKPT_DIR = "/tmp/content/full_ckpts/"
LORA_CKPT_DIR = "/tmp/content/lora_ckpts/"
PROFILING_DIR = "/tmp/content/profiling/"


def create_dir(path):
  try:
    os.makedirs(path, exist_ok=True)
    logging.info(f"Created dir: {path}")
  except OSError as e:
    logging.error(f"Error creating directory '{path}': {e}")


@kinetic.run(
  accelerator="tpu-v5litepod",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN", "WANDB_MODE"],
)
def run_tuning():
  create_dir(FULL_CKPT_DIR)
  create_dir(LORA_CKPT_DIR)
  create_dir(PROFILING_DIR)

  ignore_patterns = [
    "*.pth",  # Ignore PyTorch .pth weight files
  ]
  logging.info(f"Downloading {model_id} from Hugging Face...")
  local_model_path = snapshot_download(
    repo_id=model_id, ignore_patterns=ignore_patterns
  )
  logging.info(f"Model successfully downloaded to: {local_model_path}")

  eos_tokens = []
  generation_config_path = os.path.join(
    local_model_path, "generation_config.json"
  )
  if os.path.exists(generation_config_path):
    with open(generation_config_path, "r") as f:
      generation_configs = json.load(f)
    eos_tokens = generation_configs.get("eos_token_id", [])
    logging.info(f"Using EOS token IDs: {eos_tokens}")

  logging.info("\n--- HBM Usage BEFORE Model Load ---")
  show_hbm_usage()

  model_cp_path = local_model_path

  if "gemma-3-270m" in model_id:
    model_config = gemma3_model_lib.ModelConfig.gemma3_270m()
  elif "gemma-3-1b" in model_id:
    model_config = gemma3_model_lib.ModelConfig.gemma3_1b_it()
  else:
    raise ValueError(f"Unsupported model: {model_id}")

  mesh = jax.make_mesh(
    *MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])
  )
  with mesh:
    base_model = params_safetensors_lib.create_model_from_safe_tensors(
      model_cp_path, (model_config), mesh
    )

  tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
  if tokenizer.eos_id() not in eos_tokens:
    eos_tokens.append(tokenizer.eos_id())
    logging.info(f"Using EOS token IDs: {eos_tokens}")

  sampler = sampler_lib.Sampler(
    transformer=base_model,
    tokenizer=tokenizer if "gemma" in model_id else tokenizer.tokenizer,
    cache_config=sampler_lib.CacheConfig(
      cache_size=256,
      num_layers=model_config.num_layers,
      num_kv_heads=model_config.num_kv_heads,
      head_dim=model_config.head_dim,
    ),
  )

  input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
    "Translate this into French:\nHow's the weather today?\n",
  ]

  out_data = sampler(
    input_strings=input_batch,
    max_generation_steps=10,  # The number of steps performed when generating a response.
    eos_tokens=eos_tokens,
  )

  for input_string, out_string in zip(input_batch, out_data.text, strict=True):
    logging.info("----------------------")
    logging.info(f"Prompt:\n{input_string}")
    logging.info(f"Output:\n{out_string}")

  # Define a helper function to apply LoRA (or QLoRA) to the model.
  # This uses the 'qwix' library to inject low-rank adapters into specified layers.
  def get_lora_model(base_model, mesh, quantize=False):
    if quantize:
      # QLoRA uses 4-bit NormalFloat (nf4) quantization for base model weights
      lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
        weight_qtype="nf4",
        tile_size=128,
      )
    else:
      # Standard LoRA keeps weights in original precision
      lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
      )

    model_input = base_model.get_model_input()
    # Apply LoRA to the base model
    lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
    )

    # Ensure the LoRA model parameters are sharded according to the mesh
    with mesh:
      state = nnx.state(lora_model)
      pspecs = nnx.get_partition_spec(state)
      sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
      nnx.update(lora_model, sharded_state)

    return lora_model

  # Create LoRA or QLoRA model based on USE_QUANTIZATION hyperparameter
  lora_model = get_lora_model(base_model, mesh=mesh, quantize=USE_QUANTIZATION)

  logging.info(f"Using {'QLoRA' if USE_QUANTIZATION else 'LoRA'} model")

  # Loads the training and validation datasets
  train_ds, validation_ds = data_lib.create_datasets(
    dataset_name="mtnt/en-fr",
    global_batch_size=BATCH_SIZE,
    max_target_length=MAX_TARGET_LENGTH,
    num_train_epochs=NUM_EPOCHS,
    tokenizer=tokenizer,
  )

  def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != tokenizer.pad_id()
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
      "input_tokens": x.input_tokens,
      "input_mask": x.input_mask,
      "positions": positions,
      "attention_mask": attention_mask,
    }

  full_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/full", flush_every_n_steps=20
  )

  training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    metrics_logging_options=full_logging_options,
    checkpoint_root_directory=FULL_CKPT_DIR,
  )

  # Initialize the PeftTrainer.
  # We pass `lora_model` which has the LoRA adapters applied.
  trainer = peft_trainer.PeftTrainer(
    lora_model, optax.adamw(1e-5), training_config
  ).with_gen_model_input_fn(gen_model_input_fn)

  logging.info("Starting fine-tuning...")
  # Run the training loop within the mesh context for distributed execution
  with mesh:
    trainer.train(train_ds, validation_ds)

  if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
    wandb.init()
    logging.info("Weights & Biases initialized successfully.")


if __name__ == "__main__":
  run_tuning()
