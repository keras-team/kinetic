# -*- coding: utf-8 -*-
"""Tunix GRPO Demo

Adapted for local execution and launched on remote TPU via Kinetic.
Based on: https://tunix.readthedocs.io/en/latest/_collections/examples/grpo_gemma.html
"""

import json
import logging
import os
import re

import grain
import jax
import optax
import qwix
import tensorflow_datasets as tfds
from dotenv import load_dotenv
from flax import nnx
from huggingface_hub import snapshot_download
from orbax import checkpoint as ocp
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

import kinetic
import kinetic.credentials

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"


# ====== LoRA ======
RANK = 64
ALPHA = 64.0


# ====== GRPO ======
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 2
NUM_ITERATIONS = 3
BETA = 0.08
EPSILON = 0.2

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 10  # Reduced for quick verification
NUM_TEST_BATCHES = 2  # Reduced for quick verification
EVAL_EVERY_N_STEPS = 5
NUM_EPOCHS = 1
MAX_STEPS = 10  # Reduced for quick verification

LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1
MAX_GRAD_NORM = 0.1

CKPT_DIR = os.path.join(
  os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/content/"), "ckpts/"
)
SAVE_INTERVAL_STEPS = 5
MAX_TO_KEEP = 2

# ====== Prompts ======
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. First, think about the problem and provide your reasoning. Place it between {reasoning_start} and {reasoning_end}. Then, provide the final answer (i.e., just one numerical value) between {solution_start} and {solution_end}."""
TEMPLATE = """<start_of_turn>user {system_prompt} {question}<end_of_turn> <start_of_turn>model """


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(data_dir, split="train") -> grain.MapDataset:
  os.makedirs(data_dir, exist_ok=True)
  # Default to TFDS

  data = tfds.data_source(
    "gsm8k",
    split=split,
    data_dir=data_dir,
    builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
    download=True,
  )

  def _as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")

  dataset = (
    grain.MapDataset.source(data)
    .shuffle(seed=42)
    .map(
      lambda x: {
        "prompts": TEMPLATE.format(
          system_prompt=SYSTEM_PROMPT,
          question=_as_text(x["question"]),
        ),
        "question": _as_text(x["question"]),
        "answer": extract_hash_answer(_as_text(x["answer"])),
      }
    )
  )
  return dataset


match_format = re.compile(
  rf"^[\s]{{0,}}"
  rf"{reasoning_start}.+?{reasoning_end}.*?"
  rf"{solution_start}(.+?){solution_end}"
  rf"[\s]{{0,}}$",
  flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(prompts, completions, **kwargs):
  return [
    0 if match_format.search(response) is None else 3.0
    for response in completions
  ]


def match_format_approximately(prompts, completions, **kwargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.find(reasoning_start) == 0 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


def check_answer(prompts, completions, answer, **kwargs):
  responses = completions
  extracted_responses = [
    guess.group(1)
    if r is not None and (guess := match_format.search(r)) is not None
    else None
    for r in responses
  ]
  scores = []
  assert len(extracted_responses) == len(answer), (
    f"{extracted_responses} and {answer} have mismatching length"
  )
  for guess, true_answer in zip(extracted_responses, answer, strict=False):
    score = 0
    if guess is None or true_answer is None:
      scores.append(0)
      continue
    if guess == true_answer:
      score += 3.0
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      try:
        t_val = float(true_answer)
        if t_val == 0:
          score -= 0.5
        else:
          ratio = float(guess) / t_val
          if 0.9 <= ratio <= 1.1:
            score += 0.5
          elif 0.8 <= ratio <= 1.2:
            score += 0.25
          else:
            score -= 1.0
      except (ValueError, ZeroDivisionError):
        score -= 0.5
    scores.append(score)
  return scores


match_numbers = re.compile(
  rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)


def check_numbers(prompts, completions, answer, **kwargs):
  responses = completions
  extracted_responses = [
    guess.group(1) if (guess := match_numbers.search(r)) is not None else None
    for r in responses
  ]
  scores = []
  for guess, true_answer in zip(extracted_responses, answer, strict=False):
    if guess is None or true_answer is None:
      scores.append(0)
      continue
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except Exception:
      scores.append(0)
      continue
  return scores


def get_lora_model(base_model, mesh):
  lora_provider = qwix.LoraProvider(
    module_path=(
      ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
      ".*attn_vec_einsum"
    ),
    rank=RANK,
    alpha=ALPHA,
  )
  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
    base_model, lora_provider, **model_input
  )
  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)
  return lora_model


@kinetic.run(
  accelerator="v5e-1",
  capture_env_vars=[
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "HF_TOKEN",
    "WANDB_MODE",
    "PYTHONUNBUFFERED",
    "JAX_LOG_COMPILES",
  ],
)
def run_grpo(tokenizer_path):
  logging.basicConfig(level=logging.DEBUG)
  from absl import logging as absl_logging

  absl_logging.set_verbosity(absl_logging.DEBUG)
  logging.getLogger().setLevel(logging.DEBUG)
  # Download model
  ignore_patterns = ["*.pth"]
  print(f"Downloading {MODEL_ID} from Hugging Face...")
  local_model_path = snapshot_download(
    repo_id=MODEL_ID, ignore_patterns=ignore_patterns
  )
  print(f"Model successfully downloaded to: {local_model_path}")

  eos_tokens = []
  generation_config_path = os.path.join(
    local_model_path, "generation_config.json"
  )
  if os.path.exists(generation_config_path):
    with open(generation_config_path, "r") as f:
      generation_configs = json.load(f)
    eos_tokens = generation_configs.get("eos_token_id", [])
    print(f"Using EOS token IDs: {eos_tokens}")

  model_config = gemma_lib.ModelConfig.gemma3_1b_it()

  # ====== Sharding ======
  num_tpus = len(jax.devices())
  if num_tpus == 8:
    mesh_counts = (2, 4)
  elif num_tpus == 4:
    mesh_counts = (1, 4)
  elif num_tpus == 1:
    mesh_counts = (1, 1)
  else:
    raise ValueError(f"Unsupported number of TPUs: {num_tpus}")

  mesh_spec = [
    mesh_counts,
    ("fsdp", "tp"),
  ]

  mesh = jax.make_mesh(
    *mesh_spec, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_spec[0])
  )
  with mesh:
    gemma3 = params_safetensors_lib.create_model_from_safe_tensors(
      local_model_path, (model_config), mesh
    )

  lora_policy = get_lora_model(gemma3, mesh=mesh)

  tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
  if tokenizer.eos_id() not in eos_tokens:
    eos_tokens.append(tokenizer.eos_id())

  # Data
  dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(TRAIN_MICRO_BATCH_SIZE)[
    :NUM_BATCHES
  ]
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None

  # Optimizer
  optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=LEARNING_RATE,
      warmup_steps=WARMUP_STEPS,
      decay_steps=MAX_STEPS,
      end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
  )
  if MAX_GRAD_NORM is not None:
    optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
    )

  # Configs
  checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
  )
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=os.path.join(
      os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/content/"),
      "tmp/tensorboard/grpo",
    ),
    flush_every_n_steps=2,
  )

  cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
      rl_cluster_lib.Role.ACTOR: mesh,
      rl_cluster_lib.Role.REFERENCE: mesh,
      rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="vanilla",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
      actor_optimizer=optimizer,
      eval_every_n_steps=EVAL_EVERY_N_STEPS,
      max_steps=MAX_STEPS,
      mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
      train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
      metrics_logging_options=metrics_logging_options,
      checkpoint_root_directory=CKPT_DIR,
      checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
      max_tokens_to_generate=TOTAL_GENERATION_STEPS,
      max_prompt_length=MAX_PROMPT_LENGTH,
      kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      top_k=TOP_K,
      eos_tokens=eos_tokens,
    ),
  )
  grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
  )

  rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=gemma3,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
  )

  grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
      match_format_exactly,
      match_format_approximately,
      check_answer,
      check_numbers,
    ],
    algo_config=grpo_config,
  )
  print(
    "Calling grpo_trainer.train()."
  )
  grpo_trainer.train(train_dataset, val_dataset)
  print("GRPO training completed.")
  return


if __name__ == "__main__":
  run_grpo(kinetic.Data(GEMMA_TOKENIZER_PATH))
