"""Tunix GRPO Example

This launcher keeps Kinetic concerns explicit:

- a tiny local GSM8K-style dataset is passed with kinetic.Data(...)
- Tunix/JAX imports happen only inside the remote TPU function
- checkpoints are written under KINETIC_OUTPUT_DIR

This run is a smoke test. Increase the dataset, model size, and
max_steps before using it as a real post-training recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import kinetic

SMOKE_DATA = [
  {
    "question": "Chloe has 3 bags with 4 marbles each. How many marbles?",
    "answer": "#### 12",
  },
  {
    "question": "A pack has 8 cards. Hoff buys 2 packs. How many cards?",
    "answer": "#### 16",
  },
  {
    "question": "If there are 10 birds and 4 fly away. How many remain?",
    "answer": "#### 6",
  },
  {
    "question": "Ryan reads 5 pages per day for 7 days. How many pages?",
    "answer": "#### 35",
  },
  {
    "question": "A box has 9 pencils. Three boxes have how many pencils?",
    "answer": "#### 27",
  },
  {
    "question": "Gabriel had 20 stickers and gave away 8. How many are left?",
    "answer": "#### 12",
  },
]


def prepare_smoke_data(root: str | Path = "data/tunix_grpo_smoke") -> str:
  """Write a tiny local dataset and return the directory path.

  Replace this with your own JSONL or GCS directory for real runs. The
  remote function expects train.jsonl and test.jsonl files with question
  and answer fields.
  """
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)

  train_path = root / "train.jsonl"
  test_path = root / "test.jsonl"

  with train_path.open("w", encoding="utf-8") as f:
    for record in SMOKE_DATA[:4]:
      f.write(json.dumps(record) + "\n")

  with test_path.open("w", encoding="utf-8") as f:
    for record in SMOKE_DATA[4:]:
      f.write(json.dumps(record) + "\n")

  return str(root)


@kinetic.submit(
  accelerator="tpu-v5litepod-1",
  capture_env_vars=["HF_TOKEN", "WANDB_*"],
)
def run_tunix_grpo(
  data_dir: str,
  model_id: str = "google/gemma-3-270m-it",
  tokenizer_path: str = "gs://gemma-data/tokenizers/tokenizer_gemma3.model",
  max_steps: int = 1,
  train_limit: int = 4,
  eval_limit: int = 2,
):
  import json
  import os
  import re
  from pathlib import Path

  import grain
  import jax
  import optax
  import qwix
  from flax import nnx
  from huggingface_hub import snapshot_download
  from orbax import checkpoint as ocp
  from tunix.generate import tokenizer_adapter as tokenizer_lib
  from tunix.models.gemma3 import model as gemma3_model_lib
  from tunix.models.gemma3 import params_safetensors as params_lib
  from tunix.rl import rl_cluster as rl_cluster_lib
  from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
  from tunix.rl.rollout import base_rollout

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/tunix-grpo")
  checkpoint_dir = f"{output_dir.rstrip('/')}/checkpoints"

  reasoning_start = "<reasoning>"
  reasoning_end = "</reasoning>"
  answer_start = "<answer>"
  answer_end = "</answer>"

  system_prompt = (
    "You are given a math problem. First reason between "
    f"{reasoning_start} and {reasoning_end}. Then put the final numeric "
    f"answer between {answer_start} and {answer_end}."
  )
  prompt_template = (
    "<start_of_turn>user\n"
    "{system_prompt}\n\n"
    "{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
  )

  def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
      return None
    return text.split("####", 1)[1].strip()

  def read_jsonl(path: Path, limit: int):
    rows = []
    with path.open(encoding="utf-8") as f:
      for line in f:
        rows.append(json.loads(line))
        if len(rows) >= limit:
          break
    return rows

  def make_dataset(path: Path, limit: int):
    rows = read_jsonl(path, limit)

    def format_record(record):
      return {
        "prompts": prompt_template.format(
          system_prompt=system_prompt,
          question=record["question"],
        ),
        "question": record["question"],
        "answer": extract_hash_answer(record["answer"]),
      }

    return grain.MapDataset.source(rows).map(format_record).batch(1)

  train_dataset = make_dataset(Path(data_dir) / "train.jsonl", train_limit)
  eval_dataset = make_dataset(Path(data_dir) / "test.jsonl", eval_limit)

  num_tpus = len(jax.devices())
  if num_tpus == 8:
    mesh_shape = (2, 4)
  elif num_tpus in (1, 4):
    mesh_shape = (1, num_tpus)
  else:
    raise ValueError(f"Unsupported TPU count for this example: {num_tpus}")

  mesh = jax.make_mesh(
    mesh_shape,
    ("fsdp", "tp"),
    axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
  )

  print(f"Downloading {model_id} from Hugging Face")
  local_model_path = snapshot_download(
    repo_id=model_id,
    ignore_patterns=["*.pth"],
  )

  if "gemma-3-270m" in model_id:
    model_config = gemma3_model_lib.ModelConfig.gemma3_270m()
  elif "gemma-3-1b" in model_id:
    model_config = gemma3_model_lib.ModelConfig.gemma3_1b_it()
  else:
    raise ValueError(f"Unsupported model for this example: {model_id}")

  with mesh:
    base_model = params_lib.create_model_from_safe_tensors(
      local_model_path,
      model_config,
      mesh,
    )

  lora_provider = qwix.LoraProvider(
    module_path=(
      ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
      ".*attn_vec_einsum"
    ),
    rank=16,
    alpha=32.0,
  )
  actor = qwix.apply_lora_to_model(
    base_model,
    lora_provider,
    **base_model.get_model_input(),
  )

  with mesh:
    state = nnx.state(actor)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(actor, sharded_state)

  tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
  eos_tokens = [tokenizer.eos_id()]

  match_format = re.compile(
    rf"^\s*{reasoning_start}.+?{reasoning_end}.*?"
    rf"{answer_start}(.+?){answer_end}\s*$",
    flags=re.MULTILINE | re.DOTALL,
  )
  match_numbers = re.compile(
    rf"{answer_start}.*?([\d.]+)",
    flags=re.MULTILINE | re.DOTALL,
  )

  def match_format_exactly(prompts, completions, **kwargs):
    del prompts, kwargs
    return [
      0.0 if match_format.search(completion) is None else 3.0
      for completion in completions
    ]

  def match_format_approximately(prompts, completions, **kwargs):
    del prompts, kwargs
    scores = []
    for completion in completions:
      score = 0.0
      score += 0.5 if completion.count(reasoning_start) == 1 else -0.5
      score += 0.5 if completion.find(reasoning_start) == 0 else -0.5
      score += 0.5 if completion.count(reasoning_end) == 1 else -0.5
      score += 0.5 if completion.count(answer_start) == 1 else -0.5
      score += 0.5 if completion.count(answer_end) == 1 else -0.5
      scores.append(score)
    return scores

  def check_answer(prompts, completions, answer, **kwargs):
    del prompts, kwargs
    scores = []
    for completion, true_answer in zip(completions, answer, strict=True):
      guess = match_format.search(completion)
      if guess is None:
        scores.append(0.0)
        continue
      scores.append(3.0 if guess.group(1).strip() == true_answer else 0.0)
    return scores

  def check_numbers(prompts, completions, answer, **kwargs):
    del prompts, kwargs
    scores = []
    for completion, true_answer in zip(completions, answer, strict=True):
      guess = match_numbers.search(completion)
      if guess is None:
        scores.append(0.0)
        continue
      try:
        scores.append(
          1.5 if float(guess.group(1)) == float(true_answer) else 0.0
        )
      except ValueError:
        scores.append(0.0)
    return scores

  optimizer = optax.chain(
    optax.clip_by_global_norm(0.1),
    optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-6,
        warmup_steps=max(1, int(0.1 * max_steps)),
        decay_steps=max(1, max_steps),
        end_value=0.0,
      ),
      b1=0.9,
      b2=0.99,
      weight_decay=0.1,
    ),
  )

  checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=max(1, max_steps),
    max_to_keep=2,
  )
  training_config = rl_cluster_lib.RLTrainingConfig(
    actor_optimizer=optimizer,
    eval_every_n_steps=max(1, max_steps),
    max_steps=max_steps,
    mini_batch_size=1,
    train_micro_batch_size=1,
    checkpoint_root_directory=checkpoint_dir,
    checkpointing_options=checkpointing_options,
  )
  cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
      rl_cluster_lib.Role.ACTOR: mesh,
      rl_cluster_lib.Role.REFERENCE: mesh,
      rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="vanilla",
    offload_to_cpu=False,
    training_config=training_config,
    rollout_config=base_rollout.RolloutConfig(
      max_tokens_to_generate=128,
      max_prompt_length=128,
      kv_cache_size=512,
      temperature=0.9,
      top_p=1.0,
      top_k=50,
      eos_tokens=eos_tokens,
    ),
  )

  rl_cluster = rl_cluster_lib.RLCluster(
    actor=actor,
    reference=base_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
  )
  trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    algo_config=GRPOConfig(
      num_generations=2,
      num_iterations=1,
      beta=0.08,
      epsilon=0.2,
    ),
    reward_fns=[
      match_format_exactly,
      match_format_approximately,
      check_answer,
      check_numbers,
    ],
  )

  print("Starting Tunix GRPO smoke run")
  trainer.train(train_dataset, eval_dataset)
  return {"checkpoints": checkpoint_dir}


if __name__ == "__main__":
  local_data_dir = prepare_smoke_data()
  job = run_tunix_grpo(kinetic.Data(local_data_dir))
  print(f"Submitted Kinetic job: {job.job_id}")
  print(job.result())
