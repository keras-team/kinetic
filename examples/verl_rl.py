import os

import kinetic
from kinetic import Data

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
VERL_BASE_REPO = f"us-docker.pkg.dev/{PROJECT_ID}/kn-your-cluster-name"


@kinetic.submit(
  accelerator="gpu-h100",
  container_image="prebuilt",
  base_image_repo=VERL_BASE_REPO,
  capture_env_vars=["HF_TOKEN", "WANDB_*"],
)
def run_verl_gsm8k_ppo(
  checkpoint_dir: str,
  prepared_data_dir: str | None = None,
  train_max_samples: int = 128,
  val_max_samples: int = 128,
  total_epochs: int = 1,
):
  import subprocess
  from pathlib import Path

  verl_dir = Path("/opt/verl")
  data_dir = Path("/tmp/verl-data/gsm8k")
  checkpoint_root = Path(checkpoint_dir)
  experiment_dir = checkpoint_root / "kinetic-verl" / "gsm8k-ppo"
  experiment_dir.mkdir(parents=True, exist_ok=True)

  if prepared_data_dir is not None:
    data_dir = Path(prepared_data_dir)
  else:
    subprocess.run(
      [
        "python3",
        "examples/data_preprocess/gsm8k.py",
        "--local_save_dir",
        str(data_dir),
      ],
      cwd=verl_dir,
      check=True,
    )

  command = [
    "python3",
    "-m",
    "verl.trainer.main_ppo",
    f"data.train_files={data_dir / 'train.parquet'}",
    f"data.val_files={data_dir / 'test.parquet'}",
    f"data.train_max_samples={train_max_samples}",
    f"data.val_max_samples={val_max_samples}",
    "data.train_batch_size=16",
    "data.max_prompt_length=512",
    "data.max_response_length=512",
    "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.ppo_mini_batch_size=8",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "critic.model.path=Qwen/Qwen2.5-0.5B-Instruct",
    "critic.optim.lr=1e-5",
    "critic.ppo_micro_batch_size_per_gpu=1",
    "algorithm.kl_ctrl.kl_coef=0.001",
    "trainer.project_name=kinetic-verl",
    "trainer.experiment_name=gsm8k-ppo",
    "trainer.logger=console",
    "trainer.val_before_train=False",
    "trainer.nnodes=1",
    "trainer.n_gpus_per_node=1",
    "trainer.save_freq=1",
    "trainer.test_freq=5",
    f"trainer.total_epochs={total_epochs}",
    f"trainer.default_local_dir={experiment_dir}",
    "trainer.default_hdfs_dir=null",
    "trainer.resume_mode=auto",
  ]

  subprocess.run(command, cwd=verl_dir, check=True)

  return {"checkpoints": str(experiment_dir)}


if __name__ == "__main__":
  job = run_verl_gsm8k_ppo(
    Data("gs://your-bucket/verl-checkpoints/", fuse=True)
  )
  print(f"Submitted Kinetic job: {job.job_id}")
  print(job.result())
