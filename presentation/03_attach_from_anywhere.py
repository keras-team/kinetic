import sys

import kinetic


def main():
  if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <job_id>")
    sys.exit(1)

  job_id = sys.argv[1]
  print(f"Attaching to {job_id} ...\n")

  handle = kinetic.attach(job_id)

  print(f"  function:    {handle.func_name}")
  print(f"  backend:     {handle.backend}")
  print(f"  accelerator: {handle.accelerator}")
  print(f"  status:      {handle.status().value}\n")

  print("--- Last 20 log lines ---")
  print(handle.tail(n=20))
  print("-------------------------\n")

  print("Blocking on result (streaming live logs)...")
  result = handle.result(stream_logs=True, cleanup=False)
  print(f"\nResult: {result}")

  handle.cleanup()
  print(f"\nCleaned up {job_id}.")


if __name__ == "__main__":
  main()
