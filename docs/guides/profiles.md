# Profiles

A **profile** bundles the four settings that determine where a Kinetic
job runs — project, zone, cluster, and namespace — under a single name.
Instead of re-exporting `KINETIC_*` environment variables for each
cluster you target, you save a profile once and switch between them
with a single command.

Profiles are optional and additive. If you have never run
`kinetic profile create`, the existing env-var and prompt flow is
unchanged.

## When to use profiles

- You work with more than one cluster (for example, a dev cluster and a
  team-shared GPU cluster).
- You move between projects or namespaces often enough that exporting
  env vars manually is slowing you down.
- Multiple people share a machine or dotfiles and need clean switching
  between their own configurations.

For single-cluster setups, plain `KINETIC_*` env vars are still the
simplest path.

## Quick start

Create two profiles — the first becomes active automatically:

```bash
kinetic profile create dev-tpu \
  --project my-ml-dev --zone us-central2-b --cluster dev-tpu

kinetic profile create team-gpu \
  --project my-ml-prod --zone us-east1-b --cluster team-gpu \
  --namespace alice
```

Either pass the fields as flags (as above) or omit them to be prompted
interactively. Any field you don't pass is read from `KINETIC_*` env
vars, then prompted for if still unset.

List what's saved and switch between them:

```bash
kinetic profile ls          # active profile marked with *
kinetic profile use team-gpu
kinetic profile show        # settings of the active profile
```

The first profile you create is automatically promoted to active; after
that, the only way to change the active profile is
`kinetic profile use NAME`.

## Commands

| Command | What it does |
| --- | --- |
| `kinetic profile create [NAME]` | Save a new profile. Fills any field you don't pass as a flag from `KINETIC_*` env vars, then prompts for whatever is still missing. First profile becomes active. |
| `kinetic profile ls` | List all profiles, with `*` on the active one. |
| `kinetic profile use NAME` | Mark `NAME` as the persistent active profile. |
| `kinetic profile show [NAME]` | Show a profile's settings. Defaults to the active profile. |
| `kinetic profile rm NAME` | Delete a profile. Prompts unless `--yes`. If the removed profile was active, no fallback is chosen — run `profile use` to pick a new one. |

## Selecting a profile for one invocation

Use `--profile NAME` (or `KINETIC_PROFILE=NAME`) to target a non-active
profile without changing the stored active one. The flag belongs to the
root `kinetic` command, so it must come **before** the subcommand:

```bash
# Works:
kinetic --profile team-gpu jobs list

# Also works:
KINETIC_PROFILE=team-gpu kinetic jobs list

# Does NOT work — --profile here would be a subcommand flag, not the global override.
kinetic jobs list --profile team-gpu
```

When the selector differs from the stored active profile,
`kinetic profile ls` marks the effective profile with `*` and prints
an annotation like `Active profile: team-gpu (override; stored: dev-tpu)`
underneath the table — so it's always obvious which profile is actually
in effect for this invocation.

## Where profiles sit in the precedence chain

Profiles slot in below env vars and CLI flags:

```text
CLI flag  >  KINETIC_* env var  >  active profile  >  built-in default
```

In practice that means:

- Profiles give you good defaults without silencing explicit overrides.
- `KINETIC_PROJECT=other-proj kinetic up` still targets `other-proj`
  even if the active profile names something else.
- `kinetic --cluster adhoc status` still targets `adhoc` regardless of
  profile.

`kinetic config show` prints the currently-active profile and, for each
setting, where its resolved value came from (`profile`, a `KINETIC_*`
env var, or `default`) — run it any time you want to see which layer
won.

## Storage

Profiles live in a single JSON file at `~/.kinetic/profiles.json`:

```json
{
  "current": "team-gpu",
  "profiles": {
    "dev-tpu":  { "project": "my-ml-dev",  "zone": "us-central2-b", "cluster": "dev-tpu",  "namespace": "default"   },
    "team-gpu": { "project": "my-ml-prod", "zone": "us-east1-b",    "cluster": "team-gpu", "namespace": "alice" }
  }
}
```

Editing the file by hand works, but the CLI is the supported path.
Writes are atomic (tempfile + rename) so a crash mid-write will not
corrupt the store.

If you need to relocate the file (for example, in CI) set
`KINETIC_PROFILES_FILE` to an alternate path.

## Related pages

- [Configuration](../configuration.md) — the full precedence table and
  the list of `KINETIC_*` env vars.
- [Multiple Clusters](../advanced/clusters.md) — how Kinetic treats
  clusters as first-class targets; profiles are the ergonomic layer on
  top.
- [CLI Reference](../cli.rst) — generated reference for every flag.
