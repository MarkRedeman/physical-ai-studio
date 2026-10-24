---
name: pas-studio-cli
description: Uses the installed physicalai-studio application CLI to start Studio, check its database, import local models/datasets, and inspect or transfer Hugging Face assets. Use for Studio application commands and recovering interrupted Hub transfers. DO NOT USE FOR policy training, inference-backend conversion (use pai-train-policy or pai-export-policy), or changing Studio source.
license: Apache-2.0
---

# Use the Studio CLI

Use `physicalai-studio` for application management, not the library/Runtime
`physicalai` executable. No Studio source checkout is required. Let installed
command help define available options; use the
[Studio CLI guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/docs/studio_cli.md)
for detailed procedures rather than duplicating them here.

## Discover and run

1. Run `physicalai-studio --help`, then the selected group's/subcommand's `--help`.
   If a command is missing, report the installed version rather than patch packages.
2. Confirm the intended Studio installation and its `STORAGE_DIR`, `SETTINGS_FILE`,
   and database configuration. These commands operate locally, not through a remote
   Studio HTTP server. For Docker, run inside the appropriate backend container.
3. Choose the route below. Read its help/docs before acting; do not assume all
   commands support `--json`, `--dry-run`, or the same import association flags.
4. Confirm permission before large transfers, public publishing, overwrites,
   hardware access, or database changes. Prefer inspection and dry-run where available.
5. Check the exit code and command-specific result. Report resource IDs/revisions
   or the exact recovery command; do not equate successful import with tested inference.

| Task                             | Command entry point                                                              | Reference                                                                                                                                              |
| -------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Start Studio                     | `physicalai-studio serve`                                                        | [Application usage](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/README.md#usage)                            |
| Check database status            | `physicalai-studio db check`                                                     | `physicalai-studio db --help`                                                                                                                          |
| Import a local folder            | `physicalai-studio models import-dir` or `physicalai-studio datasets import-dir` | Selected command's `--help`                                                                                                                            |
| Inspect/import/export Hub assets | `physicalai-studio hf model` or `physicalai-studio hf dataset`                   | [Hub CLI guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/docs/huggingface_integration.md#studio-hub-cli) |

## Non-obvious boundaries

- `serve` runs migrations before starting. Never use `db clean` or initialize a new
  database to work around a missing project or configuration mismatch.
- Local-folder and Hub dataset imports have different environment requirements;
  follow their individual help. Model imports need a suitable local dataset association.
- For Hub imports, inspect first, check `importable` (not just exit code 0), and pin
  the returned commit. For interruptions, use the printed `--resume` command rather
  than starting another transfer; preserve staging/cache files and source content.
- Hub credentials come from saved Studio settings, then `HF_TOKEN`; cached Hub login
  is not automatically used. Never request or expose tokens. Publishing to the Hub
  is not conversion to ONNX/OpenVINO; use `pai-export-policy` for that workflow.
