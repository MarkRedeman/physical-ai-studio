# Studio CLI

`physicalai-studio` manages the installed Studio application and its local assets.
It is separate from the library/Runtime `physicalai` executable, which owns policy
training, benchmarking, and inference-format export commands.

## Discover the installed commands

Start with help; it does not start the server, apply migrations, or transfer data:

```bash
physicalai-studio --help
physicalai-studio serve --help
physicalai-studio datasets import-dir --help
physicalai-studio hf model import --help
```

`-h` also works. Options belong after their command, not at the root: for example,
`physicalai-studio serve --port 7861`. Available commands and options depend on the
installed version; do not assume an option exists because another group has it.

| Task | Command | Important behavior |
| --- | --- | --- |
| Start the web application | `serve` | Runs storage/database migrations before serving |
| Check database status | `db check` | Does not apply migrations, but can create directories/an empty SQLite file |
| Initialize or upgrade a database | `db init` | Applies pending schema migrations to new or existing databases |
| Migrate an installation | `db migrate` | Handles legacy storage migration and database upgrades |
| Import a local model directory | `models import-dir` | Copies by default; needs project, local dataset, and display name |
| Import ACT checkpoints | `models import-act` | Imports child-directory checkpoints (skips root-level `last.ckpt`); moves files and exports Torch/OpenVINO |
| Import a local dataset directory | `datasets import-dir` | Copies by default; needs project, existing environment, and display name |
| Inspect/import/export Hub assets | `hf model` / `hf dataset` | Supports JSON inspection and resumable transfers; [detailed guide](huggingface_integration.md#studio-hub-cli) |
| Generate OpenAPI JSON | `gen-api` | Developer utility; imports the application and overwrites the target file |

`db clean` is **destructive**, runs without confirmation, and removes project data
and jobs from the database without deleting stored asset files. It is not a
troubleshooting step for a missing project or a migration/configuration problem.

## Select the correct local installation

Commands use the local Studio storage and database; there is no remote Studio
HTTP-server selector. Use the same environment and configuration as the backend:

- `STORAGE_DIR`: persistent data root; includes datasets, models, and transfer state.
- `SETTINGS_FILE`: user settings JSON, including saved Hugging Face credentials.
- `DATABASE_FILE`: SQLite filename under `<storage_dir>/data/`.
- Database maintenance also needs the deployment's Alembic configuration
  (`ALEMBIC_CONFIG_PATH` and `ALEMBIC_SCRIPT_LOCATION` if overridden).

For Docker, run the CLI **inside the backend container**, where the correct mounts
and configuration are available. Running the executable on the host does not
select the container's database automatically. A source checkout is not required
for an installed application; contributors can prefix commands with `uv run` from
their configured backend environment.

If a project or asset is unexpectedly missing, verify these settings first.
Do not initialize or clean another database to make an import proceed. Use UUIDs
from existing Studio records/API responses, not names or Hub repository IDs.
The current CLI has no project/model/dataset listing commands.

## Start Studio

```bash
physicalai-studio serve --host 127.0.0.1 --port 7861
```

The defaults come from the `HOST` and `PORT` settings (normally `127.0.0.1:7860`).
Startup can modify storage and the database through migrations. Back up an
existing installation before upgrading it. Studio's API has no authentication
model; do not bind to an untrusted network merely to work around a connection issue.

## Import local folders

Replace UUID placeholders with existing Studio record IDs and paths with folders
on the machine running the CLI:

```bash
physicalai-studio models import-dir \
  --source-dir ./model --project-id PROJECT_UUID \
  --dataset-id DATASET_UUID --model-name "Imported model"

physicalai-studio datasets import-dir \
  --source-dir ./dataset --project-id PROJECT_UUID \
  --environment-id ENVIRONMENT_UUID --dataset-name "Imported data"

physicalai-studio models import-act \
  --source-dir ./checkpoints --project-id PROJECT_UUID \
  --dataset-id DATASET_UUID
```

The model's associated dataset must belong to the target project and describe a
suitable environment. The dataset import accepts LeRobot v3 directories and
requires an existing environment in that project. Both default to `--copy`;
`--move` removes the source folder on success. Neither command supports `--json`,
`--dry-run`, or the Hub transfer `--resume` interface.

`models import-act` imports each `.ckpt` directly inside a checkpoint
subdirectory as a separate ACT model; root-level checkpoint files are ignored.
The dataset must belong to the specified project. Checkpoint files are moved to
Studio storage before Torch and OpenVINO exports are created.

For a Hub repository, use `hf model import` or `hf dataset import`, not these
local-folder commands. In particular, **Hub dataset import creates a temporary
empty environment**, whereas local-folder import requires `--environment-id`.
See the [Hub CLI guide](huggingface_integration.md#studio-hub-cli) for exact options,
credential resolution, inspection, publishing safeguards, and recovery commands.

## Automation and troubleshooting

- Only the `hf` subcommands currently provide structured `--json` output.
  Their import/export commands additionally offer `--dry-run`. A dry run can
  read metadata/local files; it does not transfer payloads or register assets.
- `db check` exits with 0 for a current schema, 1 for connection/errors, and 2
  when migration is needed or its status cannot be determined. Inspect the
  diagnostic before authorizing migrations; an unknown status is not proof of an
  outdated schema. Missing migration configuration is not a reason to reset data.
- If a command/option is missing, compare its `--help` with your installed
  `physicalai-studio` version (`importlib.metadata.version("physicalai-studio")`
  in the same Python environment). Do not invent flags or patch installed packages.
- For interrupted Hub transfers, use the exact printed resume command and retain
  staging/cache files. A completed import is not a successful inference test.
