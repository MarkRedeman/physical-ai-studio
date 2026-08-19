# Plan: Robot Plugin Installation System

## Overview

Users will install robot plugins (e.g. `physicalai-rebot-b601-plugin`) from a new **Plugins** page (`/plugins`). A curated **manifest** (shipped in the backend) lists known plugins with their descriptions and the robots they add. The backend discovers installed plugins via the existing `physicalai.studio.catalog_plugins` entry-point group. Installing = `uv pip install <package>` into the active venv + restart; the robot catalog registry (a module-level discriminated union built at import time in `schemas/robot.py:25`) requires a server restart to pick up new robot types.

Because robot types are baked into the `Robot` union at import time, the picker and robot API can only see installed types. The manifest + a new `/api/plugins` endpoint bridge that gap: it reports *available* (manifest) vs *installed* (entry points) plugins.

---

## Backend

### 1. Plugin manifest — `application/backend/src/plugins/manifest.json` + `manifest.py`

New `src/plugins/` package (no clash with the `physicalai_studio_plugin` SDK).

Manifest shape (curated, keyed by PyPI/distribution name):

```json
[
  {
    "id": "physicalai-rebot-b601-plugin",
    "name": "ReBot Plugin",
    "description": "ReBot B601 and Arm102 robot integrations.",
    "category": "ReBot",
    "source": "first_party",
    "repo_url": "https://github.com/open-edge-platform/physical-ai-rebot-b601-plugin",
    "install_source": "git+https://github.com/open-edge-platform/physical-ai-rebot-b601-plugin@main#subdirectory=packages/physicalai-rebot-b601-plugin",
    "robots": [
      { "type": "ReBot_B601_DM_Follower", "display_name": "ReBot B601 DM Follower", "role": "follower" },
      { "type": "ReBot_Arm102_Leader", "display_name": "ReBot Arm102 Leader", "role": "leader" }
    ]
  }
]
```

- Entries for the five known plugins: `physicalai-rebot-b601-plugin`, `physicalai-bimanual-so101-plugin`, `physicalai-lekiwi-plugin`, `physicalai-lerobot-plugin` (dynamic robots — declare an empty/optional `robots` list and a note; real types come from the live catalog once installed), `physicalai-mujoco-so101-plugin`.
- `install_source` = installable spec (`uv pip install <spec>`). Must be verified against actual PyPI/GitHub availability at implementation time.
- `src/plugins/manifest.py`: Pydantic models (`PluginManifestEntry`, `ManifestRobot`, `PluginExtensionEntry`) + loader that reads the JSON via `Path(__file__).parent / "manifest.json"` (packaged by hatch `only-include = ["src"]`).

### 1b. LeRobot extensions — gated add-ons

The `physicalai-lerobot-plugin` discovers robot/teleoperator configs dynamically via
LeRobot's third-party plugin mechanism (`lerobot_robot_*` / `lerobot_teleoperator_*`
distributions). To let users add those after installing the LeRobot plugin:

- Manifest entries support `extensions: [{ id, name, description, repo_url, install_source }]`.
- Extensions are gated behind their parent plugin: `PluginManager.install` refuses to
  install an extension until the parent (`physicalai-lerobot-plugin`) is installed.
- The manifest lists the known add-ons (SO101 slider robot/teleoperator variants,
  SpaceMouse teleoperator, yam robot, xArm robot).
- The API returns them under `PluginResponse.extensions` (with installed state + version);
  the `/plugins` UI renders them inside the installed LeRobot plugin card.

### 2. Plugin manager — `src/plugins/plugin_manager.py`

`PluginManager` with:
- `list_plugins()` → for each manifest entry, merge installed state:
  - Discovery via `importlib.metadata.entry_points(group="physicalai.studio.catalog_plugins")`; match by `ep.dist.name` (normalized). Returns `name`, `version`, `Summary`.
  - For installed plugins, cross-reference the live `RobotCatalogRegistry` to mark each manifest robot `installed: true` and merge any extra catalog types (covers LeRobot's dynamic types).
  - For uninstalled plugins, robots come from the manifest with `installed: false`.
- `installed_dist(plugin_id)` → `importlib.metadata.distribution(...)` or `None`.
- `install(plugin_id)` → `subprocess.run(["uv", "pip", "install", "--python", sys.executable, install_source])` with a reasonable timeout; raises on failure.
- `robot_types_for(plugin_id)` → manifest robot types (for the uninstall guard).
- `uninstall(plugin_id)` → guarded by DB check (below), then `subprocess.run(["uv", "pip", "uninstall", "--python", sys.executable, dist_name])`.

Make `RobotCatalogRegistry` expose the entry-point distribution name for a definition (or a helper mapping catalog `type` → owning dist) so installed robots can be attributed to a plugin. Minimal change: in `_load_external_plugins`, record `self._plugin_distributions[type] = discovered_entry_point.dist.name` alongside each registration.

### 3. Uninstall safety — existing-robots guard

Mirrors the `ResourceInUseError` pattern in `robot_service.py:74-88`:
- Before uninstalling, query the sync DB session (`db.engine.get_sync_db_session`) for `ProjectRobotDB.type IN plugin.robot_types_for(...)`.
- If any rows exist, refuse with a message like "Cannot uninstall 'ReBot Plugin': 3 robot(s) use types ReBot_B601_DM_Follower, ReBot_Arm102_Leader. Delete them first." Return a 409-style error.
- This prevents the breakage: `ProjectRobotMapper.from_schema` (`project_robot_mapper.py:22`) calls `RobotAdapter.validate_python`, which would fail for unknown types and 500 the whole robots list endpoint. (Guard is the defense; no mapper changes needed.)
- Also expose the count in `GET /api/plugins` (e.g. `in_use_robot_count`) so the UI can show "in use" state and disable the uninstall button.

### 4. API — `src/api/plugins.py`

```
GET  /api/plugins                       → list[PluginResponse]
POST /api/plugins/{plugin_id}/install   → { restart_required: true }
POST /api/plugins/{plugin_id}/uninstall → { restart_required: true }
POST /api/system/restart                → 202 { status: "restarting" }
```

- `PluginResponse`: `id, name, description, category, source, repo_url, installed, installed_version, in_use_robot_count, robots: [{type, display_name, role, installed}]`.
- Install/uninstall call `PluginManager`, return `restart_required: true` on success (registry only rebuilds on boot).
- `POST /api/system/restart`: schedules a delayed `os.kill(os.getpid(), signal.SIGTERM)` (uvicorn's default handler → graceful shutdown → process exit). Docker `restart: unless-stopped` in `docker-compose.yaml:23` brings the same container back with the venv intact (venv lives inside the container, not a volume — note in docs that `docker compose down`/image rebuilds revert installed plugins).
- Register the router in `main.py`.

### 5. CLI — `src/cli/plugins.py`

New `plugin` click group, registered in `cli/__init__.py:53` (`cli.add_command(plugins)`):

```
physicalai-studio plugin list                     # table of available + installed (version, robots)
physicalai-studio plugin install <plugin-id>      # uv pip install; echoes "restart the server"
physicalai-studio plugin uninstall <plugin-id>    # guarded; uv pip uninstall
```

Reuses the same `PluginManager` + DB guard as the API. This satisfies "CLI first" — the API is a thin wrapper over the same service functions.

### 6. Backend tests

- `tests/plugins/test_plugin_manager.py`: manifest loading, installed-vs-available merging (fake distributions/entry points via monkeypatch), install/uninstall subprocess calls (mock `subprocess.run`), uninstall guard (in-memory DB with a robot of that type → refuses; no robots → proceeds).
- `tests/api/test_plugins.py`: `GET /api/plugins` shape; install/uninstall 200s; 409 on guarded uninstall; restart endpoint returns 202 (mock the kill).
- Update `tests/api/test_robot_catalog.py` as needed if registry API changes.

---

## Frontend

### 7. `/plugins` route

- `src/router.tsx`: add `plugins = root.path('/plugins')`, `paths.plugins.index`, and a route under `AppLayout` (pattern of `settings`).
- `src/routes/app/nav-items.tsx:17`: enable the existing `plugins` item (`enabled: true`) and give it an icon from `@geti-ui/ui/icons`.
- New `src/features/plugins/` module:
  - `plugins.hooks.ts` — `usePluginsQuery()` (suspense, `skipInvalidation`), `useInstallPluginMutation`, `useUninstallPluginMutation` (`meta.invalidates` → `['get','/api/plugins']`), `useRestartServerMutation`.
  - `plugins.tsx` view: two sections — **Installed** (version badge, robots list, Uninstall button, disabled + "In use by N robots" when `in_use_robot_count > 0`) and **Available** (Install button, robots it adds, GitHub link). Installed plugins render their gated **Extensions** with per-extension Install/Uninstall. Install/uninstall show per-action loading state, surface errors via `toast`, and invalidate `/api/plugins` (the backend reports installed state immediately after `uv pip install`, before restart).
  - `restart-required.tsx` banner: after any install/uninstall success, show "Restart required to activate changes" + **Restart server** button (calls restart mutation). Since the server dies on restart, the button optimistically shows "Restarting…".
  - `plugins.test.tsx` (MSW via `http` from `src/api/utils`) covering installed/available render, install/uninstall mutations, in-use disable, and restart banner.
- Route shell `src/routes/plugins/index.tsx` → `<PluginsView />`.

### 8. Robot picker — `robot-catalog-dialog.tsx`

The picker is `RobotCatalogDialog` (opened from `form.tsx:23`); `schema-form.tsx` renders the selected robot's fields and is not the picker itself. Update the dialog to show uninstalled robots:

- Fetch `GET /api/plugins` alongside `GET /api/robots/catalog`.
- For each **uninstalled** plugin, render a distinct section (e.g. "Available plugins — not installed") with the plugin's manifest robots as ghosted cards (`activeType === null`, reduced opacity, "Not installed" badge) plus an **Install plugin** action that navigates to `/plugins` (optionally with `?plugin=<id>` to scroll/highlight).
- Keep existing category grouping, role filter, "Show external plugins" toggle, and search working on installed entries; uninstalled entries respect the role filter and search too.
- Reuse the existing `CATALOG_MANIFEST` thumbnail/description mapping for known categories, falling back to the placeholder logo for unknown ones (thumbnail assets stay UI-side for now).
- Update `robot-catalog-dialog` tests + add coverage for the uninstalled section.

### 9. Regenerate API types

After the backend contract lands: `npm run build:api:download && npm run build:api`, committing `openapi-spec.json`, `openapi-spec.d.ts`, and UI changes together.

---

## Verification

- Backend: `prek run --all-files application/backend/` (ruff, pyrefly, tests).
- UI: `npm run format:check`, `npm run lint`, `npm run cyclic-deps-check`, `npm run type-check`, `npm run test:unit`, `npm run build`.
- Manual: `physicalai-studio plugin list` → shows manifest + installed; install a plugin → UI banner → restart → picker shows new robots.

## Out of scope / follow-ups

- Install-in-place inside the picker modal (link to `/plugins` for now).
- Plugin updates/version bumps beyond re-install.
- Persisting user-installed plugin state across image rebuilds (volumes for site-packages is fragile; document instead).
- `source`-based hiding of uninstalled plugins (only "external" filter affects installed catalog today).
