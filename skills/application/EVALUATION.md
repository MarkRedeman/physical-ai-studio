# Studio Skills Evaluation

## `studio-creating-a-robot-plugin`

1. **Serial follower package**: Ask the agent to create a serial follower plugin with discovery and a connection selector. It should implement and test the Runtime driver first, add an exportable driver, use a payload-level `connection` UI item, test catalog registration and builders with fakes, and use the catalog and schema endpoints after local installation.
2. **Bimanual network robot**: Ask the agent to create a two-arm TCP follower and leader package. It should use distinct stable types, a typed payload with left and right addresses, no serial connection picker, a plain Runtime composite driver, and compare its schema with `Trossen_Bimanual_WidowXAI_Follower/schema` at port 3000.
3. **Curated URDF plugin**: Ask the agent to prepare a published robot plugin with an included URDF and meshes for UI installation. It should include resources in distribution artifacts, define `RobotAsset` paths and joint mapping, test asset resolution, add a reviewed manifest entry only after the package entry point works, and verify `/catalog`, `/{type}/schema`, and `/{type}/urdf` after restart.
