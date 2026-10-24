# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check published Studio/library skill namespaces and their adapters."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import agent_skills


class PublishedSkillNamesTest(unittest.TestCase):
    """Product-prefixed using skills stay distinct from local contributor skills."""

    def check_name(self, audience: str, name: str, expected: int) -> None:
        """Validate a synthetic skill without touching repository adapters."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group in ("using", "contributing"):
                (root / "skills" / group).mkdir(parents=True)
            skill = root / "skills" / audience / name
            skill.mkdir()
            (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(agent_skills.cmd_sync(root), 0)
                self.assertEqual(agent_skills.cmd_validate(root), expected)

    def test_published_product_names(self) -> None:
        """Both application and library workflows have a product namespace."""
        for name in ("pas-studio-cli", "pai-export-policy"):
            with self.subTest(name=name):
                self.check_name("using", name, 0)

    def test_using_skills_need_a_prefix(self) -> None:
        """A bare workflow name must not occupy the shared install namespace."""
        self.check_name("using", "hf-cli", 1)

    def test_contributing_skills_do_not_use_product_prefixes(self) -> None:
        """Repository-local workflows use neither published product prefix."""
        self.check_name("contributing", "add-policy", 0)
        for name in ("pas-studio-cli", "pai-export-policy"):
            with self.subTest(name=name):
                self.check_name("contributing", name, 1)


if __name__ == "__main__":
    unittest.main()
