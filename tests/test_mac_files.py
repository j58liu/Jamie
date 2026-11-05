import zipfile
from pathlib import Path

import pytest

from assistant.security.approvals import CallbackApprovalProvider
from assistant.tools.mac_files import MacPaths, register_mac_file_tools
from assistant.tools.macos import CommandResult
from assistant.tools.registry import ToolRegistry


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Documents" / "Taxes").mkdir(parents=True)
    (home / "Downloads").mkdir()
    (home / "Desktop").mkdir()
    (home / ".ssh").mkdir()
    (home / "Library" / "Keychains").mkdir(parents=True)
    (home / "Documents" / "notes.md").write_text("Lease renewal due Dec 1\n")
    (home / "Documents" / "Taxes" / "summary.txt").write_text("W-2 from Acme\n")
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY")
    (home / "Library" / "Keychains" / "login.keychain").write_text("secret")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


class NoSpotlight:
    """Behave like a Mac without Spotlight, so the folder walk answers."""

    def __init__(self):
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append(arguments)
        from assistant.tools.base import ToolError

        if arguments[0] == "mdfind":
            raise ToolError("mdfind is unavailable")
        return CommandResult("", "")


class Spotlight:
    def __init__(self, found):
        self.found = found
        self.calls = []

    def run(self, arguments, *, input_text=None, timeout):
        self.calls.append(arguments)
        return CommandResult("\n".join(str(p) for p in self.found), "")


def tools(runner=None, **paths):
    approve = CallbackApprovalProvider(lambda request: True)
    return register_mac_file_tools(ToolRegistry(approvals=approve), MacPaths(**paths), runner)


def run(registry, tool, **args):
    return registry.execute(tool, args)


def test_reads_files_anywhere_in_home_with_tilde_paths(home):
    result = run(tools(NoSpotlight()), "mac_read_file", path="~/Documents/notes.md")
    assert result.success and "Lease renewal" in result.output


def test_private_folders_are_always_refused(home):
    registry = tools(NoSpotlight())
    for path in ("~/.ssh/id_ed25519", "~/Library/Keychains/login.keychain"):
        result = run(registry, "mac_read_file", path=path)
        assert not result.success and "not allowed" in result.output


def test_symlinks_cannot_lead_outside_the_allowed_folders(home, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (home / "Documents" / "link.txt").symlink_to(outside)
    (home / "Documents" / "keys").symlink_to(home / ".ssh")
    registry = tools(NoSpotlight())
    assert not run(registry, "mac_read_file", path="~/Documents/link.txt").success
    assert not run(registry, "mac_read_file", path="~/Documents/keys/id_ed25519").success


def test_configured_roots_and_blocks_narrow_access(home):
    registry = tools(NoSpotlight(), roots=["~/Documents"], blocked=["Documents/Taxes"])
    assert run(registry, "mac_read_file", path="~/Documents/notes.md").success
    assert not run(registry, "mac_read_file", path="~/Documents/Taxes/summary.txt").success
    assert not run(registry, "mac_list_folder", path="~/Downloads").success


def test_search_by_name_and_content_without_spotlight(home):
    registry = tools(NoSpotlight())
    by_name = run(registry, "mac_search_files", query="summary")
    assert "~/Documents/Taxes/summary.txt" in by_name.output
    by_content = run(registry, "mac_search_files", query="lease renewal", match="content")
    assert "~/Documents/notes.md" in by_content.output
    secret = run(registry, "mac_search_files", query="id_ed25519")
    assert "No files found" in secret.output


def test_spotlight_results_are_filtered_by_policy_and_kind(home):
    (home / "Documents" / "notes.png").write_bytes(b"png")
    found = [
        home / "Documents" / "notes.md",
        home / ".ssh" / "id_ed25519",
        home / "Documents" / "notes.png",
    ]
    spotlight = Spotlight(found)
    output = run(tools(spotlight), "mac_search_files", query="notes", kind="document").output
    assert "notes.md" in output and "id_ed25519" not in output and "notes.png" not in output
    assert spotlight.calls[0][:4] == ["mdfind", "-onlyin", str(home.resolve()), "-name"]


def test_recent_files_are_newest_first(home):
    import os
    import time

    old = home / "Downloads" / "old.txt"
    old.write_text("x")
    os.utime(old, (time.time() - 30 * 86_400,) * 2)
    output = run(tools(NoSpotlight()), "mac_recent_files", days=3).output
    assert "notes.md" in output and "old.txt" not in output


def test_reads_word_and_pdf_documents(home):
    docx = home / "Documents" / "cover.docx"
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document><w:body><w:p><w:r><w:t>Dear hiring team</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
    result = run(tools(NoSpotlight()), "mac_read_file", path=str(docx))
    assert result.success and "Dear hiring team" in result.output

    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with (home / "Documents" / "blank.pdf").open("wb") as file:
        writer.write(file)
    blank = run(tools(NoSpotlight()), "mac_read_file", path="~/Documents/blank.pdf")
    assert blank.success and "[page 1]" in blank.output


def test_long_files_are_read_in_parts(home):
    (home / "Documents" / "long.txt").write_text("a" * 3000)
    first = run(tools(), "mac_read_file", path="~/Documents/long.txt", max_characters=1000)
    assert "offset=1000" in first.output
    rest = run(tools(), "mac_read_file", path="~/Documents/long.txt", offset=2500)
    assert rest.output == "a" * 500


def test_binary_files_are_not_dumped(home):
    (home / "Documents" / "photo.bin").write_bytes(b"\x00\x01\x02" * 100)
    result = run(tools(), "mac_read_file", path="~/Documents/photo.bin")
    assert not result.success and "not a text file" in result.output


def test_copy_and_move_never_overwrite(home):
    registry = tools(NoSpotlight())
    copied = run(registry, "mac_copy_file", source="~/Documents/notes.md", destination="~/Desktop")
    assert copied.success and (home / "Desktop" / "notes.md").exists()
    again = run(registry, "mac_copy_file", source="~/Documents/notes.md", destination="~/Desktop")
    assert not again.success and "never overwrites" in again.output

    assert run(registry, "mac_create_folder", path="~/Documents/Archive").success
    moved = run(
        registry, "mac_move_file", source="~/Desktop/notes.md", destination="~/Documents/Archive"
    )
    assert moved.success and (home / "Documents" / "Archive" / "notes.md").exists()
    assert not run(registry, "mac_move_file", source="~", destination="~/Desktop").success
    leak = run(registry, "mac_copy_file", source="~/Documents/notes.md", destination="~/.ssh")
    assert not leak.success


def test_open_uses_the_fixed_open_command(home):
    spotlight = Spotlight([])
    result = run(tools(spotlight), "mac_open_file", path="~/Documents/notes.md", reveal=True)
    assert result.success
    assert spotlight.calls[-1] == ["open", "-R", str(home.resolve() / "Documents" / "notes.md")]
