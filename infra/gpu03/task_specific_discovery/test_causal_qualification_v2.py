"""Audit-only successor regressions; inert bytes, real links, no live/model calls."""
import ast
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from infra.gpu03.task_specific_discovery import causal_qualification as original
from infra.gpu03.task_specific_discovery import causal_qualification_v2 as audit
from infra.gpu03.task_specific_discovery import test_causal_qualification as previous
from infra.gpu03.task_specific_discovery.io_contract import ContractError


class Fixture(previous.Fixture):
    def run(self):
        with mock.patch.object(audit.worker, "ADAPTERS", self.adapter_hashes), mock.patch.object(
                audit.worker, "SAMPLING", {**audit.worker.SAMPLING, "eos_token_ids": [7]}):
            return audit.audit(self.manifest_path, self.digest)

    def link_base(self):
        model = self.root / "model"
        aliases = self.root / "base_aliases"
        aliases.mkdir()
        for name in audit.worker.BASE_FILES:
            os.chmod(model / name, 0o644)
            os.link(model / name, aliases / name)
        roster = {str(model / name): self.inputs[str(model / name)] for name in audit.worker.BASE_FILES}
        manifest = json.loads(self.manifest_path.read_text())
        manifest["bound_files"].update(roster)
        self.digest = previous.write(self.manifest_path, manifest)["sha256"]
        self.release["manifest_sha256"] = self.digest
        previous.write(self.runtime / "SUPERVISOR_RESULT.json", self.release)
        return roster


class PreviousAuditSemanticsTests(previous.QualificationTests):
    """Run every original end-to-end semantic test through the v2 auditor."""
    def make(self, **kwargs):
        return Fixture(self.root, **kwargs)


@contextmanager
def authored_policy(root, roster):
    # Test-only replacement of observed path/hash/UID, never production policy.
    with mock.patch.object(audit, "BASE_ROOT", str(root)), mock.patch.object(
            audit, "BASE_ROSTER", roster), mock.patch.object(audit, "LINK_OWNER_UID", os.getuid()):
        yield


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model"; self.model.mkdir()
        self.aliases = self.root / "aliases"; self.aliases.mkdir()
        self.roster = {}
        for name in audit.worker.BASE_FILES:
            path = self.model / name
            path.write_bytes(("authored inert " + name).encode())
            os.chmod(path, 0o644)
            os.link(path, self.aliases / name)
            self.roster[str(path)] = previous.binding(path)
        self.plan = {"schema": "task_specific_causal_plan_v1", "status": "prepared_not_dispatched",
                     "model_revision": audit.worker.REVISION, "model_snapshot": str(self.model),
                     "bound_files": copy.deepcopy(self.roster)}
        self.stack = ExitStack()
        self.stack.enter_context(authored_policy(self.model, self.roster))

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def configured(self):
        evidence = audit.Evidence()
        audit.configure_from_plan(evidence, self.plan)
        return evidence

    @property
    def path(self):
        return self.model / "config.json"

    @property
    def alias(self):
        return self.aliases / "config.json"

    def test_strict_default_and_original_reject_same_real_link(self):
        for evidence in (audit.Evidence(), original.Evidence()):
            with self.subTest(evidence=type(evidence).__module__), self.assertRaises(ContractError):
                evidence.bind(self.path, self.roster[str(self.path)])

    def test_explicit_roster_all13_hashed_with_honest_provenance(self):
        evidence = self.configured()
        for path, expected in self.roster.items():
            self.assertEqual(evidence.bind(path, expected), {"path": path, **expected})
        provenance = evidence.audit_io_provenance()
        self.assertEqual(provenance["authorized_base_roster"], self.roster)
        self.assertEqual(set(provenance["verified_linked_files"]), set(self.roster))
        self.assertTrue(provenance["full_bytes_stream_hashed"])
        self.assertFalse(provenance["filesystem_immutability_claimed"])
        for value in provenance["verified_linked_files"].values():
            self.assertEqual(value["owner_uid"], os.getuid())
            self.assertEqual(value["mode"], 0o644)
            self.assertEqual(value["nlink"], 2)

    def test_implicit_expected_still_uses_pinned_hash(self):
        self.alias.write_bytes(b"different bytes through other name")
        with self.assertRaisesRegex(ContractError, "SHA differs"):
            self.configured().bind(self.path)

    def test_missing_or_conflicting_declared_binding_rejected(self):
        for replacement in (None, {"size_bytes": 1, "sha256": "0" * 64}):
            plan = copy.deepcopy(self.plan)
            if replacement is None:
                del plan["bound_files"][str(self.path)]
            else:
                plan["bound_files"][str(self.path)] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ContractError):
                audit.configure_from_plan(audit.Evidence(), plan)
        with self.assertRaisesRegex(ContractError, "expected binding"):
            self.configured().bind(self.path, {"size_bytes": 1, "sha256": "0" * 64})

    def test_nonbase_hardlinks_rejected_even_with_matching_manifest(self):
        for name in ("source.py", "proof.json", "manifest.json", "task.json", "primitive.json", "journal.jsonl"):
            path = self.root / name; path.write_bytes(b"inert")
            os.link(path, self.root / (name + ".alias"))
            with self.subTest(name=name), self.assertRaisesRegex(ContractError, "outside exact"):
                self.configured().bind(path, previous.binding(path))

    def test_wrong_owner_mode_or_third_link_rejected(self):
        with mock.patch.object(audit, "LINK_OWNER_UID", os.getuid() + 1):
            with self.assertRaises(ContractError): self.configured().bind(self.path)
        os.chmod(self.path, 0o664)
        with self.assertRaises(ContractError): self.configured().bind(self.path)
        os.chmod(self.path, 0o644)
        os.link(self.path, self.root / "third")
        with self.assertRaises(ContractError): self.configured().bind(self.path)

    def test_symlink_and_nonregular_fail_without_opening(self):
        link = self.root / "sym"; link.symlink_to(self.path)
        with self.assertRaisesRegex(ContractError, "Noncanonical"):
            self.configured().bind(link)
        with self.assertRaisesRegex(ContractError, "Nonregular"):
            self.configured().bind(self.root)
        fifo = self.root / "fifo"; os.mkfifo(fifo)
        with self.assertRaisesRegex(ContractError, "Nonregular"):
            self.configured().bind(fifo)

    def test_alias_byte_change_after_hash_rejected_on_closure(self):
        evidence = self.configured(); evidence.bind(self.path)
        self.alias.write_bytes(b"different bytes")
        with self.assertRaisesRegex(ContractError, "changed"):
            evidence.unchanged()

    def test_samepath_inode_replacement_rejected_after_hash(self):
        evidence = self.configured(); evidence.bind(self.path)
        replacement = self.root / "replacement"; replacement.write_bytes(self.path.read_bytes())
        os.chmod(replacement, 0o644)
        os.replace(replacement, self.path)
        with self.assertRaisesRegex(ContractError, "identity changed"):
            evidence.bind(self.path)

    def test_metadata_changes_after_hash_rejected(self):
        for action in (lambda: os.chmod(self.path, 0o640), lambda: os.link(self.path, self.root / "third")):
            evidence = self.configured(); evidence.bind(self.path)
            action()
            with self.assertRaisesRegex(ContractError, "changed"):
                evidence.unchanged()
            os.chmod(self.path, 0o644)
            (self.root / "third").unlink(missing_ok=True)

    def test_alias_bytes_inode_and_metadata_races_during_hash_fail(self):
        for operation in ("bytes", "inode", "mode", "links"):
            with self.subTest(operation=operation):
                # Fresh fixture per race, without mutating a previous cached proof.
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve(); path = root / "config.json"
                    path.write_bytes(b"authored data"); path.chmod(0o644)
                    alias = root / "alias"; os.link(path, alias)
                    evidence = audit.Evidence(); evidence.base_roster = {str(path): previous.binding(path)}
                    real_fdopen = os.fdopen
                    def change():
                        if operation == "bytes": alias.write_bytes(b"different bytes")
                        if operation == "inode":
                            replacement = root / "replacement"; replacement.write_bytes(path.read_bytes())
                            replacement.chmod(0o644); os.replace(replacement, path)
                        if operation == "mode": path.chmod(0o640)
                        if operation == "links": os.link(path, root / "third")
                    class Reader:
                        def __init__(self, stream): self.stream, self.done = stream, False
                        def __enter__(self): return self
                        def __exit__(self, *args): self.stream.close()
                        def read(self, *args):
                            value = self.stream.read(*args)
                            if not self.done: self.done = True; change()
                            return value
                    def fdopen(*args, **kwargs): return Reader(real_fdopen(*args, **kwargs))
                    with mock.patch.object(audit.os, "fdopen", side_effect=fdopen):
                        with self.assertRaisesRegex(ContractError, "changed during hashing"):
                            evidence.bind(path)

    def test_other_model_root_gains_no_exception_and_cannot_replace_policy(self):
        other = copy.deepcopy(self.plan); other["model_snapshot"] = str(self.aliases)
        other["bound_files"] = {str(self.aliases / p.name): previous.binding(p) for p in self.aliases.iterdir()}
        evidence = audit.Evidence(); audit.configure_from_plan(evidence, other)
        self.assertEqual(evidence.base_roster, {})
        with self.assertRaises(ContractError): evidence.bind(self.alias)
        with self.assertRaisesRegex(ContractError, "cannot be broadened"):
            audit.configure_from_plan(self.configured(), other)


class LinkedFullAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
    def tearDown(self): self.temp.cleanup()

    def test_full312_real_hardlinks_original_rejects_new_passes_unchanged_outputs(self):
        fixture = Fixture(self.root, pairs=6); roster = fixture.link_base()
        before = {str(p): previous.binding(p) for p in self.root.rglob("*") if p.is_file()}
        with authored_policy(self.root / "model", roster):
            with self.assertRaisesRegex(ContractError, "linked"):
                previous.Fixture.run(fixture)
            proof = fixture.run()
        self.assertEqual(proof["requests"], 312)
        self.assertEqual(len(proof["retained_requests"]), 312)
        self.assertTrue(proof["null_token_parity"])
        self.assertTrue(proof["null_logits_within_tolerance"])
        self.assertEqual(len(proof["audit_io_provenance"]["verified_linked_files"]), 13)
        self.assertEqual(proof["source_files"], json.loads(fixture.manifest_path.read_text())["qualification_contract"]["source_files"])
        self.assertEqual(proof["auditor_version"], audit.AUDITOR_VERSION)
        self.assertEqual(proof["auditor_sha256"], previous.binding(audit.__file__)["sha256"])
        self.assertEqual(before, {str(p): previous.binding(p) for p in self.root.rglob("*") if p.is_file()})

    def test_manifest_must_independently_bind_all13(self):
        fixture = Fixture(self.root); roster = fixture.link_base()
        manifest = json.loads(fixture.manifest_path.read_text())
        manifest["bound_files"].pop(next(iter(roster)))
        with authored_policy(self.root / "model", roster), self.assertRaisesRegex(ContractError, "omitted"):
            audit.configure_from_qualification_manifest(audit.Evidence(), manifest)

    def test_bootstrap_task_and_manifest_cannot_be_linked(self):
        fixture = Fixture(self.root); roster = fixture.link_base()
        manifest = json.loads(fixture.manifest_path.read_text())
        os.link(fixture.task_path, self.root / "task_alias")
        with authored_policy(self.root / "model", roster), self.assertRaisesRegex(ContractError, "outside exact"):
            audit.configure_from_qualification_manifest(audit.Evidence(), manifest)

    def test_linked_input_corruption_fails_before_numerical_comparison(self):
        fixture = Fixture(self.root); roster = fixture.link_base()
        (self.root / "base_aliases" / "config.json").write_bytes(b"changed")
        with authored_policy(self.root / "model", roster), self.assertRaisesRegex(ContractError, "SHA differs"):
            fixture.run()

    def test_retained_tokens_still_must_match_with_permitted_links(self):
        def mutate(request, result):
            if request["condition_id"] == "zero_strength0":
                result.update(completion_token_ids=[2, 7], generated_token_ids=[2, 7])
        fixture = Fixture(self.root, mutate_result=mutate); roster = fixture.link_base()
        with authored_policy(self.root / "model", roster), self.assertRaisesRegex(ContractError, "token/text parity"):
            fixture.run()

    def test_original_source_unchanged_and_all_numerical_helpers_exact(self):
        self.assertEqual(previous.binding(original.__file__)["sha256"], audit.ORIGINAL_AUDITOR_SHA256)
        for name in ("require", "decode", "signature", "contract", "load_tensors", "journal_records", "coord", "batches", "main"):
            self.assertEqual(inspect.getsource(getattr(original, name)), inspect.getsource(getattr(audit, name)), name)
        # The audit body has only the explicit configure call and two extra proof fields.
        old = ast.parse(inspect.getsource(original.audit)).body[0]
        new = ast.parse(inspect.getsource(audit.audit)).body[0]
        new.body = [n for n in new.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == "configure_from_qualification_manifest")]
        returned = next(n for n in ast.walk(new) if isinstance(n, ast.Return))
        pairs = [(k, v) for k, v in zip(returned.value.keys, returned.value.values)
                 if not isinstance(k, ast.Constant) or k.value not in ("auditor_version", "audit_io_provenance")]
        returned.value.keys = [k for k, _ in pairs]; returned.value.values = [v for _, v in pairs]
        self.assertEqual(ast.dump(old), ast.dump(new))


if __name__ == "__main__":
    unittest.main()
