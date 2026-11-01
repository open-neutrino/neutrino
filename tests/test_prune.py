"""Tests for neutrino.probe.cuda.prune on cuobjdump-style PTX.

Run with: python -m pytest tests/test_prune.py
"""

import importlib
import sys
import tempfile
import unittest


def load_cuda_probe():
    # neutrino.probe.cuda reads its work directory from sys.argv[1] at import.
    argv = sys.argv
    sys.argv = [argv[0], tempfile.mkdtemp()]
    try:
        return importlib.import_module("neutrino.probe.cuda")
    finally:
        sys.argv = argv


cuda = load_cuda_probe()

HEADER = """.version 8.5
.target sm_89
.address_size 64
"""

ENTRY = """.visible .entry kernel(
\t.param .u64 kernel_param_0
)
{
\t.reg .b64 \t%rd<2>;
\tld.param.u64 \t%rd1, [kernel_param_0];
\t{ // callseq 0, 0
\t.param .b64 param0;
\tst.param.b64 \t[param0+0], %rd1;
\tcall.uni \tdependentPayload, (param0);
\t} // callseq 0
\tret;
}
"""

DECLARATION = """.extern .func dependentPayload
(
\t.param .b64 dependentPayload_param_0
)
;
"""

BANNER = """
Fatbin ptx code:
================
arch = sm_89
code version = [8,5]
host = linux
compile_size = 64bit
compressed

"""

DEFINITION = """.visible .func dependentPayload(
\t.param .b64 dependentPayload_param_0
)
{
\t.reg .b64 \t%rd<2>;
\tld.param.u64 \t%rd1, [dependentPayload_param_0];
\t{
\tret;
\t}
}
"""


class PruneFuncTest(unittest.TestCase):
    def test_declaration_before_banner_keeps_only_definition(self):
        # The issue #23 layout: a prototype, cuobjdump framing text, then the
        # definition from another PTX image.
        ptx = HEADER + ENTRY + DECLARATION + BANNER + HEADER + DEFINITION
        _, func_section, entry_section, entry_name = cuda.prune(ptx, "kernel")
        self.assertEqual(entry_name, "kernel")
        self.assertEqual(func_section, DEFINITION[DEFINITION.index(".func"):].rstrip("\n"))
        self.assertNotIn("Fatbin ptx code", func_section)
        self.assertNotIn(".extern", func_section)
        self.assertTrue(entry_section.startswith(".visible .entry kernel("))

    def test_declaration_without_definition_is_skipped(self):
        ptx = HEADER + ENTRY + DECLARATION
        _, func_section, _, _ = cuda.prune(ptx, "kernel")
        self.assertEqual(func_section, "")

    def test_definition_is_kept(self):
        ptx = HEADER + ENTRY + DEFINITION
        _, func_section, _, _ = cuda.prune(ptx, "kernel")
        self.assertEqual(func_section, DEFINITION[DEFINITION.index(".func"):].rstrip("\n"))

    def test_definition_without_body_raises(self):
        ptx = HEADER + ENTRY + ".visible .func dependentPayload(\n\t.param .b64 p\n)\n"
        with self.assertRaises(ValueError):
            cuda.prune(ptx, "kernel")

    def test_unterminated_body_raises(self):
        ptx = HEADER + ENTRY + DEFINITION.rstrip("}\n")
        with self.assertRaises(ValueError):
            cuda.prune(ptx, "kernel")


if __name__ == "__main__":
    unittest.main()
