# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

import os
import unittest
from unittest.mock import patch

from dune import packagemetadata


class EmptyBuildMetaData:
    def combine_across_modules(self, key):
        return []


class ExtractCMakeFlagsTest(unittest.TestCase):
    def extract(self, flags):
        with patch.object(packagemetadata, "_buildMetaData", EmptyBuildMetaData()):
            with patch.dict(os.environ, {"DUNE_CMAKE_FLAGS": flags}, clear=True):
                return packagemetadata._extractCMakeFlags()

    def test_flags_with_and_without_types(self):
        flags = self.extract(
            "-DUNTYPED=value "
            "-DSTRING_VALUE:STRING=typed "
            "-DBOOL_VALUE:BOOL=ON "
            "-DURL=https://example.com:8443/path"
        )

        self.assertEqual(flags["UNTYPED"], "value")
        self.assertEqual(flags["STRING_VALUE"], "typed")
        self.assertIs(flags["BOOL_VALUE"], True)
        self.assertEqual(flags["URL"], "https://example.com:8443/path")

    def test_first_colon_separates_name_and_type(self):
        flags = self.extract("-DNAME:TYPE:EXTRA=value")

        self.assertEqual(flags, {"NAME": "value"})

    def test_spaces_and_quotes(self):
        flags = self.extract(
            '-DVALUE="two words" '
            "'-DWHOLE_ARGUMENT=also two words' "
            "-DLITERAL_QUOTES='\"quoted value\"' "
            '-DEMPTY=""'
        )

        self.assertEqual(flags["VALUE"], "two words")
        self.assertEqual(flags["WHOLE_ARGUMENT"], "also two words")
        self.assertEqual(flags["LITERAL_QUOTES"], '"quoted value"')
        self.assertEqual(flags["EMPTY"], "")


if __name__ == "__main__":
    unittest.main()
