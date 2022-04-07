import unittest

from file_utils import *


class Test(unittest.TestCase):
    def setUp(self):
        self.base_path = PurePath("test") / "file_utils_test"
        self.a = self.base_path / "a"
        self.dota = self.base_path / ".a"
        self.atxt = self.base_path / "a.txt"
        self.sub = self.base_path / "sub"
        self.suba = self.sub / "a"
        self.subatxt = self.sub / "a.txt"

    def test_deduplicate(self):
        s = deduplicate("aa", "_")
        self.assertEqual(s, "aa")

        s = deduplicate("a_a", "_")
        self.assertEqual(s, "a_a")

        s = deduplicate("a__a", "_")
        self.assertEqual(s, "a_a")

        s = deduplicate("aa_", "_")
        self.assertEqual(s, "aa_")

        s = deduplicate("aa__", "_")
        self.assertEqual(s, "aa_")

    def test_fix_delimiters(self):
        RECOGNIZED = ["_", "-"]
        OUT = "_"
        STRING = "a_b-c__d_-e-_f--g"
        RESULT_DUPE = "a_b_c__d__e__f__g"
        s = fix_delimiters(
            STRING,
            out_delimiter=OUT,
            recognized_delimiters=RECOGNIZED,
            do_deduplicate=False,
        )
        self.assertEqual(s, RESULT_DUPE)

        RESULT = "a_b_c_d_e_f_g"
        s = fix_delimiters(STRING, out_delimiter=OUT, recognized_delimiters=RECOGNIZED)
        self.assertEqual(s, RESULT)

    def test_get_contents(self):
        contents = get_contents(self.base_path)
        self.assertEqual(len(contents), 4)
        self.assertEqual(contents[0], self.dota)
        self.assertEqual(contents[1], self.a)
        self.assertEqual(contents[2], self.atxt)
        self.assertEqual(contents[3], self.sub)

        contents = get_contents(self.base_path, ext="txt")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0], self.atxt)

        contents = get_contents(self.base_path, ext=".txt")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0], self.atxt)

        contents = get_contents(self.base_path, recursive=True)
        self.assertEqual(len(contents), 6)
        self.assertEqual(contents[0], self.dota)
        self.assertEqual(contents[1], self.a)
        self.assertEqual(contents[2], self.atxt)
        self.assertEqual(contents[3], self.sub)
        self.assertEqual(contents[4], self.suba)
        self.assertEqual(contents[5], self.subatxt)

        contents = get_contents(self.base_path, ext="txt", recursive=True)
        self.assertEqual(len(contents), 2)
        self.assertEqual(contents[0], self.atxt)
        self.assertEqual(contents[1], self.subatxt)

    def test_generate_file_names(self):
        BASE_NAME = "file_1"
        RESULT = [PurePath("file_1.txt")]
        names = generate_file_names(BASE_NAME, "txt")
        self.assertEqual(names, RESULT)

        INDICES = list(range(1, 4))
        RESULT = ["file_1_1.txt", "file_1_2.txt", "file_1_3.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        names = generate_file_names(BASE_NAME, ".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        FOLDER = PurePath("folder")
        RESULT = [FOLDER / r for r in RESULT]
        names = generate_file_names(BASE_NAME, ".txt", indices=INDICES, folder=FOLDER)
        self.assertEqual(names, RESULT)

        RESULT = ["1.txt", "2.txt", "3.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        names = generate_file_names(ext=".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        FOLDER = PurePath("folder")
        RESULT = [FOLDER / r for r in RESULT]
        names = generate_file_names(ext=".txt", indices=INDICES, folder=FOLDER)
        self.assertEqual(names, RESULT)

        INDICES = list(range(1, 4))
        RESULT = ["1_file_1.txt", "2_file_1.txt", "3_file_1.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        names = generate_file_names(
            BASE_NAME, ".txt", indices=INDICES, make_indices_prefixed=True
        )
        self.assertEqual(names, RESULT)

        with self.assertRaises(ValueError):
            names = generate_file_names(ext=".txt")

    def test_get_subfolders(self):
        subs = get_subfolders(self.base_path)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0], self.sub)

    def test_lcp(self):
        self.assertEqual(lcp("interspecies", "interstellar", "interstate"), "inters")
        self.assertEqual(lcp("throne", "throne"), "throne")
        self.assertEqual(lcp("throne", "dungeon"), "")
        self.assertEqual(lcp("cheese"), "cheese")
        self.assertEqual(lcp(""), "")
        self.assertEqual(lcp("prefix", "suffix"), "")
        self.assertEqual(lcp("foo", "foobar"), "foo")

    def test_lcs(self):
        STRINGS = [
            "x!983_abcdefghij0983q580",
            "abcdefghijklmnopqrstuvwxyz",
            "_abcdefghij",
        ]
        RESULT = "abcdefghij"
        self.assertEqual(lcs(*STRINGS), RESULT)

    def test_Files(self):
        BASE_NAME = "file_1"
        ALT_BASE_NAME = "file-1"
        FOLDER = "folder"
        INDICES = list(range(1, 4))
        RESULT = ["file_1_1.txt", "file_1_2.txt", "file_1_3.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        RESULT = [FOLDER / r for r in RESULT]
        RESULT_S = ["file_1_suffix_1.txt", "file_1_suffix_2.txt", "file_1_suffix_3.txt"]
        RESULT_S = [PurePath(r) for r in RESULT_S]
        RESULT_S = [FOLDER / r for r in RESULT_S]
        RESULT_I = ["1.txt", "2.txt", "3.txt"]
        RESULT_I = [PurePath(r) for r in RESULT_I]
        RESULT_I = [FOLDER / r for r in RESULT_I]

        files = Files(FOLDER, BASE_NAME)
        files_idem = (Files(FOLDER, BASE_NAME) / FOLDER).parent
        self.assertEqual(files.root, files_idem.root)

        files = Files(FOLDER, BASE_NAME)
        self.assertEqual(files.name, BASE_NAME)

        files = Files(FOLDER, BASE_NAME) + BASE_NAME
        self.assertEqual(files.name, files.delimiter.join([BASE_NAME, BASE_NAME]))
        files.name = BASE_NAME
        self.assertEqual(files.name, BASE_NAME)

        files = Files(FOLDER, BASE_NAME) + BASE_NAME
        files.delimiter = "-"
        self.assertEqual(files.delimiter, "-")
        self.assertEqual(
            files.name, files.delimiter.join([ALT_BASE_NAME, ALT_BASE_NAME])
        )
        files.name = ALT_BASE_NAME
        self.assertEqual(files.name, ALT_BASE_NAME)

        files = Files(FOLDER, BASE_NAME)
        with self.assertRaises(AssertionError):
            files.delimiter = "x"

        files = Files(FOLDER, BASE_NAME, allowed_delimiters=["x"])
        try:
            files.delimiter = "x"
        except AssertionError:
            self.fail()

        files_no_ext = Files(FOLDER, BASE_NAME)
        names = files_no_ext.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        files_no_ext_s = Files(FOLDER, BASE_NAME) + "suffix"
        names_s = files_no_ext_s.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names_s, RESULT_S)

        files_no_ext_i = Files(FOLDER)
        names_i = files_no_ext_i.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names_i, RESULT_I)

        files_ext = Files(FOLDER, BASE_NAME, "wrong")
        names = files_ext.generate_file_names(indices=INDICES)
        for n, r in zip(names, RESULT):
            self.assertNotEqual(n, r)

        files_ext = Files(FOLDER, BASE_NAME, "wrong")
        names = files_ext.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        files_ext = Files(FOLDER, BASE_NAME, "wrong")
        files_ext.ext = "txt"
        names = files_ext.generate_file_names(indices=INDICES)
        self.assertEqual(names, RESULT)

        files_ext_s = Files(FOLDER, BASE_NAME, "wrong") + "suffix"
        names_s = files_ext_s.generate_file_names(indices=INDICES)
        for n, r in zip(names_s, RESULT):
            self.assertNotEqual(n, r)

        names_s = files_ext_s.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names_s, RESULT_S)

        files_ext_i = Files(FOLDER, "wrong")
        names_i = files_ext_i.generate_file_names(indices=INDICES)
        for n, r in zip(names_i, RESULT_I):
            self.assertNotEqual(n, r)

        names_i = files_ext_i.generate_file_names(".txt", indices=INDICES)
        for n, r in zip(names_i, RESULT_I):
            self.assertNotEqual(n, r)

        files_ext = Files("", BASE_NAME, "txt") / FOLDER
        names = files_ext.generate_file_names(indices=INDICES)
        self.assertEqual(names, RESULT)

        files_ext_s = Files("", BASE_NAME, "txt") / FOLDER
        files_ext_s = files_ext_s + "suffix"
        names_s = files_ext_s.generate_file_names(indices=INDICES)
        self.assertEqual(names_s, RESULT_S)

        files_ext_i = Files("", "txt") / FOLDER
        files_ext_i = files_ext_i + "suffix"
        names_i = files_ext_i.generate_file_names(indices=INDICES)
        for n, r in zip(names_i, RESULT_I):
            self.assertNotEqual(n, r)

        files = Files(FOLDER, BASE_NAME)
        files_copy = files / FOLDER
        names = files.generate_file_names(indices=INDICES)
        names_copy = files_copy.generate_file_names(indices=INDICES)
        for n, nc in zip(names, names_copy):
            self.assertNotEqual(n, nc)

        files = Files(FOLDER, BASE_NAME)
        files_copy = files + BASE_NAME
        names = files.generate_file_names(indices=INDICES)
        names_copy = files_copy.generate_file_names(indices=INDICES)
        for n, nc in zip(names, names_copy):
            self.assertNotEqual(n, nc)


if __name__ == "__main__":
    unittest.main()
