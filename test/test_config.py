import unittest
import json
from pathlib import PurePath, Path

import jsonschema

from config import ConfigFile


class TestConfigFile(unittest.TestCase):
    def setUp(self):
        self.root_path = PurePath("test")
        self.init_path = self._prep_path(PurePath("test.json"))
        self.init_config = ConfigFile(self.init_path)
        self.init_config_static = ConfigFile(self.init_path)
        self.empty_config = ConfigFile()

        with open(str(self.init_path)) as f:
            self.init_json = json.load(f)

        self.save_path = self._prep_path(PurePath("save.json"))
        self.schema_path = self._prep_path(PurePath("test.schema.json"))

        self.new_key = "zzz"
        self.new_data = {"foo": "bar", "baz": ["qux"]}
        self.new_json = self.init_json.copy()
        self.new_json[self.new_key] = self.new_data

        self.list_key = "list_config"
        self.dict_key = "dict_config"
        self.list_copy_key = "list_config_copy"
        self.nested_key = "nested_dict_config"

    def _prep_path(self, path):
        t = type(path)
        return t(self.root_path) / path

    def test_CF_clear(self):
        self.init_config.clear()
        self.assertEqual(self.init_config, self.empty_config)

    def test_CF_contains(self):
        self.assertTrue(self.list_key in self.init_config)

    def test_CF_delitem(self):
        del self.init_config[self.list_key]
        self.assertFalse(self.list_key in self.init_config)

    def test_CF_eq(self):
        config_copy = ConfigFile(self.init_path)
        self.assertEqual(config_copy, self.init_config)

    def test_CF_iter(self):
        for i in self.init_config:
            self.assertTrue(True)

    def test_CF_get_path(self):
        path = self.init_config.path
        self.assertEqual(path, self.init_path)

    def test_CF_len(self):
        self.assertEqual(len(self.init_config), len(self.init_json))

    def test_CF_ne(self):
        self.assertNotEqual(self.init_config, self.empty_config)
        self.assertNotEqual(self.init_config, self.init_config[self.list_key])
        self.assertNotEqual(self.init_config, self.init_config[self.dict_key])
        self.assertNotEqual(self.init_config, {})
        self.assertNotEqual(self.init_config, [])
        self.assertNotEqual(self.init_config, ())
        self.assertNotEqual(self.init_config, "")
        self.assertNotEqual(self.init_config, 0.0)
        self.assertNotEqual(self.init_config, 0)
        self.assertNotEqual(self.init_config, None)

    def test_CF_overwrite_off(self):
        assert not Path(self.save_path).is_file()
        try:
            self.init_config.path = self.save_path
            self.init_config.overwrite_off()
            self.init_config.save()
            self.init_config[self.new_key] = self.new_data
            self.init_config.save()
            save_config = ConfigFile(self.save_path)
            self.assertEqual(save_config, self.init_config_static)
            self.assertNotEqual(save_config, self.init_config)
        finally:
            if Path(self.save_path).is_file():
                Path(self.save_path).unlink()

    def test_CF_overwrite_on(self):
        assert not Path(self.save_path).is_file()
        try:
            self.init_config.path = self.save_path
            self.init_config.overwrite_on()
            self.init_config.save()
            self.init_config[self.new_key] = self.new_data
            self.init_config.save()
            save_config = ConfigFile(self.save_path)
            self.assertNotEqual(save_config, self.init_config_static)
            self.assertEqual(save_config, self.init_config)
        finally:
            if Path(self.save_path).is_file():
                Path(self.save_path).unlink()

    def test_CF_overwrite_get_set(self):
        assert not Path(self.save_path).is_file()
        try:
            self.assertFalse(self.init_config.overwrite)
            self.init_config.path = self.save_path
            self.init_config.overwrite = True
            self.assertTrue(self.init_config.overwrite)
            self.init_config.save()
            self.init_config[self.new_key] = self.new_data
            self.init_config.save()
            save_config = ConfigFile(self.save_path)
            self.assertNotEqual(save_config, self.init_config_static)
            self.assertEqual(save_config, self.init_config)
        finally:
            if Path(self.save_path).is_file():
                Path(self.save_path).unlink()

    def test_CF_repr(self):
        self.assertEqual(self.init_config.__repr__(), self.init_json.__repr__())

    def test_CF_save(self):
        assert not Path(self.save_path).is_file()
        try:
            self.init_config.path = self.save_path
            self.init_config.save()
            save_config = ConfigFile(self.save_path)
            self.assertEqual(save_config, self.init_config)
        finally:
            if Path(self.save_path).is_file():
                Path(self.save_path).unlink()

    def test_CF_on_change_callback(self):
        key = "done"
        check = {key: False}

        def callback(x):
            check[key] = True

        self.init_config.on_change_callbacks = {"test": callback}
        self.init_config[self.new_key] = self.new_data
        self.assertTrue(check[key])

    def test_CF_schema(self):
        try:
            schema_config = ConfigFile(self.init_path, self.schema_path)
        except jsonschema.ValidationError as e:
            self.fail()

        assert not Path(self.save_path).is_file()
        try:
            schema_config.path = self.save_path
            schema_config.overwrite_off()
            schema_config.save()
            schema_config[self.new_key] = self.new_data
            schema_config.save()
        except jsonschema.ValidationError as e:
            self.fail()
        finally:
            if Path(self.save_path).is_file():
                Path(self.save_path).unlink()

    def test_CF_setitem(self):
        self.init_config[self.new_key] = self.new_data
        self.assertEqual(self.init_config.to_json(), self.new_json)

    def test_CF_set_path(self):
        self.init_config.path = self.save_path
        path = self.init_config.path
        self.assertEqual(path, self.save_path)

    def test_CF_to_json(self):
        config = self.init_config.to_json()
        self.assertEqual(config, self.init_json)

    def test_CL_append(self):
        self.init_config[self.list_key].append(self.new_data)
        check = self.init_config[self.list_copy_key].to_json()
        check.append(self.new_data)
        self.assertEqual(self.init_config[self.list_key], check)

    def test_CL_clear(self):
        self.init_config[self.list_key].clear()
        self.assertEqual(self.init_config[self.list_key].to_json(), [])

    def test_CF_contains(self):
        self.assertTrue(
            self.init_config[self.list_key][0] in self.init_config[self.list_key]
        )
        not_exists = max(self.init_config[self.list_key]) + 1
        self.assertFalse(not_exists in self.init_config[self.list_key])

    def test_CF_getattr(self):
        check = self.init_config[self.dict_key]
        value = getattr(self.init_config, self.dict_key)
        self.assertEqual(value, check)

    def test_CF_setattr(self):
        setattr(self.init_config, self.new_key, self.new_data)
        value = self.init_config[self.new_key]
        self.assertEqual(value, self.new_data)

    def test_CL_count(self):
        exists = self.init_config[self.list_key][0]
        self.assertEqual(self.init_config[self.list_key].count(exists), 1)
        not_exists = max(self.init_config[self.list_key]) + 1
        self.assertEqual(self.init_config[self.list_key].count(not_exists), 0)

    def test_CL_delitem(self):
        item = self.init_config[self.list_key][0]
        del self.init_config[self.list_key][0]
        self.assertFalse(item in self.init_config[self.list_key])

    def test_CL_eq(self):
        self.assertEqual(
            self.init_config[self.list_key], self.init_config[self.list_copy_key]
        )
        self.assertEqual(self.init_config[self.list_key], self.init_json[self.list_key])

    def test_CL_extend(self):
        self.init_config[self.list_key].extend(self.new_data)
        check = self.init_config[self.list_copy_key].to_json()
        check.extend(self.new_data)
        self.assertEqual(self.init_config[self.list_key], check)

    def test_CL_index(self):
        first = self.init_config[self.list_key][0]
        self.assertEqual(self.init_config[self.list_key].index(first), 0)
        self.assertEqual(self.init_config[self.list_key].index(first, 0, -1), 0)
        with self.assertRaises(ValueError):
            self.init_config[self.list_key].index(first, 1, -1)
        not_exists = max(self.init_config[self.list_key]) + 1
        with self.assertRaises(ValueError):
            self.init_config[self.list_key].index(not_exists)

    def test_CL_insert(self):
        self.init_config[self.list_key].insert(0, self.new_data)
        check = self.init_config[self.list_copy_key].to_json()
        check.insert(0, self.new_data)
        self.assertEqual(self.init_config[self.list_key], check)

    def test_CL_len(self):
        self.assertEqual(len(self.init_config), len(self.init_json))

    def test_CL_ne(self):
        self.assertNotEqual(
            self.init_config[self.list_key],
            self.init_config[self.nested_key][self.list_key],
        )
        self.assertNotEqual(
            self.init_config[self.list_key],
            self.init_json[self.nested_key][self.list_key],
        )
        self.assertNotEqual(self.init_config[self.list_key], self.init_config)
        self.assertNotEqual(self.init_config[self.list_key], {})
        self.assertNotEqual(self.init_config[self.list_key], [])
        self.assertNotEqual(self.init_config[self.list_key], ())
        self.assertNotEqual(self.init_config[self.list_key], "")
        self.assertNotEqual(self.init_config[self.list_key], 0.0)
        self.assertNotEqual(self.init_config[self.list_key], 0)
        self.assertNotEqual(self.init_config[self.list_key], None)

    def test_CL_pop(self):
        original = self.init_config[self.list_key].to_json()
        popped = self.init_config[self.list_key].pop()
        self.assertEqual(popped, original[-1])
        original = self.init_config[self.list_copy_key].to_json()
        popped = self.init_config[self.list_copy_key].pop(0)
        self.assertEqual(popped, original[0])

    def test_CL_remove(self):
        first = self.init_config[self.list_key][0]
        self.init_config[self.list_key].remove(first)
        check = self.init_config[self.list_copy_key].to_json()
        check.remove(first)
        self.assertEqual(self.init_config[self.list_key], check)

    def test_CL_reverse(self):
        self.init_config[self.list_key].reverse()
        check = self.init_config[self.list_copy_key].to_json()
        check.reverse()
        self.assertEqual(self.init_config[self.list_key], check)

    def test_CL_setitem(self):
        self.init_config[self.list_key][0], self.init_config[self.list_key][1] = (
            self.init_config[self.list_key][1],
            self.init_config[self.list_key][0],
        )
        self.init_config[self.list_key].sort()
        self.assertEqual(
            self.init_config[self.list_key], self.init_config[self.list_copy_key]
        )

    def test_CL_sort(self):
        self.init_config[self.list_key].sort(key=lambda x: x, reverse=True)
        check = self.init_config[self.list_copy_key].to_json()
        check.sort(key=lambda x: x, reverse=True)
        self.assertEqual(self.init_config[self.list_key], check)

    def test_CD_clear(self):
        self.init_config[self.dict_key].clear()
        self.assertEqual(self.init_config[self.dict_key].to_json(), {})

    def test_CD_contains(self):
        exists = list(self.init_config[self.dict_key].keys())[0]
        self.assertTrue(exists in self.init_config[self.dict_key])
        not_exists = ""
        self.assertFalse(not_exists in self.init_config[self.dict_key])

    def test_CD_delitem(self):
        key = list(self.init_config[self.dict_key].keys())[0]
        del self.init_config[self.dict_key][key]
        self.assertFalse(key in self.init_config[self.list_key])

    def test_CD_eq(self):
        copy = self.init_config[self.dict_key].to_json()
        self.assertEqual(copy, self.init_config[self.dict_key])

    def test_CD_fromkeys(self):
        value = self.init_config[self.dict_key].fromkeys([1])
        self.assertEqual(value, {1: None})
        value = self.init_config[self.dict_key].fromkeys([1], 0)
        self.assertEqual(value, {1: 0})

    def test_CD_get(self):
        key = list(self.init_config[self.dict_key].keys())[0]
        value = self.init_config[self.dict_key].get(key)
        self.assertEqual(value, self.init_config[self.dict_key].to_json()[key])
        not_exists = ""
        value = self.init_config[self.dict_key].get(not_exists)
        self.assertIsNone(value)
        value = self.init_config[self.dict_key].get(not_exists, 0)
        self.assertEqual(value, 0)

    def test_CD_items(self):
        items = self.init_config[self.dict_key].items()
        check = self.init_config[self.dict_key].to_json().items()
        self.assertEqual(list(items), list(check))

    def test_CD_keys(self):
        keys = self.init_config[self.dict_key].keys()
        check = self.init_config[self.dict_key].to_json().keys()
        self.assertEqual(list(keys), list(check))

    def test_CD_list(self):
        value = list(self.init_config[self.dict_key])
        check = list(self.init_config[self.dict_key].to_json().keys())
        self.assertEqual(value, check)

    def test_CD_ne(self):
        self.assertNotEqual(
            self.init_config[self.dict_key],
            self.init_config[self.nested_key][self.dict_key],
        )
        self.assertNotEqual(
            self.init_config[self.dict_key],
            self.init_json[self.nested_key][self.dict_key],
        )
        self.assertNotEqual(self.init_config[self.dict_key], self.init_config)
        self.assertNotEqual(self.init_config[self.dict_key], {})
        self.assertNotEqual(self.init_config[self.dict_key], [])
        self.assertNotEqual(self.init_config[self.dict_key], ())
        self.assertNotEqual(self.init_config[self.dict_key], "")
        self.assertNotEqual(self.init_config[self.dict_key], 0.0)
        self.assertNotEqual(self.init_config[self.dict_key], 0)
        self.assertNotEqual(self.init_config[self.dict_key], None)

    def test_CD_pop(self):
        key = list(self.init_config[self.dict_key].keys())[0]
        popped = self.init_config[self.dict_key].pop(key)
        self.assertEqual(popped, self.init_json[self.dict_key][key])
        not_exists = ""
        popped = self.init_config[self.dict_key].get(not_exists)
        self.assertIsNone(popped)
        popped = self.init_config[self.dict_key].get(not_exists, 0)
        self.assertEqual(popped, 0)

    def test_CD_popitem(self):
        item = list(self.init_config[self.dict_key].items())[-1]
        popped = self.init_config[self.dict_key].popitem()
        self.assertEqual(popped, list(self.init_json[self.dict_key].items())[-1])

    def test_CD_reversed(self):
        check = list(self.init_config[self.dict_key].keys())
        check.reverse()
        self.assertEqual(list(reversed(self.init_config[self.dict_key])), check)

    def test_CD_setdefault(self):
        self.init_config[self.dict_key].setdefault(self.new_key, self.new_data)
        self.assertEqual(self.init_config[self.dict_key][self.new_key], self.new_data)
        not_exists = ""
        self.init_config[self.dict_key].setdefault(not_exists)
        self.assertIsNone(self.init_config[self.dict_key][not_exists])
        self.init_config[self.dict_key].pop(not_exists)
        self.init_config[self.dict_key].setdefault(not_exists, 0)
        self.assertEqual(self.init_config[self.dict_key][not_exists], 0)

    def test_CD_setitem(self):
        self.init_config[self.dict_key][self.new_key] = self.new_data
        self.assertEqual(self.init_config[self.dict_key][self.new_key], self.new_data)

    def test_CD_update(self):
        check = self.init_config[self.dict_key].to_json()
        check.update({self.new_key: self.new_data})
        self.init_config[self.dict_key].update({self.new_key: self.new_data})
        self.assertEqual(self.init_config[self.dict_key], check)

    def test_CD_values(self):
        values = self.init_config[self.dict_key].values()
        check = self.init_config[self.dict_key].to_json().values()
        self.assertEqual(list(values), list(check))

    def test_CD_getattr(self):
        key = list(self.init_config[self.dict_key].keys())[0]
        check = self.init_config[self.dict_key][key]
        value = getattr(self.init_config[self.dict_key], key)
        self.assertEqual(value, check)

    def test_CD_setattr(self):
        setattr(self.init_config[self.dict_key], self.new_key, self.new_data)
        value = self.init_config[self.dict_key][self.new_key]
        self.assertEqual(value, self.new_data)


if __name__ == "__main__":
    unittest.main()
