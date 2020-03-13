from itertools import groupby, takewhile
from pathlib import Path, PurePath
from typing import Any, Iterable, List, Optional, Sequence, Union

from suffix_trees.STree import STree

PathLike = Union[Path, PurePath, str]


# TODO write tests
def append_suffix(path: PathLike, suffix: Union[str, None], delimiter: str = "_"):
    """Joins suffix to the end of path using delimiter.
    """
    if suffix is None:
        return path
    else:
        p = PurePath(path)
        return p.parent / (delimiter.join([p.stem, suffix]) + p.suffix)


def deduplicate(s: PathLike, delimiter: str) -> PathLike:
    """Removed consecutive duplicates of a delimiter.
    """
    T = type(s)
    out = []
    for key, group in groupby(str(s)):
        if key == delimiter:
            out.append(key)
        else:
            out.append("".join(list(group)))
    return T("".join(out))


def fix_delimiters(
    s: PathLike,
    out_delimiter: str = "_",
    recognized_delimiters: Sequence[str] = ["_", "-", " "],
    do_deduplicate: bool = True,
) -> PathLike:
    """Changes all recognized delimiters to a uniform delimiter, and optionally
    removes duplicates. Deduplication is performed by default. It is not best
    practice to modify file separators using this function. Please use the
    built-in pathlib library.

    Inputs:

    s: PathLike objects with delimiters to replace.

    out_delimiter: The delimiter desired in the output.

    recognized_delimiters: A sequence of delimiters to replace by out_delimiter.

    deduplicate: A bool indicating whether deduplication should be performed.

    Returns:

    A PathLike object of the same type as s with all matching
    recognized_delimiters replaced by out_delimiter.
    """
    T = type(s)
    s = str(s)
    for d in recognized_delimiters:
        s = s.replace(d, out_delimiter)
    if do_deduplicate:
        s = deduplicate(s, delimiter=out_delimiter)
    return T(s)


def get_contents(
    folder: PathLike, ext: Optional[PathLike] = None, recursive: bool = False
) -> List[PurePath]:
    """Returns the file contents of the supplied folder as a list of PurePath. The
    optional ext argument can be used to filter the results to a single
    extension. Can also be used recursively.
    """
    if ext is not None:
        ext = str(ext)
    glob = _create_glob(ext)
    if recursive:
        glob = str(PurePath("**") / glob)
    contents = list(Path(folder).glob(glob))
    contents = [PurePath(c) for c in contents]
    contents.sort()
    return contents


def generate_file_names(
    name: PathLike,
    ext: Optional[str] = None,
    indices: Optional[Iterable[Any]] = None,
    delimiter: str = "_",
    folder: Optional[PathLike] = None,
) -> List[PurePath]:
    """Generates a list of file names from a supplied name and indices. The
    name parts are joined by a delimiter into a base name. The indices are then
    joined to the base name by the delimiter forming a list of file names. The
    extension is appended with a dot. If a folder is supplied, it is prepended
    to the base name. The number of file names is equal to the number of
    indices. If indices is not supplied, one file name will be returned.

    NOTE: It is not recommended to use this function to join paths! Use the
    built-in pathlib module instead, and supply it as the optional folder
    instead.

    NOTE: This function does not check filenames for validity. That problem is
    really hard. See e.g. https://stackoverflow.com/q/9532499/4132985

    Arguments:

    "base_name_parts": An iterable of objects that can be converted to string
    using str(). These will be joined together using the delimiter to form a
    base name.

    "ext": A string representation of a file extension. Does not need leading
    dot (".") character.

    "indices": An iterable of objects that can be converted to string using
    str(). These will be appended to the base name to produce file names, one at
    a time. If not supplied, only one file name will be returned.

    "delimiter": A string used to join the base name parts and append the
    indices.

    "folder": A PurePath object to a folder location. This is not checked.

    Returns:

    A list of PurePath file names created by joining the folder, base name
    parts, indices (one per file name), and extension, in that order. OS-level
    validity of the resulting paths is not checked!
    """
    if ext is not None:
        ext = str(_normalize_ext(ext))
    else:
        ext = ""
    if indices is not None:
        indices = [str(i) for i in indices]
        names = [delimiter.join([name, index]) + ext for index in indices]
    else:
        names = [str(name) + ext]
    names = [PurePath(f) for f in names]
    if folder is not None:
        names = [PurePath(folder) / fn for fn in names]
    return names


def get_subfolders(folder: PathLike) -> List[PurePath]:
    subfolders = Path(folder).glob("./*")
    subfolders = [PurePath(f) for f in subfolders if f.is_dir()]
    subfolders.sort()
    return subfolders


def lcp(*s: PathLike):
    """Returns longest common prefix of input strings.

    c/o: https://rosettacode.org/wiki/Longest_common_prefix#Python:_Functional
    """
    strings = [str(x) for x in s]
    return "".join(ch[0] for ch in takewhile(lambda x: min(x) == max(x), zip(*strings)))


def lcs(*s: PathLike):
    """Returns longest common substring of input strings.
    """
    strings = [str(x) for x in s]
    return STree(strings).lcs()


def _create_glob(ext: str = None) -> str:
    if ext is None:
        return "*"
    else:
        return "*{}".format(_normalize_ext(str(ext)))


def _normalize_ext(ext: str) -> str:
    assert ext is not None
    if ext == "":
        return ext
    else:
        return "." + ext.lstrip(".")


class Files:
    """A class abstraction for the concept of files on a filesystem. Not
    intended to be comprehensive, but intended to be used for preparing file
    paths for bulk file writing.

    Inputs:

    root_folder: The root folder containing the files.

    base_name: The base name of the files to be written. Roughly equivalent to
    """

    def __init__(
        self,
        root_folder: PathLike,
        base_name: str,
        ext: Optional[str] = None,
        delimiter: str = "_",
        allowed_delimiters: Sequence[str] = ["_", "-", " "],
    ):
        self._root = PurePath(root_folder)
        self._base = base_name
        self._ext = ext
        self._delimiter = delimiter
        self._allowed_delimiters = allowed_delimiters
        self._fix_delimiters()

    @property
    def delimiter(self):
        return self._delimiter

    @delimiter.setter
    def delimiter(self, value: str):
        assert value in self._allowed_delimiters
        self._delimiter = value
        self._fix_delimiters()

    @property
    def ext(self):
        return self._ext

    @ext.setter
    def ext(self, value: Optional[str]):
        if value is not None:
            value = _normalize_ext(value)
        self._ext = value

    @property
    def name(self):
        return self._base

    @name.setter
    def name(self, value: str):
        self._base = value
        self._fix_delimiters()

    @property
    def parent(self):
        f = self._copy()
        f._root = f._root.parent
        return f

    @property
    def root(self):
        return self._root

    def __add__(self, suffix: str):
        """Appends a suffix to the base name, joined with delimiter.
        """
        f = self._copy()
        f._base = f._delimiter.join([f._base, suffix])
        return f

    def __truediv__(self, sub: PathLike):
        """Appends a directory to the root folder.
        """
        f = self._copy()
        f._root = f._root / sub
        return f

    def mkdir(self, *args, **kwargs):
        """Creates the current root dir. See Path.mkdir() for arguments.
        """
        Path(self._root).mkdir(*args, **kwargs)

    def get_subfolders(self):
        """Returns a list of PurePath of the folders contained in the root
        folder.
        """
        return get_subfolders(self._root)

    def generate_file_names(self, ext: Optional[str] = None, *args, **kwargs):
        """Generates a list of file names from a supplied name and indices. See
        documentation of free function generate_file_names(). Accepts all inputs
        except name and folder, which are supplied by the class.
        """
        if ext is None:
            ext = self._ext
        return generate_file_names(
            name=self._base,
            folder=self._root,
            ext=ext,
            delimiter=self._delimiter,
            *args,
            **kwargs
        )

    def _copy(self):
        return Files(self._root, self._base, self._ext, self._delimiter)

    def _fix_delimiters(self):
        self._base = str(
            fix_delimiters(
                self._base,
                out_delimiter=self._delimiter,
                recognized_delimiters=self._allowed_delimiters,
            )
        )
