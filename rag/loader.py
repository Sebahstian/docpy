"""Extract Symbols from installed Python libraries using the inspect module.

This is DocPy's data source: instead of scraping web docs or PDFs, we use
Python's introspection to pull signatures, docstrings, and source code from
any library installed in the current environment.

Graceful degradation: C-extension libraries (numpy, scipy core, etc.) often
don't expose signatures or source code to inspect. The loader catches these
failures and returns whatever IS available (usually the docstring).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterator

from rag.types import Symbol


class PythonDocsLoader:
    """Loads documentation Symbols from any installed Python library.

    Usage:
        loader = PythonDocsLoader()
        symbols = list(loader.load_library("requests"))
        print(f"Found {len(symbols)} symbols")
    """

    # Skip names that start with these prefixes — internal/private stuff
    SKIP_PREFIXES = ("_", "test_")

    # Skip these specific module names — they're rarely useful for users
    SKIP_MODULE_PARTS = {
        "tests",
        "_internal",
        "_vendor",
        "__pycache__",
        "f2py",
        "__main__",
        "conftest",
        "setup",
        "_build_utils",
    }

    def load_library(self, library_name: str) -> Iterator[Symbol]:
        """Yield every public Symbol in a library and its submodules.

        Args:
            library_name: A pip-installed package name like 'requests' or 'pathlib'.

        Yields:
            Symbol objects, one per function/class/method found.

        Raises:
            ImportError: if the library isn't installed.
        """
        root_module = importlib.import_module(library_name)
        yield from self._walk_module(root_module)

        # Walk all submodules too (e.g. requests.sessions, requests.auth)
        if hasattr(root_module, "__path__"):
            for module_info in pkgutil.walk_packages(
                root_module.__path__, prefix=f"{library_name}."
            ):
                if self._should_skip_module(module_info.name):
                    continue
                submodule = self._safe_import(module_info.name)
                if submodule is not None:
                    yield from self._walk_module(submodule)

    @staticmethod
    def _safe_import(module_name: str):
        """Import a submodule, suppressing stdout/stderr and swallowing all errors.

        Some submodules (e.g. numpy.f2py CLI entry points) print usage text or
        even call sys.exit() on import. We redirect output to devnull and catch
        BaseException (not just Exception) so SystemExit can't kill our walk.
        """
        import contextlib
        import io

        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return importlib.import_module(module_name)
        except BaseException:
            # Catch BaseException so SystemExit / KeyboardInterrupt from
            # misbehaving module-level code can't abort the whole load.
            return None

    def _walk_module(self, module) -> Iterator[Symbol]:
        """Extract all public functions and classes from a single module."""
        module_name = getattr(module, "__name__", "unknown")

        for name, obj in inspect.getmembers(module):
            if name.startswith(self.SKIP_PREFIXES):
                continue

            # Only keep things actually defined in this module (not re-imports)
            obj_module = getattr(obj, "__module__", None)
            if obj_module != module_name:
                continue

            if inspect.isfunction(obj):
                yield self._extract_symbol(obj, kind="function", module_name=module_name)
            elif inspect.isclass(obj):
                yield self._extract_symbol(obj, kind="class", module_name=module_name)
                # Also extract public methods of the class
                yield from self._walk_class_methods(obj, module_name)

    def _walk_class_methods(self, cls, module_name: str) -> Iterator[Symbol]:
        """Extract public methods from a class."""
        for name, method in inspect.getmembers(cls):
            if name.startswith(self.SKIP_PREFIXES):
                continue
            if not (inspect.isfunction(method) or inspect.ismethod(method)):
                continue
            # Skip methods inherited from base classes (e.g. object)
            if getattr(method, "__module__", None) != module_name:
                continue
            yield self._extract_symbol(method, kind="method", module_name=module_name)

    def _extract_symbol(self, obj, kind: str, module_name: str) -> Symbol:
        """Build a Symbol from a Python object, gracefully handling extraction failures."""
        name = getattr(obj, "__name__", "unknown")
        qualname = getattr(obj, "__qualname__", name)

        return Symbol(
            name=name,
            qualname=qualname,
            module=module_name,
            kind=kind,
            signature=self._try_signature(obj),
            docstring=self._try_docstring(obj),
            source=self._try_source(obj),
        )

    @staticmethod
    def _try_signature(obj) -> str | None:
        """Get the signature as a string, or None if inspect can't introspect it.

        C-extension functions often don't expose signatures.
        """
        try:
            return str(inspect.signature(obj))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _try_docstring(obj) -> str | None:
        """Get the cleaned docstring, or None if missing/empty."""
        doc = inspect.getdoc(obj)
        return doc if doc else None

    @staticmethod
    def _try_source(obj) -> str | None:
        """Get the Python source code, or None if unavailable.

        C-extension symbols and built-ins have no Python source.
        """
        try:
            return inspect.getsource(obj)
        except (OSError, TypeError):
            return None

    def _should_skip_module(self, module_name: str) -> bool:
        """Skip internal/test submodules."""
        parts = module_name.split(".")
        return any(part in self.SKIP_MODULE_PARTS for part in parts)
