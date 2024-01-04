import platform
from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def comparison(session: nox.Session) -> None:
    session.install(".[test,comparison]")
    session.run("pytest", *session.posargs, env={"JAX_ENABLE_X64": "1"})


@nox.session(python=PYTHON_VERSIONS)
def doctest(session: nox.Session) -> None:
    if platform.system() == "Windows":
        module = Path(session.virtualenv.location) / "Lib" / "site-packages" / "tinygp"
    else:
        module = (
            Path(session.virtualenv.location)
            / "lib"
            / f"python{session.python}"
            / "site-packages"
            / "tinygp"
        )
    session.install(".[test]", "numpyro")
    session.run("pytest", "--doctest-modules", "-v", str(module), *session.posargs)
