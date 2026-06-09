"""Build doxygen documentation and copy into Sphinx output."""

import os
import shutil
import subprocess
import pathlib
import sys
from sphinx.util import logging

from .utils import on_rtd


def generate_doxygen(app):
    """Build doxygen documentation if needed."""
    if is_generate_doxygen(app):
        build_doxygen(app)


def get_src_dest_paths(app) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Get (source, destination) paths for user and developer doxygen HTML."""
    conf_dir = pathlib.Path(app.confdir)
    output_dir = pathlib.Path(app.builder.outdir)

    return [
        (  # USER documentation
            (conf_dir / "../doxygen/html").resolve().absolute(),
            (output_dir / "doxygen").absolute(),
        ),
        (  # DEVELOPER documentation
            (conf_dir / "../doxygen_dev/html").resolve().absolute(),
            (output_dir / "doxygen_dev").absolute(),
        ),
    ]


def is_generate_doxygen(app) -> bool:
    """Check if doxygen should be built."""
    logger = logging.getLogger(__name__)

    if on_rtd():
        logger.info("Doxygen: create because we are on read the docs.")
        return True

    if not shutil.which("doxygen"):
        logger.warning(
            "Doxygen: could not find 'doxygen' executable. Skip building doxygen documentation."
        )
        return False

    for _, dest in get_src_dest_paths(app):
        if not dest.exists():
            logger.info(f"Doxygen: build because {dest} does not exist.")
            return True

    if "ALPAKAV_DOC_DOXYGEN" in os.environ:
        env_value = os.environ["ALPAKAV_DOC_DOXYGEN"]
        if env_value in ("1", "ON"):
            logger.info(
                f"Doxygen: force build via environment variable ALPAKAV_DOC_DOXYGEN={env_value}"
            )
            return True
        if env_value in ("0", "OFF"):
            logger.info(
                f"Doxygen: disable build via environment variable ALPAKAV_DOC_DOXYGEN={env_value}"
            )
            return False

        logger.error(
            f"Doxygen: unknown value for environment variable ALPAKAV_DOC_DOXYGEN={env_value}"
        )
        sys.exit(1)

    return True


def build_doxygen(app):
    """Run doxygen and copy rendered HTML into Sphinx output."""
    docs_dir = pathlib.Path(app.confdir).parent

    logger = logging.getLogger(__name__)
    for cmd in (["doxygen"], ["doxygen", "Doxyfile_dev"]):
        logger.info(f"Run {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=docs_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            logger.error(f"{' '.join(cmd)} failed:\n{result.stderr}")
            sys.exit(result.returncode)

    for src, dest in get_src_dest_paths(app):
        logger.info(f"copy from {src}\n       to {dest}")
        if src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            logger.error(f"Doxygen HTML not found at: {src}")
            sys.exit(1)
