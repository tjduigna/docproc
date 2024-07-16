
from __future__ import annotations

import io
import logging
from argparse import ArgumentParser
from os import environ
from pathlib import Path
from shutil import which
from subprocess import PIPE, STDOUT, Popen, check_output
from sys import argv as sys_argv
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("docproc")


class Proc(Popen):
    """
    An opinionated Popen with an execute method
    to mirror communicate that handles input as strings,
    supports early return, and logs as the process runs.
    Also retain stdout and stderr as a list of strings
    for downstream access.
    """

    def __init__(
        self,
        *args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            text=text,
            **kwargs,
        )
        self._stderr = stderr
        # squelch the static type checkers that
        # see these pipes as None since we're using
        # PIPEs for everything
        self.stdin: io.TextIOWrapper
        self.stdout: io.TextIOWrapper
        self.stderr: io.TextIOWrapper

    def execute(
        self,
        *,
        inputs: Optional[str] = None,
    ) -> None:
        """
        Like Popen.communicate but tail output

        Parameters
        ----------
        inputs : str
            a command to pass to the subprocess

        """
        if isinstance(self.args, list):
            LOG.info(" ".join(map(str, self.args)))
        else:
            LOG.info(self.args)

        streams = [("stdout", "stdout"), ("stderr", "stderr")]
        if self._stderr == STDOUT:
            streams = [("stdout", "stream")]
        with self:
            try:
                if inputs is not None:
                    self.stdin.write(inputs)
                    self.stdin.close()
                for attr, stream in streams:
                    for line in getattr(self, attr):
                        LOG.info(f"{stream}: {line.rstrip()}")
            except Exception:
                self.kill()

def get_dev_tag() -> str:
    """
    Get the current git describe which includes base image tag
    and whether or not the repository is in a dirty (uncommitted)
    state.

    Returns
    -------
    tag : str
        current git describe
    """
    return check_output(
        [
            "git",
            "describe",
            "--tags",
            "--always",
            "--dirty",
            "--abbrev=8",
        ],
        text=True,
    ).strip()


def get_image() -> str:
    """
    Minimize the diff if any of this changes
    """
    url = "ghcr.io"
    project = "tjduigna"
    image_name = "docproc"
    return f"{url}/{project}/{image_name}"


def get_docker() -> str:
    """
    Sanity check for docker in environment
    """
    docker = which("docker")
    if docker is None:
        raise ValueError("could not find docker executable!")
    return docker


def get_env(tag: str | None = None) -> dict[str, str]:
    """
    Get the build env that simulates docker compose

    Parameters
    ----------
    tag : str, default=None
        the tag to use instead of the current dev tag

    Returns
    -------
    env : dict[str, str]
        the build environment
    """
    image = get_image()
    image_repo = image.replace("/docproc", "")
    dev_tag = get_dev_tag()
    base_tag = dev_tag.split("-")[0]
    build_tag = tag or dev_tag
    LOG.info(f"get_env: base_tag={base_tag} build_tag={build_tag}")
    return {
        **environ,
        **{
            "BASE_TAG": base_tag,
            "IMAGE_REPO": image_repo,
            "BUILD_TAG": build_tag,
        },
    }


def get_index_url() -> str:
    return "https://pypi.org/simple"


def pull_base_image(build: bool = False, push: bool = False) -> None:
    """
    Make sure the base image is up to date. Optionally
    push the image tag to the configured registry.

    Parameters
    ----------
    push : bool, default=False
        if True, push the tagged image
    """
    docker = get_docker()
    env = get_env()
    cmd = [
        docker,
        "pull",
        env["IMAGE_REPO"] + "/docproc-base",
    ]
    Proc(cmd, env=env).execute()
    if build:
        env["INDEX_URL"] = get_index_url()
        cmd = [
            docker,
            "build",
            "-f",
            "dockerfiles/base/Dockerfile",
            "-t",
            f"{env['IMAGE_REPO']}/docproc-base:{env['BASE_TAG']}",
            "--secret",
            "id=INDEX_URL",
            "--build-arg",
            f"BASE_TAG={env['BASE_TAG']}",
            (Path.cwd() / "dockerfiles/base").as_posix(),
        ]
        Proc(cmd, env=env).execute()
    if push:
        cmd = [
            docker,
            "push",
            f"{env['IMAGE_REPO']}/docproc-base:{env['BASE_TAG']}",
        ]
        Proc(cmd, env=env).execute()


def main(argv: Optional[List[str]] = None):
    """
    The main entrypoint for the docker helper

    Parameters
    ----------
    argv : List[str], default=None
        the command line arguments
    """
    if argv is None:
        argv = sys_argv[1:]
    parser = ArgumentParser(prog="docproc docker helper")
    subs = parser.add_subparsers(help="subcommand", dest="command")
    pull = subs.add_parser("pull", help="Pull the base image")
    pull.add_argument(
        "--push",
        default=False,
        action="store_true",
        help="Push the image tag",
    )
    pull.add_argument(
        "--build",
        default=False,
        action="store_true",
        help="Build the image before running",
    )
    ns, _ = parser.parse_known_args()
    nsargs = vars(ns)
    command = nsargs.pop("command")
    if command == "pull":
        pull_base_image(**nsargs)


if __name__ == "__main__":
    main()
