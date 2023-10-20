#!/usr/bin/python3


import argparse
import subprocess
from pathlib import Path
from shutil import rmtree

from colorama import Fore, init

# init colors for all plateforms
init()

# init parser
parser = argparse.ArgumentParser(description=__doc__, usage="module_venv")


def main() -> None:
    """Launch the venv creation process."""
    # read arguments (there should be none)
    parser.parse_args()

    # welcome the user
    print(f"{Fore.LIGHTCYAN_EX}Initialization of virtual environment 10.0{Fore.RESET}")

    # create a venv folder
    print('Creating the venv directory: "module-venv"')
    venv_dir = Path.home() / "module-venv"
    venv_dir.mkdir(exist_ok=True)

    # create a venv folder associated with the current repository
    print(f'Creating a venv directory for the current app: "{Path.cwd().name}"')
    current_dir_venv = venv_dir / Path.cwd().name

    # empty the folder from anything already in there
    # equivalement to flushing the existing venv (it's just faster than pip)
    if current_dir_venv.is_dir():
        print(f"{Fore.RED}Flushing the existing venv{Fore.RESET}")
        rmtree(current_dir_venv)

    current_dir_venv.mkdir(exist_ok=True)

    # init the venv
    print(
        f"{Fore.YELLOW}Initializing the venv, this process will take some minutes.{Fore.RESET}"
    )

    subprocess.run(["python3", "-m", "venv", str(current_dir_venv)], cwd=Path.cwd())

    # extract python and pip from thevenv
    pip = current_dir_venv / "bin" / "pip"
    python3 = current_dir_venv / "bin" / "python3"

    subprocess.run([str(pip), "install", "--upgrade", "pip"], cwd=Path.cwd())

    subprocess.run([str(pip), "install", "ipykernel"], cwd=Path.cwd())

    # install all the requirements
    req = Path.cwd() / "requirements.txt"
    subprocess.run(
        [
            str(pip),
            "install",
            "--no-cache-dir",
            "-r",
            str(req),
        ],
        cwd=Path.cwd(),
    )

    # create the kernel from venv
    name = f"{Path.cwd().stem}"
    display_name = f"(test) {name}"
    subprocess.run(
        [
            str(python3),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            name,
            "--display-name",
            display_name,
        ],
        cwd=Path.cwd(),
    )

    # display last message to the end user
    print(
        f'{Fore.GREEN}The test venv have been created, it can be found in the kernel list as "{display_name} .{Fore.RESET}'
    )

    return


if __name__ == "__main__":
    main()
