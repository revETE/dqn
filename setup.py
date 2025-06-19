# -*- coding: utf-8 -*
from setuptools import setup

setup(
    name="dqn",
    version="0.1.0",
    packages=["dqn"],
    package_dir={"dqn": "dqn"},
    entry_points={
        "console_scripts": [
            "dqn-train = dqn.cli:train_cli",
            "dqn-evaluate = dqn.cli:eval_cli",
        ]
    },
)
