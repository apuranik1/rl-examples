from setuptools import setup

setup(
    name="rl-examples",
    version="0.1.0",
    author="Alok Puranik",
    install_requires=[
        "numpy~=1.16",
    ],
    extras_require={"test": ["black==19.3b0", "mypy==0.711", "flake8~=3.7"]},
)
