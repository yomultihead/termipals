from setuptools import setup, find_packages

install_requires = [
    "transformers",
    "accelerate",
    "tqdm",
    "python-dotenv",
    "requests", # Added requests
    "protobuf", # Added protobuf
]

setup(
    name="termipals",
    version="0.1.0",
    description="Add some joy to your terminal with friendly ASCII art animals!",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "termipals=termipals.cli.main:main",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "termipals": ["assets/animals/*.txt"],
    },
) 