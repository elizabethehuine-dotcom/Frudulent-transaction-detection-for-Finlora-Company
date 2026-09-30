from setuptools import setup, find_packages

setup(
    name="fraudulent_transaction_detection",
    version="0.1.0",
    author="Elizabeth Ehuine",
    author_email="elizabethehuine@gmail.com",
    # Tell setuptools exactly which folders to ignore
    packages=find_packages(exclude=["finlora_env*", "Notebook*", "Finlora_Dataset*"])
)
