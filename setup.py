from setuptools import setup, find_packages

setup(
    name="plant_monitor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires='>=3.9',
    author="Team Plant Monitor",
    description="Smart Plant Health Monitoring System",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)
