from setuptools import setup, find_packages

install_requires = [
    "economicsl@git+https://github.com/ox-inet-resilience/py-economicsl@master",
    "resilience@git+https://github.com/ox-inet-resilience/resilience@master",
]

setup(
    name="sw_stresstest",
    version="0.1",
    description="System-wide stress testing",
    url="https://github.com/ox-inet-resilience/sw_stresstest",
    keywords="abm stress-testing",
    author="INET Oxford",
    author_email="rhtbot@protonmail.com",
    license="Apache",
    packages=find_packages(where="src"),
    setup_requires=["setuptools>=42.0"],
    install_requires=install_requires,
    zip_safe=False,
)
