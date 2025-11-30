from setuptools import setup, find_packages

setup(
    name="FTIR-Gas-Density-Estimation",
    version="1.0.0",
    author="Lakshya Suri",
    author_email="lakshyasuri3@gmail.com",
    packages=find_packages(),
    package_data={
        'spectral_analyser': ['config.json']
    },
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'spectral-analyser=spectral_analyser.run:main',
        ],
    },
    python_requires='>=3.12',
    description="FTIR-Gas-Density-Estimation analyses a FTIR spectrum to identify CO2 "
                "and H2O gases and quantify them."
)
