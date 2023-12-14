import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mimimum_housing",
    version="0.0.2",
    author="Nguyen Ngoc Bien",
    author_email="biennn1@mbbak.com.vn",
    description="Project face recogition system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https:/gitlab.mbbank.com.vn/modeling_team/biennn1/minimum_housing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.*',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
    ]
)
