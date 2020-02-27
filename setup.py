import setuptools

setuptools.setup(
    name="rl_reuben",  # Replace with your own username
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    packages=['games', 'rl_utils'],
    install_requires=['torch', 'gym'],
    python_requires='>=3.6',
)
