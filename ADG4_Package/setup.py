from setuptools import setup, find_packages

setup(
	name = "ADG4",
	version = "0.1.0",
	author="Harvard Group 28",
	description="ADG4 library",
	packages=find_packages(),
	install_requires=["numpy", "pytest"],
	long_description="ADG4 library by group 28",
	long_description_content_type='text/markdown',
	url="https://github.com/CS107-gharvar4d/cs107-FinalProject",
	python_requires=">=3.6",
	install_requires=["numpy","pandas"]
)
