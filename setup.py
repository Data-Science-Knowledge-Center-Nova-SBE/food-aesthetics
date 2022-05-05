from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()


setup_args = dict(
    name = "food-aesthetics",
    version = "0.0.1",
    description = "Infer aesthetics metrics from food images.",
    long_description = README,
    packages = find_packages(),
    author = 'Alessandro Gambetti'
)

if __name__ == '__main__':
    setup(
        name = "food-aesthetics",
        version = "0.0.1",
        description = "Infer aesthetics metrics from food images.",
        py_modules = ['food_aesthetics'],
        package_dir = {'':'src'},
        include_package_data = True
    )
