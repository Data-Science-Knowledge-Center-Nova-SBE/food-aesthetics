from setuptools import setup

setup(
    name = "food-aesthetics",
    version = "0.0.1",
    description = "Infer aesthetics metrics from food images.",
    py_modules = ['food_aesthetics'],
    package_dir = {'':'src'},
    include_package_data = True

)
