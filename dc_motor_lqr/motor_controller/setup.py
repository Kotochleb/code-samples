from setuptools import find_packages
from setuptools import setup

package_name = 'motor_controller'

setup(
    name=package_name,
    version='0.20.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Krzysztof Wojciechowski',
    author_email='krzy.wojciecho@gmail.com',
    maintainer='Krzysztof Wojciechowski',
    maintainer_email='krzy.wojciecho@gmail.com',
    keywords=['ROS'],
    classifiers=[
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python',
    ],
    description=(
        'Motor controller written in python for a torque control'
    ),
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'logger = motor_controller.logger:main',
        ],
    },
)