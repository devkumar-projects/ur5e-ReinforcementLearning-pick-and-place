from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur5e_rl_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'urdf'),  glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dev Kumar',
    maintainer_email='devk79036@gmail.com',
    description='RL pick-and-place for UR5e in Gazebo Harmonic',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = ur5e_rl_gazebo.train:main',
            'demo  = ur5e_rl_gazebo.demo:main',
        ],
    },
)
