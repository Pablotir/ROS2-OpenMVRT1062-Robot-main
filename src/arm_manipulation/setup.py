import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'arm_manipulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pablo',
    maintainer_email='pablo@todo.todo',
    description='SO-ARM101 6-DOF robotic arm manipulation with ChArUco-calibrated eye-in-hand vision',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_driver_node = arm_manipulation.arm_driver_node:main',
            'hand_eye_tf_broadcaster = arm_manipulation.hand_eye_tf_broadcaster:main',
            'detection_node = arm_manipulation.detection_node:main',
            'pick_place_node = arm_manipulation.pick_place_node:main',
        ],
    },
)
