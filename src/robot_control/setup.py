import os
from glob import glob
from setuptools import setup

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='pablo',
    maintainer_email='',
    description='AI-guided exploration for mecanum Jetson robot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'robot_control_node      = robot_control.robot_control_node:main',
            'camera_receiver         = robot_control.camera_receiver:main',
            'slam_receiver           = robot_control.slam_receiver:main',
            'usb_camera_node         = robot_control.usb_camera_node:main',
            'ai_navigator            = robot_control.ai_navigator:main',
            'exploration_controller  = robot_control.exploration_controller:main',
            'llm_planner             = robot_control.llm_planner:main',
        ],
    },
)
