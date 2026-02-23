from setuptools import setup
import os
from glob import glob

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.1',
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
    description='Robot control bridge + simple LLM simulator',
    license='MIT',
    entry_points={
        'console_scripts': [
            'robot_control_node = robot_control.robot_control_node:main',
            'llm_planner = robot_control.llm_planner:main',
       	    'camera_receiver = robot_control.camera_receiver:main',
        ],
    },
)
