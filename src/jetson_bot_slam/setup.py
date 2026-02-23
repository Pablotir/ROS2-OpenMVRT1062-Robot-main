from setuptools import setup
import os
from glob import glob

package_name = 'jetson_bot_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pablo',
    maintainer_email='robot@example.com',
    description='SLAM and autonomous exploration for mecanum Jetson robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arduino_bridge   = jetson_bot_slam.arduino_bridge_node:main',
            'mecanum_odometry = jetson_bot_slam.mecanum_odometry_node:main',
            'motor_driver     = jetson_bot_slam.motor_driver_node:main',
        ],
    },
)
