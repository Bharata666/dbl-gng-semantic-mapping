from setuptools import setup

package_name = 'dbl_gng_objects'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bharata',
    maintainer_email='bharata@todo.todo',
    description='Per-object DBL-GNG topological representation for ROS 2 semantic maps',
    license='Apache-2.0',
    entry_points={'console_scripts': ['gng_objects_node = dbl_gng_objects.gng_objects_node:main']},
)
