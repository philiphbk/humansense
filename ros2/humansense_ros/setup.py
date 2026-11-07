from glob import glob

from setuptools import find_packages, setup

package_name = "humansense_ros"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HumanSense contributors",
    maintainer_email="maintainers@humansense.invalid",
    description="ROS 2 interface for HumanSense",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "humansense_node = humansense_ros.node:main",
            "video_publisher = humansense_ros.video_publisher:main",
        ],
    },
)
