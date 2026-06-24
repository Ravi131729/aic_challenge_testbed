from setuptools import find_packages, setup

package_name = "nic_policy_node"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ravi",
    maintainer_email="ravi@example.com",
    description="NIC insertion policy for the AIC policy API.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [],
    },
)
