from setuptools import find_packages, setup

package_name = "aic_teleoperation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="johntangz",
    maintainer_email="johntangz@intrinsic.ai",
    description="Utility scripts for AIC teleoperation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cartesian_keyboard_teleop = aic_teleoperation.cartesian_keyboard_teleop:main",
            "joint_keyboard_teleop = aic_teleoperation.joint_keyboard_teleop:main",
            "cam_test = aic_teleoperation.cam_test:main",
            "mag_test = aic_teleoperation.mag_test:main",
            "reference_pose_initializer = aic_teleoperation.reference_pose_initializer:main",
            "board_pose_estimator = aic_teleoperation.board_pose_estimator:main",
            "move2task = aic_teleoperation.move2task:main",
            "cheat_teleop = aic_teleoperation.cheatcode_tf_teleop:main",
            "insert_task = aic_teleoperation.insert:main",
            "port_finder = aic_teleoperation.port_finder:main",
            "sc_port_finder = aic_teleoperation.sc_port_finder:main",
        ],
    },
)
