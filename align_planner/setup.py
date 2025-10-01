from setuptools import find_packages, setup
import os
package_name = 'align_planner'



data_files=[        
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]


def package_files(data_files, directory_list):

    paths_dict = {}

    for directory in directory_list:

        for (path, directories, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)

                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),    
    data_files=package_files(data_files, ['align_planner/utils/',
                                          'align_planner/models/',
                                          'align_planner/mppi/',
                                          'align_planner/train/',
                                          'launch/',
                                          'config/',
                                          'align_planner/']),
 
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hojin Lee',
    maintainer_email='hojin@unist.ac.kr',
    description='Uncertinaty-aware traversability prediction and navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'align_planner_node = align_planner.align_planner_node:main',
            'data_logger_node = align_planner.data_logger_node:main'
        ],
    },
)

