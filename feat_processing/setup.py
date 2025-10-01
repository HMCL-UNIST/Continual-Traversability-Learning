from setuptools import find_packages, setup
import os
package_name = 'feat_processing'



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
    data_files=package_files(data_files, ['feat_processing/models/', 'launch/', 'feat_processing/train/', 'feat_processing/','config/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hojin Lee',
    maintainer_email='hojin@unist.ac.kr',
    description='TODO: Package description',
    license='MIT',
    entry_points={
        'console_scripts': [
            'feat_processing_node = feat_processing.feat_processing_node:main',
            'feature_recording_node = feat_processing.feat_processing_recording_node:main',
        ],
    },
)

