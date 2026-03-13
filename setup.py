# setup.py

from setuptools import setup, find_packages

setup(
    name='traffic_counter',  # Your package name
    version='1.0.0',         # Version of your package
    description='A traffic counting application using YOLO11 and OpenCV',
    author='Finn Morris',      # Your name
    author_email='finn.morris00@gmail.com',  # Your email
    url='https://github.com/finnmo/traffic-counter',  # URL to your project (e.g., GitHub repo)
    packages=find_packages(),  # Automatically find packages in your project
    entry_points={
        'console_scripts': [
            'traffic-counter=traffic_counter.scripts.run:main',  # Command to run your application
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.26',
        'pandas>=2.2',
        'ultralytics>=8.3',
        'opencv-python>=4.10',
        'PyYAML>=6.0',
        'torch>=2.5',
    ],
)
