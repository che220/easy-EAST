from setuptools import setup, find_packages

setup(
    # Package information
    name='text_detection_east',
    version='1.0.0',

    # Package data
    packages=find_packages(),
    include_package_data=True,
    url='',

    # Insert dependencies list here
    install_requires=[
        'boto3=1.9.120',
        'sagemaker=1.18.7',
        's3fs=0.2.0',
        'pandas=0.24.0',
        'scipy=1.2.1',
        'scikit-learn=0.20.3',
        'tensorflow-gpu=1.12.0',
        'opencv-python=4.0.0.21',
        'numpy=1.16.2',
        'imutils=0.5.2'
    ]
)
