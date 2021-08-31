from setuptools import setup, find_packages

setup(
    name='rtotools',
    version='0.1.0',    
    description='Real-Time Optimization Tools',
    url='https://github.com/vicrsp/rto',
    author='Victor Ruela',
    author_email='victorspruela@ufmg.br',
    license='MIT',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8.5",
    install_requires=['pandas',
                      'numpy',
                      'matplotlib',
                      'seaborn',
                      'sklearn',
                      'bunch',
                      'scipy',
                      'smt'
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)