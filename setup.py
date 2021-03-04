from setuptools import setup, find_packages


def install_deps():
    """Reads requirements.txt and preprocess it
    to be feed into setuptools.

    This is the only possible way (we found)
    how requirements.txt can be reused in setup.py
    using dependencies from private github repositories.

    Links must be appendend by `-{StringWithAtLeastOneNumber}`
    or something like that, so e.g. `-9231` works as well as
    `1.1.0`. This is ignored by the setuptools, but has to be there.

    Warnings:
        to make pip respect the links, you have to use
        `--process-dependency-links` switch. So e.g.:
        `pip install --process-dependency-links {git-url}`

    Returns:
         list of packages and dependency links.
    """
    default = open('requirements.txt', 'r').readlines()
    new_pkgs = []
    links = []
    for resource in default:
        if 'git+ssh' in resource:
            pkg = resource.split('#')[-1]
            links.append(resource.strip() + '-9876543210')
            new_pkgs.append(pkg.replace('egg=', '').rstrip())
        else:
            new_pkgs.append(resource.strip())
    return new_pkgs, links

pkgs, new_links = install_deps()

setup(
    name='awesome_align',
    install_requires=pkgs,
    dependency_links=new_links,
    version='0.1.0',
    license='BSD 3-Clause',
    description='An awesome word alignment tool',
    packages=find_packages(
        include = ['awesome_align', 'awesome_align.*'],
        exclude = ['run_align.py', 'run_train.py']
    ),
    entry_points={
        "console_scripts": [
            "awesome-align=awesome_align.run_align:main",
            "awesome-train=awesome_align.run_train:main",
        ],
    },
    url='https://github.com/neulab/awesome-align',
)
