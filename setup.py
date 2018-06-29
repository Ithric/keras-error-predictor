from distutils.core import setup

setup(
    name="keras-errorpred",
    version="0.0.1",
    author="Ronny Kaste",
    author_email="rkaste@gmail.com",
    packages=["errpred"],
    include_package_data=True,
    license="MIT License",
    description="Estimate the aleatoric and epistemic uncertainty of any keras model"
)