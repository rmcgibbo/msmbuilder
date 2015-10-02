echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH

if [[ "$TRAVIS_PULL_REQUEST" != "false" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi

if [[ `python -c "import sys; print(sys.version_info[:2])"` != "(2, 7)" ]]; then
    echo "No deploy on PYTHON_VERSION=${PYTHON_VERSION}"; exit 0
fi


if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi

conda install --yes anaconda-client
binstar upload -u omnia -t $BINSTAR_TOKEN --force `conda build devtools/conda-recipe --output`

# Create the docs and push them to S3
# -----------------------------------

conda install --yes pip
conda install --yes `conda build devtools/conda-recipe --output`
pip install msmb_theme s3cmd

# Install stuff for running the example IPython notebooks
conda install --yes `cat doc/requirements.txt | xargs`

cd doc && make html && cd -
cat /tmp/sphinx-*
python devtools/ci/push-docs-to-s3.py
python devtools/ci/update-versions.py
