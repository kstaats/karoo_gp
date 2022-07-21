# Releasing Karoo GP

This document describes how to create a new release of Karoo GP and
publish it on PyPI.


## Preparing for a release

- [ ] Ensure you are on `master` and that your clone is up to date:
   ```sh
   git switch master  # switch to the master branch
   git pull origin master  # pull remote changes from the main repo
   ```
- [ ] Bump the version number in `karoo_gp/__init__.py`, update the
      `RELEASE_NOTES.txt` file, and commit.
- [ ] Tag the commit with e.g. `git tag -a v2.5.0 -m 'Karoo GP 2.5.0' feb5c01`,
      using the commit id of the version bump.
- [ ] (Optional) For new major/minor releases, create a new branch:
   ```sh
   git switch -c 2.5  # create a new branch for 2.5
   ```

Note: the tag should use the `vX.Y.Z` format.  If it is an *alpha*/*beta*/*rc*
you can use e.g. `v2.5.0a0`.  Branches should only include `X.Y`.


## Setup a venv for the release

This step is optional if you don't want to create a venv or if you
already have one.

- [ ] Create the venv and activate it:
  ```sh
  python3 -m venv release
  source release/bin/activate
  ```

- [ ] Install dependencies in the venv:
  ```sh
  python3 -m pip install --upgrade pip
  python3 -m pip install --upgrade build twine
  ```

## Build the package

- [ ] Build the package in the `dist/` dir with:
  ```sh
  python3 -m build
  ```
- [ ] Check the content of the `dist/` dir:
  ```sh
  $ ls dist
  karoo_gp-2.5.0-py3-none-any.whl  karoo_gp-2.5.0.tar.gz
  ```

Note: if there are old releases in the `dist/` dir, you might want to
move them elsewhere or remove them before the next step.


## Publish on test.pypi.org

Before publishing on PyPI, it is recommended to publish on test.pypi.org.

Keep in mind that once a release has been uploaded, it can not be
replaced with a different release with the same version number.
You might want to publish one or more release candidate (e.g.
`v2.5.0rc0`) before publishing the final release.

- [ ] Upload/publish the package(s) with `twine`:
  ```sh
  python3 -m twine upload --repository testpypi dist/*
  ```
- [ ] When prompted, enter your `test.pypi.org` username and password.
- [ ] Check that the upload was successful: https://test.pypi.org/project/karoo-gp/
- [ ] Deactivate the venv using:
  ```sh
  deactivate
  ```

Note: this will upload all the packages in the `dist/` dir.  If a
package has already been uploaded, it will be ignored.


## Test the package

- [ ] Create a separate venv and activate it:
  ```sh
  python3 -m venv test_install
  cd test_install/
  source bin/activate
  ```
- [ ] Download and install the package from test.pypi.org:
  ```sh
  python3 -m pip install --index-url https://test.pypi.org/simple/ \
          --extra-index-url https://pypi.org/simple/ karoo-gp
  ```
- [ ] Test Karoo GP to make sure that everything works.
- [ ] Deactivate the venv using:
  ```sh
  deactivate
  ```


## Upload the package on PyPI

If the test was successful, you can upload the package to PyPI.

- [ ] Activate again the `release` venv:
  ```sh
  source release/bin/activate
  ```
- [ ] Double-check the content of the `dist/` dir.
- [ ] Upload/publish the package(s) with `twine`:
  ```sh
  python3 -m twine upload dist/*
  ```
- [ ] When prompted, enter your `pypi.org` username and password.
- [ ] Check that the upload was successful: https://pypi.org/project/karoo-gp/
- [ ] Deactivate the venv using:
  ```sh
  deactivate
  ```

Note: `pypi.org` and `test.pypi.org` require two separate and
independent accounts.


## Test the package

The package is now installable with `python3 -m pip install karoo-gp`.
You can test this directly on your machine, on a new venv, or in the
`test_install` venv.  If you reuse the same venv, remember to uninstall
the `test.pypi.org` version with `pip3 uninstall -y karoo-gp` first.

- [ ] Download and install the package from PyPI:
  ```sh
  python3 -m pip install karoo-gp
  ```
- [ ] Test Karoo GP to make sure that everything works.

If every works, congratulations!


## Post-release clean-up

If you haven't done it yet, you can push your changes to GitHub and
possibly delete the venv(s).

- [ ] Push the changes to GitHub:
  ```sh
  git push origin master  # push the version bump commit (or a branch)
  git push origin v2.5.0  # push the release tag
  ```
- [ ] Delete the venv(s) if you don't need them anymore:
  ```sh
  rm -rf test_install
  rm -rf release
  ```


## References

* https://packaging.python.org/en/latest/tutorials/packaging-projects/
