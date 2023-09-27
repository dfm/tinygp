# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source under the MIT License and welcomes contributions in the form of bug
reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://github.com/dfm/tinygp)
- [Documentation](https://tinygp.readthedocs.io)
- [Issue Tracker](https://github.com/dfm/tinygp/issues)

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/dfm/tinygp/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to
reproduce the issue. In particular, please include a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to request a feature

Feel free to request features on the [Issue
Tracker](https://github.com/dfm/tinygp/issues).

## How to test the project

```bash
python -m pip install nox
python -m nox -s test -p 3.10
```

## How to submit changes

Open a [Pull Request](https://github.com/dfm/tinygp/pulls).

We use the [towncrier](https://github.com/twisted/towncrier) package to manage
release notes, so you should include a "news fragment" in the `/news` directory
describing your changes. We'll be happy to help on the pull request thread to
format that appropriately.

## Making a new release

These are the steps that need to be run when minting a new release:

1. Make sure that your local copy of main is up-to-date:

   ```bash
   git pull origin main
   ```

2. Check out a new branch:

   ```bash
   git checkout -b release
   ```

3. Update the release notes, using the version number that you're going to bump
   to (try running with `--draft` first):

   ```bash
   python -m towncrier build --version THE_NEXT_VERSION
   ```

4. Open a PR with the new release notes and make sure that all the tests pass.
5. Make a release on GitHub (it's good practice to make a release candidate
   first, just to be safe) with the appropriate version number.
