# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2-beta.2](https://github.com/denehoffman/rustitude/compare/v0.1.2-beta.1...v0.1.2-beta.2) - 2023-12-29

### Other
- Update and rename main.yml to release-plz.yml
- Create main.yml
- typos in documentation
- update so doctests pass
- Release rustitude version 0.1.2-beta.1
- update documentation
- overhaul of Parameters and how evaluate calls work, amplitudes keep track of external parameters, implemented ParticleSwarm with likelihood
- remove all data from repo
- add Variantly to clear up a bunch of boilerplate, some other aesthetic changes
- more pedantic clippy suggestions, some documentation too
- add some flags for running
- linting fixes
- remove with_deps
- add debugging to release (for now)
- update .gitignore
- add bacon.toml
- Merge branch 'main' of github.com:denehoffman/rustitude
- remove main.rs from crate
- add some derives for the parameter structs, changed their name to &str and a few other modifications, also add some likelihood code
- bump version
- alias ndarray-linalg features
- Create README.md
- Create LICENSE
- Delete data.root
- Create rust.yml
- actually we can have parquets back for now
- remove data files
- Initial commit

## v0.1.3 (2023-12-29)

### New Features

 - <csr-id-545f263a714e9d8ed7fc91c7250af97275fe9738/> add Pow trait and pow function to Amplitude
 - <csr-id-0dff19617e8264c61a9c1569b06a56797c4f55d3/> allow for different amplitudes for data and MC
   This ensures we can assign weights as part of the amplitude, but users
   can choose whether they want weighted MC or not. Also makes it easy if
   your branch names differ between the two files, you only have to
   re-implement some things.
 - <csr-id-676daf37764d153e4d7c4898c9784b3243814f2b/> add Branch struct
   Branch is a convenience wrapper for getting data from the Dataset
   without duplicating or copying anything into a new variable dependency.

### Bug Fixes

 - <csr-id-2f5de7f864a35d38f9c6d612a4e3db5354b4c2fe/> doctests were taking Strings after I changed to &str

### Refactor

 - <csr-id-2c0a933b1e2861987b172cdfc81a479ced68792c/> move data extraction into dataset and propogate errors
 - <csr-id-3442057a56351b0fb3a5c53fb89df242c36a4c66/> change inputs to functions to &str

### Style

 - <csr-id-e85e1ca1fb81476ac90249b57bcdc60e22881d9a/> try out logging fit steps
 - <csr-id-f0b655bac628665bf7f1a479f3e941d0280407ae/> remove some commented-out code

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 10 commits contributed to the release.
 - 8 commits were understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Merge branch 'main' of github.com:denehoffman/rustitude ([`0e180b9`](https://github.com/denehoffman/rustitude/commit/0e180b9bfc7504d92be07ff999affa307d78861f))
    - Try out logging fit steps ([`e85e1ca`](https://github.com/denehoffman/rustitude/commit/e85e1ca1fb81476ac90249b57bcdc60e22881d9a))
    - Remove some commented-out code ([`f0b655b`](https://github.com/denehoffman/rustitude/commit/f0b655bac628665bf7f1a479f3e941d0280407ae))
    - Add Pow trait and pow function to Amplitude ([`545f263`](https://github.com/denehoffman/rustitude/commit/545f263a714e9d8ed7fc91c7250af97275fe9738))
    - Allow for different amplitudes for data and MC ([`0dff196`](https://github.com/denehoffman/rustitude/commit/0dff19617e8264c61a9c1569b06a56797c4f55d3))
    - Add Branch struct ([`676daf3`](https://github.com/denehoffman/rustitude/commit/676daf37764d153e4d7c4898c9784b3243814f2b))
    - Doctests were taking Strings after I changed to &str ([`2f5de7f`](https://github.com/denehoffman/rustitude/commit/2f5de7f864a35d38f9c6d612a4e3db5354b4c2fe))
    - Move data extraction into dataset and propogate errors ([`2c0a933`](https://github.com/denehoffman/rustitude/commit/2c0a933b1e2861987b172cdfc81a479ced68792c))
    - Change inputs to functions to &str ([`3442057`](https://github.com/denehoffman/rustitude/commit/3442057a56351b0fb3a5c53fb89df242c36a4c66))
    - Delete .github/workflows/release-plz.yml ([`7562e69`](https://github.com/denehoffman/rustitude/commit/7562e699053e758b1606391bfebcab54447b316e))
</details>

## v0.1.2-beta.2 (2023-12-29)

### Chore

 - <csr-id-8d596bf94049e0cd4327902bf63b9ae240c51a13/> release

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 6 commits contributed to the release.
 - 1 commit was understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Merge pull request #2 from denehoffman/release-plz-2023-12-29T05-39-06Z ([`432a50f`](https://github.com/denehoffman/rustitude/commit/432a50fc89bf310aca6bfbd57766d1e5f5a961ad))
    - Release ([`8d596bf`](https://github.com/denehoffman/rustitude/commit/8d596bf94049e0cd4327902bf63b9ae240c51a13))
    - Update and rename main.yml to release-plz.yml ([`d7ecaf0`](https://github.com/denehoffman/rustitude/commit/d7ecaf0fabadf92d8073eeab40be4490a8de9083))
    - Create main.yml ([`5260781`](https://github.com/denehoffman/rustitude/commit/52607811145a82e98938b0f009711cbbb5493523))
    - Typos in documentation ([`dc53009`](https://github.com/denehoffman/rustitude/commit/dc530098b4ff2dd6195e8b00fe4365d2d0d4abb8))
    - Update so doctests pass ([`f8d544b`](https://github.com/denehoffman/rustitude/commit/f8d544bd5775474bc19bb9e09ccbce78ad092eeb))
</details>

## v0.1.2-beta.1 (2023-12-29)

### Chore

 - <csr-id-0b5ace3f3d7f9c2549e2011a488bbd35d55290d9/> Release rustitude version 0.1.2-beta.1

### Commit Statistics

<csr-read-only-do-not-edit/>

 - 30 commits contributed to the release over the course of 7 calendar days.
 - 1 commit was understood as [conventional](https://www.conventionalcommits.org).
 - 0 issues like '(#ID)' were seen in commit messages

### Commit Details

<csr-read-only-do-not-edit/>

<details><summary>view details</summary>

 * **Uncategorized**
    - Release rustitude version 0.1.2-beta.1 ([`0b5ace3`](https://github.com/denehoffman/rustitude/commit/0b5ace3f3d7f9c2549e2011a488bbd35d55290d9))
    - Update documentation ([`0a4329a`](https://github.com/denehoffman/rustitude/commit/0a4329ae32d1ccc82bb439f4db019ea558e34892))
    - Overhaul of Parameters and how evaluate calls work, amplitudes keep track of external parameters, implemented ParticleSwarm with likelihood ([`16536e3`](https://github.com/denehoffman/rustitude/commit/16536e3aae353b15dc2fe6777ed656486ef82130))
    - Remove all data from repo ([`14ea8f0`](https://github.com/denehoffman/rustitude/commit/14ea8f0ef61fe4615239656f519d84806f790cf2))
    - Add Variantly to clear up a bunch of boilerplate, some other aesthetic changes ([`997a27c`](https://github.com/denehoffman/rustitude/commit/997a27c7a0242ebfd62023f07f40a288899be5f1))
    - More pedantic clippy suggestions, some documentation too ([`ff6c741`](https://github.com/denehoffman/rustitude/commit/ff6c741e4ae566cc2a31196d4dc3307f191ec211))
    - Add some flags for running ([`6d43621`](https://github.com/denehoffman/rustitude/commit/6d436211e1bba318fe61369ce3bf0d5fdbbbe312))
    - Linting fixes ([`87d9e93`](https://github.com/denehoffman/rustitude/commit/87d9e93eefe017b666502b9c6d25a6aae5eb3a52))
    - Remove with_deps ([`fd63c5b`](https://github.com/denehoffman/rustitude/commit/fd63c5baa8111de14cc1021eea063f172e743a5d))
    - Add debugging to release (for now) ([`59a361a`](https://github.com/denehoffman/rustitude/commit/59a361a94f36265606e04fcc1e2d46fb31127aca))
    - Update .gitignore ([`72384f0`](https://github.com/denehoffman/rustitude/commit/72384f0817079a4bfc2a471641715598b6f6b38e))
    - Add bacon.toml ([`bea460e`](https://github.com/denehoffman/rustitude/commit/bea460eb6e6cb6c60d6c6c6ee0bbafc9371d7189))
    - Merge branch 'main' of github.com:denehoffman/rustitude ([`cc21ca1`](https://github.com/denehoffman/rustitude/commit/cc21ca159716b313a33ea45f7d4f4809a2e1106b))
    - Remove main.rs from crate ([`79df6a6`](https://github.com/denehoffman/rustitude/commit/79df6a643ecfcb6a340740d30bce22fbf1a419e3))
    - Offload barrier factor and mass calcualtion to variables, huge speedup ([`7d99261`](https://github.com/denehoffman/rustitude/commit/7d99261c1e0eaef2f58e370d1b5e6409782b95d1))
    - Change input to evaluate_on and par_evaluate_all from vectors to slices ([`3dcd766`](https://github.com/denehoffman/rustitude/commit/3dcd7665faa015d1d0dca54798519e597df660b5))
    - Update unwrapping functions ([`bd2dd88`](https://github.com/denehoffman/rustitude/commit/bd2dd889b5b49c11f91053de197f941d450a7a60))
    - Add some upgrades to the standard library for performance (looks like parking_lot helps a lot) ([`f698a8e`](https://github.com/denehoffman/rustitude/commit/f698a8e4f198c1cb140b2217ea1f87c3a7d0b971))
    - Fix compilation on ARM Macs ([`87c0ce8`](https://github.com/denehoffman/rustitude/commit/87c0ce88b85633b9517debe832fae5eee306df77))
    - Update rust.yml ([`392afdf`](https://github.com/denehoffman/rustitude/commit/392afdf76715397f17dc0b0481be20a6a27fc7b4))
    - Add some derives for the parameter structs, changed their name to &str and a few other modifications, also add some likelihood code ([`38d4448`](https://github.com/denehoffman/rustitude/commit/38d4448cf90748342bf6b0a9985fa5fb01b0827a))
    - Bump version ([`b7e4b93`](https://github.com/denehoffman/rustitude/commit/b7e4b9319e1f26a04bdbce49a1c3a50ec11b7f87))
    - Alias ndarray-linalg features ([`7dc3c60`](https://github.com/denehoffman/rustitude/commit/7dc3c602976e2c5b0da215123f5605422c7c539e))
    - Create README.md ([`2a82b08`](https://github.com/denehoffman/rustitude/commit/2a82b08812755f8c480627a09b0bcf764f3c3a76))
    - Create LICENSE ([`2c500da`](https://github.com/denehoffman/rustitude/commit/2c500da9a9c7123c0516840ce92926a07a19135d))
    - Delete data.root ([`48f2fae`](https://github.com/denehoffman/rustitude/commit/48f2faedfdf3d5bd9050bbacc38e647edda11afc))
    - Create rust.yml ([`865837a`](https://github.com/denehoffman/rustitude/commit/865837a8ef8beab96c17a0f61b7bc95e5a0995d2))
    - Actually we can have parquets back for now ([`12d7cc4`](https://github.com/denehoffman/rustitude/commit/12d7cc43563f3201161f82f244d928d96019f193))
    - Remove data files ([`b18eb01`](https://github.com/denehoffman/rustitude/commit/b18eb0132e4d6843fd988d3826739cb6b45c9f9b))
    - Initial commit ([`ba95984`](https://github.com/denehoffman/rustitude/commit/ba959845593f7906a2b3e1be247373bf3c5e4635))
</details>

