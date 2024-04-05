<p align="center">
  <img
    width="800"
    src="https://raw.githubusercontent.com/denehoffman/rustitude/main/media/logo.png"
  />
</p>
<p align="center">
    <h1 align="center">Demystifying Amplitude Analysis</h1>
</p>

<p align="center">
  <a href="https://github.com/denehoffman/rustitude/commits/main/" alt="Lastest Commits">
    <img src="https://img.shields.io/github/last-commit/denehoffman/rustitude/main" /></a>
  <a href="https://github.com/denehoffman/rustitude/actions" alt="Build Status">
    <img src="https://img.shields.io/github/actions/workflow/status/denehoffman/rustitude/rust.yml" /></a>
  <a href="LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/denehoffman/rustitude" /></a>
  <a href="https://crates.io/crates/rustitude" alt="Rustitude on crates.io">
    <img src="https://img.shields.io/crates/v/rustitude" /></a>
  <a href="https://docs.rs/rustitude/latest/rustitude/" alt="Rustitude documentation on docs.rs">
    <img src="https://img.shields.io/docsrs/rustitude" /></a>
</p>


### Note: This project is still very much under development and not recommended for use in actual research projects (yet)

### Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Usage](#Usage)

## Overview
Amplitude analysis is the scientific process of fitting models (amplitudes) to data in order to extract additional information. In particle physics, this is often synonymous with partial wave analysis (PWA), a process commonly used to determine angular momentum quantum numbers for decaying particles. The goal of Rustitude is to establish a framework which is generic enough to fit almost any data to any model, but specific enough to be quick and easy to use in typical particle physics studies. There are three core ideas which this crate tries to follow to make fits fast and efficient:

1. Every function has the ability to be paralellized indvidually using [rayon](https://github.com/rayon-rs/rayon).
2. Model builders can separate their code efficiently into pieces which only need to be calculated once for the whole dataset (called "Resolvables") and those which depend on parameters, which could change at every evaluation (called "amplitudes"). Amplitudes implement `SNode` or `CNode` depending on their output type (real/scalar or complex) as well as several mathematical operations which allow users to join multiple amplitudes into larger models. By precalculating things which don't change, we trade RAM usage now for evaluation speed later.
3. `Dataset` structs only allow storage of scalars, vectors, and matrices (both real and complex), and Rustitude leaves it up to users to convert their data format into `Dataset`s. Unfortunately, ROOT files do not yet have well-documented R/W libraries like Python's [uproot](https://pypi.org/project/uproot/) or Julia's [UpROOT.jl](https://github.com/JuliaHEP/UpROOT.jl), so the current recommendation is to convert files to a more versatile format like parquet.

## Installation

Cargo provides the usual command for including this crate in a project:
```sh
cargo add rustitude
```

While documentation will be made available at [docs.rs](https://docs.rs/), I still haven't decoupled the core library from the executable I use for testing, so there is an OpenBLAS dependency which breaks the online documentation. However, if you clone the project,
```sh
git clone git@github.com:denehoffman/rustitude.git
```
you can build the docs yourself:
```sh
cd rustitude
cargo doc
```

## Usage
TBD
