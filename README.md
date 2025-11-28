# shapels

jaxtyping + LSP.

**shapels** provides _shape inference_ in your editor for torch tensors.

![showcase of shapels inside helix](./assets/readme_showcase.png)


## Installation


For now, only compiling from source is supported.

The first step is to [install Rust](https://www.rust-lang.org/tools/install):

```bash
# Unix-like OS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After cloning this repository, it can be installed with [cargo](https://doc.rust-lang.org/stable/cargo/guide/cargo-home.html):

```bash
git clone https://github.com/carrascomj/shapels.git
cd shapels
cargo install --path .
```


## FAQ

* **Can I run **shapels** in [helix](https://helix-editor.com/)?**

Of course! Just add it to your `languages.toml`:

```toml
[language-server]
shapels = {command = "shapels"}

[[language]]
name = "python"
scope = "source.python"
injection-regex = "py(thon)?"
file-types = ["py", "pyi", "py3", "pyw", "ptl", "rpy", "cpy", "ipy", "pyt", { glob = ".python_history" }, { glob = ".pythonstartup" }, { glob = ".pythonrc" }, { glob = "*SConstruct" }, { glob = "*SConscript" }, { glob = "*sconstruct" }]
shebangs = ["python", "uv"]
roots = ["pyproject.toml", "setup.py", "poetry.lock", "pyrightconfig.json"]
comment-token = "#"
# WARN: probably you have other LS already here, just add shapels to the list
language-servers = ["shapels"]
indent = { tab-width = 4, unit = "    " }  
```

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

