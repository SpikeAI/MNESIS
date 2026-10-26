(api-reference)=
# API Reference

Core computations live in two modules: {mod}`mnesis_chains` (pattern
generators, the `HD_SNN` network, analytical initialisation, training and inference)
and {mod}`mnesis_boilerplate` (imports, device setup, the `Params` dataclass and
utilities). Documentation is generated from the docstrings via
`sphinx.ext.autodoc`/`napoleon`.

## mnesis_chains — core network and pattern generators

```{eval-rst}
.. automodule:: mnesis_chains
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__
```

## mnesis_boilerplate — hyperparameters and utilities

```{eval-rst}
.. automodule:: mnesis_boilerplate
   :members: Params, pprint, printfig, flip_bits, stop, approx_equals,
             purge_unfinished_scans, get_scores, get_f1score, get_beta_errorbar, SpikeF1scoreLoss,
             get_cosine_schedule_with_warmup
   :undoc-members:
   :show-inheritance:
```
