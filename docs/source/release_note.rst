Release notes
-------------

v0.6.0
''''''

- Reduce peak memory for transition-map inference. CoSpar computes graph
  diffusion and transition-map smoothing in sparse row blocks, and keeps the
  similarity and clone-coupling matrices sparse. It applies the existing
  pruning and convergence criteria.
- Fix similarity-cache validation with ``use_full_Smatrix=False``, which
  builds the graph on the selected cells. Fix inference when
  ``later_time_point`` is provided and ``clonal_time_points`` uses its default.
- Add the ``color_bar_label`` option to ``cospar.pl.barcode_heatmap`` and render
  heatmaps with rasterized cells for sharper exports.
- Accept the optional ``format`` keyword in the Newick tree constructor for
  compatibility with existing calls.

On a 70,000-cell sample from ``cospar.datasets.hematopoiesis_130K()``, inference
with the default ``[15, 10, 5]`` smoothing schedule used 17.54 GiB peak resident
memory on a Mac with 32 GiB RAM. On a 5,000-cell sample, peak memory fell from
3.11 to 1.31 GiB. The transition and intraclone transition maps retained
identical nonzero positions and agreed within 1.2e-16 in absolute value.


v0.5.0
''''''

- Expand support for newer Python versions and update compatibility with
  recent NumPy and SciPy APIs.
- Use Matplotlib for gene-expression trajectory plots and remove the
  Plotnine and scikit-misc dependencies.
- Refresh the dependency lockfile and update CI to retrieve test fixtures
  through Git LFS.


v0.3.1
''''''
- Update cospar.tl.fate_coupling. The coupling score now account for sampling heterogeneity between cell types (so that each row adds to 1). Besides, each clone represents an independent measurement and therefore should contribute equally to the coupling calculation, we then normalize the matrix per column to calculate within each clone the fraction of cells in each cell type (clone normalization).

- Add cospar.tl.pvalue_for_fate_coupling


v0.2.1
''''''

Major changes from v0.1.8 to v0.2.1:
    - Split each plotting function into two parts: computing the results (stored at cospar.tl.**) and actually plotting the result (stored at cospar.pl.**).
    - Update the notebooks to accomodate these changes.
    - Update the datasets in the cloud to add more annotations.
    - Re-organize the content of the plot, tool, and tmap modules.
    - Fix stochasticity when running HighVar method to generate the initialized map.
    - Fix generating X_clone from the cell_id-by-barcode_id list.
    - Add a few more functions: :func:`cospar.pl.clonal_fates_across_time`, :func:`cospar.pl.clonal_reports`,  :func:`cospar.pl.embedding_genes`, :func:`cospar.tl.fate_biased_clones`
    - Update :func:`cospar.pl.barcode_heatmap` to order clones in a better way
    - Fix the docs.
    - Adopt "Raise ValueError" method for error handling.
    - Unify error checking at the beginning of several functions.

v0.1.8
''''''

This is used in running the notebooks that generate figures for the published paper. To run the original notebooks, you should switch to this version.
