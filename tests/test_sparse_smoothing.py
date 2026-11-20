from pathlib import Path

import anndata
import numpy as np
import scipy.sparse as sp

import cospar as cs
from cospar.tmap import _utils


def test_diffusion_applies_pruning_at_the_original_global_density(
    tmp_path, monkeypatch
):
    n_cells = 40
    cells = np.arange(n_cells)
    adjacency = sp.csr_matrix(
        (
            np.ones(2 * n_cells),
            (
                np.tile(cells, 2),
                np.concatenate(((cells - 1) % n_cells, (cells + 1) % n_cells)),
            ),
        ),
        shape=(n_cells, n_cells),
    )
    adata = anndata.AnnData(shape=(n_cells, 1))
    adata.obsp["connectivities"] = adjacency
    monkeypatch.setattr(
        _utils,
        "sparse_row_blocks",
        lambda shape: ((i, min(i + 6, shape[0])) for i in range(0, shape[0], 6)),
    )

    threshold = 0.35
    prefix = str(tmp_path / "diffusion")
    result = _utils.generate_similarity_matrix(
        adata,
        prefix,
        round_of_smooth=4,
        save_subset=False,
        truncation_threshold=threshold,
        use_existing_KNN_graph=True,
    )
    expected = np.eye(n_cells)
    for depth in range(1, 5):
        expected = 0.1 * expected + 0.45 * adjacency @ expected
        if np.count_nonzero(expected > 0) / expected.size >= 0.1:
            expected[expected < threshold * expected.max(axis=1, keepdims=True)] = 0
        np.testing.assert_allclose(
            sp.load_npz(f"{prefix}_SM{depth}.npz").toarray(), expected
        )
    assert sp.isspmatrix_csr(result)


def test_selected_similarity_cache_uses_selected_cell_count(tmp_path, monkeypatch):
    adata = anndata.read_h5ad(
        Path(__file__).parent / "data" / "test_adata_preprocessed.h5ad"
    )
    monkeypatch.setattr(cs.settings, "data_path", str(tmp_path))
    options = {
        "clonal_time_points": ["4", "6"],
        "use_full_Smatrix": False,
        "smooth_array": [5],
        "max_iter_N": 1,
    }
    first = cs.tmap.infer_Tmap_from_multitime_clones(adata, **options)
    cached = cs.tmap.infer_Tmap_from_multitime_clones(adata, **options)
    assert first.n_obs < adata.n_obs
    for key in ("transition_map", "intraclone_transition_map"):
        assert sp.isspmatrix_csr(first.uns[key])
        assert first.uns[key].nnz > 0
        assert (first.uns[key] != cached.uns[key]).nnz == 0


def test_later_time_point_accepts_default_clonal_time_points(tmp_path, monkeypatch):
    adata = anndata.read_h5ad(
        Path(__file__).parent / "data" / "test_adata_preprocessed.h5ad"
    )
    monkeypatch.setattr(cs.settings, "data_path", str(tmp_path))
    options = {"later_time_point": "6", "smooth_array": [0], "max_iter_N": 1}
    time_points = ["2", "4", "6"]

    inferred = cs.tmap.infer_Tmap_from_multitime_clones(adata, **options)
    explicit = cs.tmap.infer_Tmap_from_multitime_clones(
        adata, clonal_time_points=time_points, **options
    )

    assert time_points == ["2", "4", "6"]
    for key in ("transition_map", "intraclone_transition_map"):
        assert inferred.uns[key].nnz > 0
        assert (inferred.uns[key] != explicit.uns[key]).nnz == 0
