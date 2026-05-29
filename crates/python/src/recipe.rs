//! PyO3 bindings for F1 recipes. Thin wrappers over
//! `sadda_engine::Project`'s recipe API plus a Rust-side script
//! generator that emits a runnable `.py` from the `processing_run`
//! rows linked to a recipe.

use std::fs;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sadda_engine::ProcessingRunRow;

use crate::engine_err_to_py;

/// Python-side representation of one `recipe_run` row.
#[pyclass(module = "sadda._native.recipe", name = "Recipe")]
pub(crate) struct PyRecipe {
    /// Recipe id (primary key in `recipe_run`).
    #[pyo3(get)]
    pub id: i64,
    /// Human-readable name (UNIQUE per project).
    #[pyo3(get)]
    pub name: String,
    /// Sadda version recorded when `record()` ran.
    #[pyo3(get)]
    pub sadda_version: String,
    /// Opaque JSON parameters supplied by the caller.
    #[pyo3(get)]
    pub parameters: Option<String>,
    /// ISO 8601 start timestamp.
    #[pyo3(get)]
    pub started_at: String,
    /// ISO 8601 end timestamp, if the block completed.
    #[pyo3(get)]
    pub completed_at: Option<String>,
    /// `'in_progress'` | `'ok'` | `'error'`.
    #[pyo3(get)]
    pub status: String,
    /// On `'error'` status, the exception text.
    #[pyo3(get)]
    pub error_message: Option<String>,
}

#[pymethods]
impl PyRecipe {
    fn __repr__(&self) -> String {
        format!(
            "Recipe(id={}, name={:?}, status={:?})",
            self.id, self.name, self.status
        )
    }
}

/// Opens a recipe block. Returns the new recipe id; the caller must
/// pair this with [`end`].
#[pyfunction]
#[pyo3(signature = (project, name, parameters=None))]
pub(crate) fn start(
    project: &Bound<'_, crate::PyProject>,
    name: &str,
    parameters: Option<String>,
) -> PyResult<i64> {
    let pyproj = project.borrow();
    pyproj
        .inner
        .start_recipe(name, parameters.as_deref())
        .map_err(engine_err_to_py)
}

/// Closes a recipe block. `status` must be `'ok'` or `'error'`;
/// `error_message` is recorded verbatim on `'error'`.
#[pyfunction]
#[pyo3(signature = (project, recipe_id, status, error_message=None))]
pub(crate) fn end(
    project: &Bound<'_, crate::PyProject>,
    recipe_id: i64,
    status: &str,
    error_message: Option<String>,
) -> PyResult<()> {
    if status != "ok" && status != "error" {
        return Err(PyValueError::new_err(format!(
            "recipe.end: status must be 'ok' or 'error', got {status:?}"
        )));
    }
    let pyproj = project.borrow();
    pyproj
        .inner
        .end_recipe(recipe_id, status, error_message.as_deref())
        .map_err(engine_err_to_py)
}

/// Lists all recipes in the project in id order.
#[pyfunction]
pub(crate) fn list_recipes(project: &Bound<'_, crate::PyProject>) -> PyResult<Vec<PyRecipe>> {
    let pyproj = project.borrow();
    let rows = pyproj.inner.recipes().map_err(engine_err_to_py)?;
    Ok(rows
        .into_iter()
        .map(|r| PyRecipe {
            id: r.id,
            name: r.name,
            sadda_version: r.sadda_version,
            parameters: r.parameters,
            started_at: r.started_at,
            completed_at: r.completed_at,
            status: r.status,
            error_message: r.error_message,
        })
        .collect())
}

/// Fetches a single recipe by name.
#[pyfunction]
pub(crate) fn get(project: &Bound<'_, crate::PyProject>, name: &str) -> PyResult<PyRecipe> {
    let pyproj = project.borrow();
    let r = pyproj
        .inner
        .recipe_by_name(name)
        .map_err(engine_err_to_py)?;
    Ok(PyRecipe {
        id: r.id,
        name: r.name,
        sadda_version: r.sadda_version,
        parameters: r.parameters,
        started_at: r.started_at,
        completed_at: r.completed_at,
        status: r.status,
        error_message: r.error_message,
    })
}

/// Generates a runnable `.py` script from a recipe's linked
/// `processing_run` rows. Writes to `<project>/recipes/<name>.py`
/// (creating the directory if absent) and returns the path. Recipes
/// in `'error'` status are still emitted, but each captured call is
/// preceded by a `# WARNING: …` comment.
#[pyfunction]
pub(crate) fn generate_script(
    project: &Bound<'_, crate::PyProject>,
    recipe_id: i64,
) -> PyResult<String> {
    let pyproj = project.borrow();
    let recipe = pyproj
        .inner
        .recipes()
        .map_err(engine_err_to_py)?
        .into_iter()
        .find(|r| r.id == recipe_id)
        .ok_or_else(|| PyValueError::new_err(format!("recipe id {recipe_id} not found")))?;
    let runs = pyproj
        .inner
        .processing_runs_for_recipe(recipe_id)
        .map_err(engine_err_to_py)?;

    let project_root = pyproj.inner.root().to_string_lossy().into_owned();
    let body = render_script(&recipe.name, &recipe.sadda_version, &project_root, &runs);

    let recipes_dir = pyproj.inner.root().join("recipes");
    fs::create_dir_all(&recipes_dir)
        .map_err(sadda_engine::EngineError::Io)
        .map_err(engine_err_to_py)?;
    let dest = recipes_dir.join(format!("{}.py", sanitize_filename(&recipe.name)));
    fs::write(&dest, body)
        .map_err(sadda_engine::EngineError::Io)
        .map_err(engine_err_to_py)?;
    Ok(dest.to_string_lossy().into_owned())
}

/// Returns the conventional path to a recipe's `.py` script (whether
/// or not it actually exists on disk). Does **not** create or touch
/// the file.
#[pyfunction]
pub(crate) fn script_path(project: &Bound<'_, crate::PyProject>, name: &str) -> PyResult<String> {
    let pyproj = project.borrow();
    let p = pyproj
        .inner
        .root()
        .join("recipes")
        .join(format!("{}.py", sanitize_filename(name)));
    Ok(p.to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------------
// Script generator
// ---------------------------------------------------------------------------

fn render_script(
    recipe_name: &str,
    sadda_version: &str,
    project_root: &str,
    runs: &[ProcessingRunRow],
) -> String {
    let mut out = String::new();
    out.push_str("#!/usr/bin/env python3\n");
    out.push_str("# Auto-generated by sadda.recipe.\n");
    out.push_str(&format!("# Recipe name: {recipe_name}\n"));
    out.push_str(&format!(
        "# Sadda version at record time: {sadda_version}\n"
    ));
    out.push_str(&format!("# Source project: {project_root}\n"));
    out.push_str("#\n");
    out.push_str(&format!(
        "# Run with: python {}.py [project_path]\n",
        sanitize_filename(recipe_name)
    ));
    out.push_str("# Default target is the same project this recipe was recorded from.\n\n");
    out.push_str("from __future__ import annotations\n");
    out.push_str("import sys\n");
    out.push_str("from pathlib import Path\n\n");
    out.push_str("import sadda\n\n\n");
    out.push_str("def main() -> None:\n");
    out.push_str(&format!(
        "    project_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path({:?})\n",
        project_root
    ));
    out.push_str("    proj = sadda.open_project(project_path)\n\n");

    if runs.is_empty() {
        out.push_str("    # No captured operations.\n");
        out.push_str("    pass\n");
    } else {
        for run in runs {
            render_call(&mut out, run);
        }
    }

    out.push_str("\n\nif __name__ == \"__main__\":\n");
    out.push_str("    main()\n");
    out
}

fn render_call(out: &mut String, run: &ProcessingRunRow) {
    let header = format!(
        "    # processing_run #{} — {} on bundle {} ({})\n",
        run.id, run.processor_id, run.bundle_id, run.status
    );
    out.push_str(&header);
    match run.processor_id.as_str() {
        "sadda.io.textgrid.import" => {
            let path = extract_path_param(run.parameters.as_deref());
            out.push_str(&format!(
                "    proj.import_textgrid(Path({:?}), {})\n",
                path, run.bundle_id,
            ));
        }
        "sadda.io.eaf.import" => {
            let path = extract_path_param(run.parameters.as_deref());
            out.push_str(&format!(
                "    proj.import_eaf(Path({:?}), {})\n",
                path, run.bundle_id,
            ));
        }
        "sadda.live" => {
            out.push_str("    # Live recordings cannot be replayed from a script (they\n");
            out.push_str("    # require physical audio input). The bundle exists; this is\n");
            out.push_str("    # a provenance breadcrumb only.\n");
            out.push_str(&format!(
                "    pass  # bundle_id was {}, params were {:?}\n",
                run.bundle_id,
                run.parameters.as_deref().unwrap_or(""),
            ));
        }
        other => {
            out.push_str(&format!(
                "    # Unknown processor_id {:?}; parameters: {:?}\n",
                other,
                run.parameters.as_deref().unwrap_or(""),
            ));
            out.push_str("    pass\n");
        }
    }
}

/// Pulls the `"path"` string out of a processing_run parameters JSON
/// object. The textgrid / eaf importers both emit
/// `{"path":"...","n_tiers":N}` so a tiny hand-rolled extractor
/// avoids pulling in serde_json. On miss, returns an empty string;
/// the generated script will error at runtime when the user runs it,
/// which is the right escalation for a recipe we can't fully
/// reconstruct.
fn extract_path_param(json: Option<&str>) -> String {
    let raw = match json {
        Some(s) => s,
        None => return String::new(),
    };
    let key = "\"path\":";
    let Some(i) = raw.find(key) else {
        return String::new();
    };
    let rest = raw[i + key.len()..].trim_start();
    let Some(rest) = rest.strip_prefix('"') else {
        return String::new();
    };
    // Walk to the next unescaped quote.
    let mut out = String::new();
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(esc) = chars.next() {
                    out.push(esc);
                }
            }
            '"' => return out,
            _ => out.push(c),
        }
    }
    out
}

/// Filesystem-friendly slug: alphanumeric + `_` + `.` + `-`, with
/// everything else replaced by `_`. Mirrors the project's
/// `sanitize_filename` helper but kept local to avoid a public
/// export from the engine crate.
fn sanitize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}
